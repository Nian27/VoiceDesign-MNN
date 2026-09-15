# P5.5c: isolate GraphB hidden divergence — hook base.model (inner transformer) to
# capture the ACTUAL frame-0 inputs_embeds/position_ids used in the decode step.
import os, json, numpy as np, torch, MNN, functools, traceback
os.environ['HF_HUB_OFFLINE'] = '1'
D = r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/device-smoke'
B = r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/device-smoke/g4b_boundary'
OFF = D + '/p55_official'
LOG = open(D + '/p55c_graphb_iso.log', 'w')
def log(s):
    LOG.write(s + '\n'); LOG.flush()
def cos_fn(a, b):
    a = np.asarray(a, dtype=np.float32).ravel(); b = np.asarray(b, dtype=np.float32).ravel()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
log('loading...')
from qwen_tts import Qwen3TTSModel
gw = Qwen3TTSModel.from_pretrained(r'E:/AndroidStudioProjects/qwen3tts-mnn/models/qwen3-tts/Qwen3-TTS-12Hz-1.7B-VoiceDesign', dtype=torch.float32, device_map='cpu')
talker = gw.model.talker
base = talker.model
log('LOADED')

rec = {'n_base_calls': 0, 'frame0_emb': None, 'frame0_pos': None, 'frame0_cp': None, 'frame0_past_hidden': None, 'frame0_input_ids': None}
orig_base = base.forward
@functools.wraps(orig_base)
def patched_base(*args, **kwargs):
    rec['n_base_calls'] += 1
    emb = kwargs.get('inputs_embeds', args[0] if args else None)
    if emb is not None and emb.shape[1] == 1 and rec['frame0_emb'] is None:
        rec['frame0_emb'] = emb.detach().float().cpu().numpy().astype(np.float32)
        rec['frame0_pos'] = kwargs.get('position_ids').detach().float().cpu().numpy().astype(np.float32)
        rec['frame0_cp'] = kwargs.get('cache_position').detach().long().cpu().numpy().astype(np.int64)
    return orig_base(*args, **kwargs)
base.forward = patched_base

orig_talker = talker.forward
@functools.wraps(orig_talker)
def patched_talker(*args, **kwargs):
    emb = kwargs.get('inputs_embeds', args[0] if args else None)
    is_prefill = emb is not None and emb.shape[1] > 1
    if not is_prefill and rec['frame0_input_ids'] is None:
        iid = kwargs.get('input_ids')
        rec['frame0_input_ids'] = iid.detach().long().cpu().numpy().astype(np.int64) if iid is not None else None
        ph = kwargs.get('past_hidden')
        rec['frame0_past_hidden'] = ph.detach().float().cpu().numpy().astype(np.float32) if ph is not None else None
    return orig_talker(*args, **kwargs)
talker.forward = patched_talker

texts = ['小家伙，修炼之事急不得，根基若是不稳，日后吃亏的还是你自己。']
instructs = ['苍老男性，低沉略沙哑，声音有岁月感，语气沉稳、有威严，但并不凶狠。']
wavs, sr = gw.generate_voice_design(text=texts, instruct=instructs, language='Chinese', non_streaming_mode=True)
talker.forward = orig_talker
base.forward = orig_base
log('base calls ' + str(rec['n_base_calls']) + ' frame0 cp=' + str(rec['frame0_cp'].tolist()) + ' input_ids=' + str(rec['frame0_input_ids'].tolist() if rec['frame0_input_ids'] is not None else None))
log('official frame0 emb shape ' + str(rec['frame0_emb'].shape) + ' pos ' + str(rec['frame0_pos'].shape))
log('official frame0 past_hidden shape ' + str(rec['frame0_past_hidden'].shape))

emb0 = np.load(B + '/step_decode_inputs.npz')['emb'].astype(np.float32)
pos0 = np.load(B + '/step_decode_inputs.npz')['pos'].astype(np.float32)
log('')
log('cos(g4b emb0, official frame0 emb) = %.7f' % cos_fn(emb0, rec['frame0_emb']))
log('g4b emb0 maxabs %.3f  official emb maxabs %.3f' % (float(np.abs(emb0).max()), float(np.abs(rec['frame0_emb']).max())))
log('g4b pos0 = ' + str(pos0.ravel().tolist()))
log('official pos = ' + str(rec['frame0_pos'].ravel().tolist()))

# torch replay with OFFICIAL frame0 emb
from transformers.cache_utils import DynamicCache
kvk62 = np.load(B + '/kv_k_decode.npy').astype(np.float32)
kvv62 = np.load(B + '/kv_v_decode.npy').astype(np.float32)
kv = DynamicCache()
for i in range(28):
    kv.update(torch.from_numpy(kvk62[i]), torch.from_numpy(kvv62[i]), i)
cp0 = int(rec['frame0_cp'][0])
pos_t = torch.from_numpy(rec['frame0_pos'])
emb_t = torch.from_numpy(rec['frame0_emb'])
with torch.inference_mode():
    out = base(inputs_embeds=emb_t, position_ids=pos_t, past_key_values=kv, cache_position=torch.tensor([cp0]), use_cache=True, output_hidden_states=True)
h_torch = out.last_hidden_state.detach().float().numpy().astype(np.float32)

traj = np.load(OFF + '/official_traj.npz')
h_off = traj['hidden_out'][0].astype(np.float32)
log('cos(torch replay [official emb], official hidden) = %.7f' % cos_fn(h_torch, h_off))

# MNN with OFFICIAL frame0 emb
g = MNN.Interpreter(D + '/graphb_deploy_hand_fp16.mnn')
g.setExternalFile(D + '/graphb_deploy_hand_fp16.mnn.weight')
gs = g.createSession()
def g_feed(name, arr, dt):
    t = g.getSessionInput(gs, name)
    ht = MNN.Tensor(tuple(arr.shape), dt, arr.ravel(), MNN.Tensor_DimensionType_Caffe)
    t.copyFromHostTensor(ht)
def g_pull(name):
    t = g.getSessionOutput(gs, name)
    osh = [d if d > 0 else 1 for d in t.getShape()]
    ho = MNN.Tensor(osh, MNN.Halide_Type_Float, MNN.Tensor_DimensionType_Caffe)
    t.copyToHostTensor(ho)
    return np.asarray(ho.getNumpyData()).reshape(osh).astype(np.float32)
MAX_KV = 768
kvk = np.zeros((28,1,8,MAX_KV,128), dtype=np.float32)
kvv = np.zeros((28,1,8,MAX_KV,128), dtype=np.float32)
kvk[:,:,:,:62,:] = kvk62[:,:,:,:62,:]
kvv[:,:,:,:62,:] = kvv62[:,:,:,:62,:]
g_feed('inputs_embeds', rec['frame0_emb'], MNN.Halide_Type_Float)
g_feed('position_ids', rec['frame0_pos'], MNN.Halide_Type_Float)
g_feed('cache_position', np.array([cp0], dtype=np.int32), MNN.Halide_Type_Int)
g_feed('kv_k', kvk, MNN.Halide_Type_Float)
g_feed('kv_v', kvv, MNN.Halide_Type_Float)
g.runSession(gs)
h_mnn = g_pull('hidden_states')

log('')
log('cos(torch replay, MNN) = %.7f' % cos_fn(h_torch, h_mnn))
log('cos(MNN, official hidden) = %.7f' % cos_fn(h_mnn, h_off))
log('cos(torch replay, official) = %.7f' % cos_fn(h_torch, h_off))
log('torch maxabs %.3f  MNN maxabs %.3f  official maxabs %.3f' % (
    float(np.abs(h_torch).max()), float(np.abs(h_mnn).max()), float(np.abs(h_off).max())))
print('P55C_DONE torch_vs_off=%.7f mnn_vs_off=%.7f torch_vs_mnn=%.7f' % (
    cos_fn(h_torch, h_off), cos_fn(h_mnn, h_off), cos_fn(h_torch, h_mnn)), flush=True)
