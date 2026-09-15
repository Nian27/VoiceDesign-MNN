# P5.5d: capture FULLY self-consistent official frame-0 state in ONE run, then
# replay torch + MNN with those EXACT inputs and compare vs official hidden_out[0].
# This closes the question: is the 0.922 due to stale g4b kv62, or a real GraphB bug?
import os, json, numpy as np, torch, MNN, functools, traceback
os.environ['HF_HUB_OFFLINE'] = '1'
D = r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/device-smoke'
B = r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/device-smoke/g4b_boundary'
OFF = D + '/p55_official'
OUT = D + '/p55d_frame0'
os.makedirs(OUT, exist_ok=True)
LOG = open(OUT + '/iso.log', 'w')
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

rec = {'frame0': None}
orig_base = base.forward
@functools.wraps(orig_base)
def patched_base(*args, **kwargs):
    emb = kwargs.get('inputs_embeds', args[0] if args else None)
    if emb is not None and emb.shape[1] == 1 and rec['frame0'] is None:
        pkv = kwargs.get('past_key_values')
        kk = torch.stack([pkv.layers[i].keys for i in range(28)], 0) if hasattr(pkv, 'layers') else None
        vv = torch.stack([pkv.layers[i].values for i in range(28)], 0) if hasattr(pkv, 'layers') else None
        rec['frame0'] = {
            'emb': emb.detach().float().cpu().numpy().astype(np.float32),
            'pos': kwargs.get('position_ids').detach().float().cpu().numpy().astype(np.float32),
            'cp': kwargs.get('cache_position').detach().long().cpu().numpy().astype(np.int64),
            'kv_k': kk.detach().float().cpu().numpy().astype(np.float32),
            'kv_v': vv.detach().float().cpu().numpy().astype(np.float32),
        }
    return orig_base(*args, **kwargs)
base.forward = patched_base

orig_talker = talker.forward
@functools.wraps(orig_talker)
def patched_talker(*args, **kwargs):
    emb = kwargs.get('inputs_embeds', args[0] if args else None)
    is_prefill = emb is not None and emb.shape[1] > 1
    if not is_prefill and rec['frame0'] is not None and 'input_ids' not in rec['frame0']:
        iid = kwargs.get('input_ids')
        ph = kwargs.get('past_hidden')
        rec['frame0']['input_ids'] = iid.detach().long().cpu().numpy().astype(np.int64) if iid is not None else None
        rec['frame0']['past_hidden'] = ph.detach().float().cpu().numpy().astype(np.float32) if ph is not None else None
    return orig_talker(*args, **kwargs)
talker.forward = patched_talker

texts = ['小家伙，修炼之事急不得，根基若是不稳，日后吃亏的还是你自己。']
instructs = ['苍老男性，低沉略沙哑，声音有岁月感，语气沉稳、有威严，但并不凶狠。']
wavs, sr = gw.generate_voice_design(text=texts, instruct=instructs, language='Chinese', non_streaming_mode=True)
talker.forward = orig_talker
base.forward = orig_base

f0 = rec['frame0']
log('captured frame0: cp=%s input_ids=%s kv_len=%d emb %s pos %s past_hidden %s' % (
    f0['cp'].tolist(), f0['input_ids'].tolist() if f0['input_ids'] is not None else None,
    f0['kv_k'].shape[3], f0['emb'].shape, f0['pos'].shape, f0['past_hidden'].shape))
np.savez(OUT + '/frame0.npz', emb=f0['emb'], pos=f0['pos'], cp=f0['cp'], kv_k=f0['kv_k'], kv_v=f0['kv_v'], input_ids=f0['input_ids'], past_hidden=f0['past_hidden'])

# compare with stale g4b kv62
kvk62 = np.load(B + '/kv_k_decode.npy').astype(np.float32)
log('cos(this-run kv62, g4b kv62) = %.7f' % cos_fn(f0['kv_k'], kvk62))

# torch replay with EXACT official frame0 inputs
from transformers.cache_utils import DynamicCache
kv = DynamicCache()
for i in range(28):
    kv.update(torch.from_numpy(f0['kv_k'][i]), torch.from_numpy(f0['kv_v'][i]), i)
cp0 = int(f0['cp'][0])
with torch.inference_mode():
    out = base(inputs_embeds=torch.from_numpy(f0['emb']), position_ids=torch.from_numpy(f0['pos']),
               past_key_values=kv, cache_position=torch.tensor([cp0]), use_cache=True, output_hidden_states=True)
h_torch = out.last_hidden_state.detach().float().numpy().astype(np.float32)

traj = np.load(OFF + '/official_traj.npz')
h_off = traj['hidden_out'][0].astype(np.float32)
log('cos(torch replay [this-run frame0], official hidden) = %.7f' % cos_fn(h_torch, h_off))

# MNN with EXACT official frame0 inputs
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
kvk[:,:,:,:62,:] = f0['kv_k'][:,:,:,:62,:]
kvv[:,:,:,:62,:] = f0['kv_v'][:,:,:,:62,:]
g_feed('inputs_embeds', f0['emb'], MNN.Halide_Type_Float)
g_feed('position_ids', f0['pos'], MNN.Halide_Type_Float)
g_feed('cache_position', np.array([cp0], dtype=np.int32), MNN.Halide_Type_Int)
g_feed('kv_k', kvk, MNN.Halide_Type_Float)
g_feed('kv_v', kvv, MNN.Halide_Type_Float)
g.runSession(gs)
h_mnn = g_pull('hidden_states')

log('')
log('torch vs MNN        = %.7f' % cos_fn(h_torch, h_mnn))
log('MNN vs official     = %.7f' % cos_fn(h_mnn, h_off))
log('torch vs official   = %.7f' % cos_fn(h_torch, h_off))
log('torch maxabs %.3f  MNN maxabs %.3f  official maxabs %.3f' % (
    float(np.abs(h_torch).max()), float(np.abs(h_mnn).max()), float(np.abs(h_off).max())))
print('P55D_DONE torch_vs_off=%.7f mnn_vs_off=%.7f torch_vs_mnn=%.7f kv_stale_cos=%.7f' % (
    cos_fn(h_torch, h_off), cos_fn(h_mnn, h_off), cos_fn(h_torch, h_mnn), cos_fn(f0['kv_k'], kvk62)), flush=True)
