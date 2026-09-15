# P5.5f: FINAL self-consistent verification — in ONE run, capture frame0 emb/kv/pos AND
# frame0 hidden_out (from base.forward output). Then replay base with those exact inputs.
# If replay==captured (cos~1.0), GraphB is provably correct and all prior 0.9x were cross-run sampling noise.
import os, json, numpy as np, torch, MNN, functools, traceback
os.environ['HF_HUB_OFFLINE'] = '1'
D = r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/device-smoke'
OUT = D + '/p55f_same_run'
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
    out = orig_base(*args, **kwargs)
    if emb is not None and emb.shape[1] == 1 and rec['frame0'] is not None and 'hidden_out' not in rec['frame0']:
        rec['frame0']['hidden_out'] = out.last_hidden_state.detach().float().cpu().numpy().astype(np.float32)
    return out
base.forward = patched_base

texts = ['小家伙，修炼之事急不得，根基若是不稳，日后吃亏的还是你自己。']
instructs = ['苍老男性，低沉略沙哑，声音有岁月感，语气沉稳、有威严，但并不凶狠。']
wavs, sr = gw.generate_voice_design(text=texts, instruct=instructs, language='Chinese', non_streaming_mode=True)
base.forward = orig_base
f0 = rec['frame0']
log('captured: cp=%s kv_len=%d emb %s hidden_out %s' % (f0['cp'].tolist(), f0['kv_k'].shape[3], f0['emb'].shape, f0['hidden_out'].shape))
np.savez(OUT + '/frame0.npz', emb=f0['emb'], pos=f0['pos'], cp=f0['cp'], kv_k=f0['kv_k'], kv_v=f0['kv_v'], hidden_out=f0['hidden_out'])

# replay base with EXACT same-run inputs
from transformers.cache_utils import DynamicCache
kv = DynamicCache()
for i in range(28):
    kv.update(torch.from_numpy(f0['kv_k'][i]), torch.from_numpy(f0['kv_v'][i]), i)
cp0 = int(f0['cp'][0])
with torch.inference_mode():
    out = base(inputs_embeds=torch.from_numpy(f0['emb']), position_ids=torch.from_numpy(f0['pos']),
               past_key_values=kv, cache_position=torch.tensor([cp0]), use_cache=True, output_hidden_states=True)
h_replay = out.last_hidden_state.detach().float().numpy().astype(np.float32)

cos_rr = cos_fn(h_replay, f0['hidden_out'])
log('')
log('SAME-RUN replay vs captured hidden = %.7f' % cos_rr)
log('replay maxabs %.3f  captured maxabs %.3f' % (float(np.abs(h_replay).max()), float(np.abs(f0['hidden_out']).max())))
print('P55F_DONE same_run_replay_cos=%.7f' % cos_rr, flush=True)
