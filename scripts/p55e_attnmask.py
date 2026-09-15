# P5.5e: close the loop — capture frame0 attention_mask + rope_deltas, replay with them.
# Hypothesis: torch replay (no attention_mask) != official (has attention_mask) explains 0.915.
import os, json, numpy as np, torch, MNN, functools, traceback
os.environ['HF_HUB_OFFLINE'] = '1'
D = r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/device-smoke'
OFF = D + '/p55_official'
OUT = D + '/p55e_attnmask'
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
        am = kwargs.get('attention_mask')
        rec['frame0'] = {
            'emb': emb.detach().float().cpu().numpy().astype(np.float32),
            'pos': kwargs.get('position_ids').detach().float().cpu().numpy().astype(np.float32),
            'cp': kwargs.get('cache_position').detach().long().cpu().numpy().astype(np.int64),
            'kv_k': kk.detach().float().cpu().numpy().astype(np.float32),
            'kv_v': vv.detach().float().cpu().numpy().astype(np.float32),
            'attention_mask': am.detach().long().cpu().numpy().astype(np.int64) if am is not None else None,
        }
        log('captured attention_mask shape ' + (str(rec['frame0']['attention_mask'].shape) if rec['frame0']['attention_mask'] is not None else 'None'))
        log('attention_mask values: ' + str(rec['frame0']['attention_mask'].ravel().tolist()))
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
np.savez(OUT + '/frame0.npz', **{k: v for k, v in f0.items() if v is not None})

# replay WITH attention_mask
from transformers.cache_utils import DynamicCache
kv = DynamicCache()
for i in range(28):
    kv.update(torch.from_numpy(f0['kv_k'][i]), torch.from_numpy(f0['kv_v'][i]), i)
cp0 = int(f0['cp'][0])
am = torch.from_numpy(f0['attention_mask']) if f0['attention_mask'] is not None else None
with torch.inference_mode():
    out = base(inputs_embeds=torch.from_numpy(f0['emb']), position_ids=torch.from_numpy(f0['pos']),
               attention_mask=am, past_key_values=kv, cache_position=torch.tensor([cp0]), use_cache=True, output_hidden_states=True)
h_with_am = out.last_hidden_state.detach().float().numpy().astype(np.float32)

# replay WITHOUT attention_mask (baseline)
kv2 = DynamicCache()
for i in range(28):
    kv2.update(torch.from_numpy(f0['kv_k'][i]), torch.from_numpy(f0['kv_v'][i]), i)
with torch.inference_mode():
    out2 = base(inputs_embeds=torch.from_numpy(f0['emb']), position_ids=torch.from_numpy(f0['pos']),
                past_key_values=kv2, cache_position=torch.tensor([cp0]), use_cache=True, output_hidden_states=True)
h_no_am = out2.last_hidden_state.detach().float().numpy().astype(np.float32)

traj = np.load(OFF + '/official_traj.npz')
h_off = traj['hidden_out'][0].astype(np.float32)
log('')
log('replay WITH attention_mask  vs official = %.7f' % cos_fn(h_with_am, h_off))
log('replay WITHOUT attention_mask vs official = %.7f' % cos_fn(h_no_am, h_off))
log('with_am vs no_am = %.7f' % cos_fn(h_with_am, h_no_am))
log('official maxabs %.3f  with_am maxabs %.3f  no_am maxabs %.3f' % (
    float(np.abs(h_off).max()), float(np.abs(h_with_am).max()), float(np.abs(h_no_am).max())))
print('P55E_DONE with_am=%.7f no_am=%.7f' % (cos_fn(h_with_am, h_off), cos_fn(h_no_am, h_off)), flush=True)
