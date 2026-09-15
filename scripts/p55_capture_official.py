# P5.5a: capture FULL official trajectory — per-decode-step:
#   input code0 (input_ids), past_hidden_in, hidden_out, codec_ids (16), logits
# This is the ground truth for first-divergence comparison.
import os, json, numpy as np, torch, functools, time, traceback
os.environ['HF_HUB_OFFLINE'] = '1'
D = r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/device-smoke'
OUT = D + '/p55_official'
os.makedirs(OUT, exist_ok=True)
LOG = open(OUT + '/capture.log', 'w')
def log(s):
    LOG.write(s + '\n'); LOG.flush()
from qwen_tts import Qwen3TTSModel
gw = Qwen3TTSModel.from_pretrained(r'E:/AndroidStudioProjects/qwen3tts-mnn/models/qwen3-tts/Qwen3-TTS-12Hz-1.7B-VoiceDesign', dtype=torch.float32, device_map='cpu')
talker = gw.model.talker
log('LOADED')
orig_fwd = talker.forward
rec = {'prefill_hidden': None, 'decode': [], 'call': 0}
@functools.wraps(orig_fwd)
def patched(*args, **kwargs):
    rec['call'] += 1
    emb = kwargs.get('inputs_embeds', args[0] if args else None)
    out = orig_fwd(*args, **kwargs)
    is_prefill = emb is not None and emb.shape[1] > 1
    if is_prefill:
        if rec['prefill_hidden'] is None and hasattr(out, 'past_hidden') and out.past_hidden is not None:
            rec['prefill_hidden'] = out.past_hidden.detach().float().cpu().numpy().astype(np.float32)
            log('PREFILL past_hidden saved ' + str(rec['prefill_hidden'].shape))
    else:
        cp = kwargs.get('cache_position')
        cpn = int(cp[0].item()) if cp is not None else -1
        in_ids = kwargs.get('input_ids')
        code0 = int(in_ids[0,0].item()) if in_ids is not None else -1
        ph = kwargs.get('past_hidden')
        ph_in = ph.detach().float().cpu().numpy().astype(np.float32) if ph is not None else None
        h_out = out.past_hidden.detach().float().cpu().numpy().astype(np.float32) if hasattr(out, 'past_hidden') and out.past_hidden is not None else None
        lg = out.logits.detach().float().cpu().numpy().astype(np.float32) if hasattr(out, 'logits') and out.logits is not None else None
        # codec_ids = [code0] + 15 codes: output.hidden_states = (base_hidden_states, codec_ids)
        cids = None
        if hasattr(out, 'hidden_states') and out.hidden_states is not None and isinstance(out.hidden_states, (tuple, list)) and len(out.hidden_states) >= 2:
            c2 = out.hidden_states[1]
            if c2 is not None:
                cids = c2.detach().long().cpu().numpy().astype(np.int32) if torch.is_tensor(c2) else None
        rec['decode'].append({'cp': cpn, 'code0_in': code0, 'past_hidden_in': ph_in, 'hidden_out': h_out, 'codec_ids': cids, 'logits': lg})
    return out
talker.forward = patched
texts = ['小家伙，修炼之事急不得，根基若是不稳，日后吃亏的还是你自己。']
instructs = ['苍老男性，低沉略沙哑，声音有岁月感，语气沉稳、有威严，但并不凶狠。']
t0 = time.time()
try:
    wavs, sr = gw.generate_voice_design(text=texts, instruct=instructs, language='Chinese', non_streaming_mode=True)
    log('GEN OK sr ' + str(sr) + ' wav ' + str(len(wavs[0])) + ' secs ' + str(round(time.time()-t0,1)))
except Exception as e:
    log('GEN ERR ' + repr(e)[:300]); traceback.print_exc(file=LOG)
talker.forward = orig_fwd
n = len(rec['decode'])
log('decode frames captured: ' + str(n))
# pack arrays
cps = np.array([d['cp'] for d in rec['decode']], dtype=np.int32)
code0_in = np.array([d['code0_in'] for d in rec['decode']], dtype=np.int32)
hid_out = np.stack([d['hidden_out'] for d in rec['decode'] if d['hidden_out'] is not None]).astype(np.float32)
ph_in = np.stack([d['past_hidden_in'] for d in rec['decode'] if d['past_hidden_in'] is not None]).astype(np.float32)
cids = np.stack([d['codec_ids'] for d in rec['decode'] if d['codec_ids'] is not None]).astype(np.int32)
lg = np.stack([d['logits'] for d in rec['decode'] if d['logits'] is not None]).astype(np.float32)
np.savez(OUT + '/official_traj.npz',
         prefill_hidden=rec['prefill_hidden'].astype(np.float32),
         cps=cps, code0_in=code0_in, hidden_out=hid_out, past_hidden_in=ph_in, codec_ids=cids, logits=lg)
with open(OUT + '/manifest.json', 'w') as f:
    json.dump({'n_decode': n, 'cps': cps.tolist(), 'code0_in': code0_in.tolist(),
               'hid_out_shape': list(hid_out.shape), 'ph_in_shape': list(ph_in.shape),
               'cids_shape': list(cids.shape), 'lg_shape': list(lg.shape)}, f, indent=2)
log('SAVED official_traj.npz  hid_out ' + str(hid_out.shape) + ' ph_in ' + str(ph_in.shape) + ' cids ' + str(cids.shape) + ' lg ' + str(lg.shape))
print('P55_OFFICIAL_CAPTURED', n, flush=True)
