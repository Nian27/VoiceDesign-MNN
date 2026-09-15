# G4-B0 probe: does base forward hook fire during generate? (quick test, short text)
import os, json, numpy as np, torch, time
os.environ['HF_HUB_OFFLINE'] = '1'
MODEL = r'E:/AndroidStudioProjects/qwen3tts-mnn/models/qwen3-tts/Qwen3-TTS-12Hz-1.7B-VoiceDesign'
from qwen_tts import Qwen3TTSModel
gw = Qwen3TTSModel.from_pretrained(MODEL, dtype=torch.float32, device_map='cpu')
talker = gw.model.talker
base = talker.model
print('LOADED', flush=True)
calls = []
def hk(mod, args, kwargs, output):
    emb = kwargs.get('inputs_embeds', args[0] if args else None)
    cp = kwargs.get('cache_position')
    pkv = kwargs.get('past_key_values')
    kl = None
    if hasattr(pkv, 'layers') and len(pkv.layers) > 0 and pkv.layers[0].keys is not None:
        kl = pkv.layers[0].keys.shape
    calls.append({'emb': tuple(emb.shape) if emb is not None else None, 'cp': cp.tolist() if cp is not None else None, 'kv0': kl})
    print('CALL', len(calls), 'emb', tuple(emb.shape) if emb is not None else None, 'cp', cp.tolist() if cp is not None else None, 'kv0', kl, flush=True)
h = base.register_forward_hook(hk)
texts = ['小家伙，修炼之事急不得。']
instructs = ['苍老男性，低沉沙哑。']
t0 = time.time()
try:
    wavs, sr = gw.generate_voice_design(text=texts, instruct=instructs, language='Chinese', non_streaming_mode=True)
    print('gen ok sr', sr, 'calls', len(calls), 'secs', round(time.time()-t0,1), flush=True)
except Exception as e:
    print('gen FAIL', repr(e)[:300], 'calls_so_far', len(calls), 'secs', round(time.time()-t0,1), flush=True)
h.remove()
print('PROBE_DONE calls=' + str(len(calls)), flush=True)