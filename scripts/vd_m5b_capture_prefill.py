import os, numpy as np, torch, functools
os.environ['HF_HUB_OFFLINE'] = '1'
D = r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/device-smoke'
OUT = D + '/vd_m5b'
os.makedirs(OUT, exist_ok=True)
from qwen_tts import Qwen3TTSModel
gw = Qwen3TTSModel.from_pretrained(r'E:/AndroidStudioProjects/qwen3tts-mnn/models/qwen3-tts/Qwen3-TTS-12Hz-1.7B-VoiceDesign', dtype=torch.float32, device_map='cpu')
talker = gw.model.talker
base = talker.model
cap = {}
orig = base.forward
@functools.wraps(orig)
def patched(*args, **kwargs):
    emb = kwargs.get('inputs_embeds', args[0] if args else None)
    if emb is not None and 'emb' not in cap:
        cap['emb'] = emb.detach().float().cpu().numpy().astype(np.float32)
        cap['pos'] = kwargs.get('position_ids').detach().float().cpu().numpy().astype(np.float32)
    return orig(*args, **kwargs)
base.forward = patched
texts = ['小家伙，修炼之事急不得，根基若是不稳，日后吃亏的还是你自己。']
instructs = ['苍老男性，低沉略沙哑，声音有岁月感，语气沉稳、有威严，但并不凶狠。']
try:
    wavs, sr = gw.generate_voice_design(text=texts, instruct=instructs, language='Chinese', non_streaming_mode=True, max_new_tokens=1)
except Exception as e:
    print('gen stopped early (prefill already captured):', repr(e)[:120])
base.forward = orig
print('PREFILL LEN =', cap['emb'].shape[1], 'emb', cap['emb'].shape, 'pos', cap['pos'].shape)
print('pos head =', cap['pos'].ravel()[:3], 'tail =', cap['pos'].ravel()[-3:])
cap['emb'].tofile(OUT + '/prefill_emb.raw'); cap['pos'].tofile(OUT + '/prefill_pos.raw')
np.save(OUT + '/prefill_emb.npy', cap['emb']); np.save(OUT + '/prefill_pos.npy', cap['pos'])
print('sizes emb', os.path.getsize(OUT+'/prefill_emb.raw'), 'pos', os.path.getsize(OUT+'/prefill_pos.raw'))
print('PREFILL_CAPTURE_DONE')
