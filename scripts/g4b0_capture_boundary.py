# G4-B0: capture real VoiceDesign GraphA->GraphB boundary via forward hook
import os, json, numpy as np, torch, io, time
os.environ['HF_HUB_OFFLINE'] = '1'
MODEL = r'E:/AndroidStudioProjects/qwen3tts-mnn/models/qwen3-tts/Qwen3-TTS-12Hz-1.7B-VoiceDesign'
OUT = r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/device-smoke/g4b_boundary'
os.makedirs(OUT, exist_ok=True)
from qwen_tts import Qwen3TTSModel
gw = Qwen3TTSModel.from_pretrained(MODEL, dtype=torch.float32, device_map='cpu')
talker = gw.model.talker
base = talker.model
print('LOADED', flush=True)
# hook: capture prefill (first base call) inputs + kv outputs
captured = {}
call_count = [0]
def hook_fn(mod, args, kwargs, output):
    call_count[0] += 1
    if call_count[0] == 1:  # prefill
        captured['inputs_embeds'] = kwargs.get('inputs_embeds', args[0] if args else None).detach().clone()
        captured['position_ids'] = kwargs.get('position_ids')
        captured['cache_position'] = kwargs.get('cache_position')
        pkv = kwargs.get('past_key_values')
        if hasattr(pkv, 'layers') and len(pkv.layers) > 0:
            captured['kv_k'] = torch.stack([pkv.layers[i].keys for i in range(28)], 0).detach().clone()
            captured['kv_v'] = torch.stack([pkv.layers[i].values for i in range(28)], 0).detach().clone()
            print('PREFILL KV captured:', tuple(captured['kv_k'].shape), flush=True)
        else:
            print('PREFILL pkv empty/pre', flush=True)
        print('PREFILL emb', tuple(captured['inputs_embeds'].shape), 'pos', None if captured['position_ids'] is None else tuple(captured['position_ids'].shape), flush=True)
    else:
        print('call', call_count[0], 'kv_len?', flush=True)
h = base.register_forward_hook(hook_fn)
texts = ['小家伙，修炼之事急不得，根基若是不稳，日后吃亏的还是你自己。']
instructs = ['苍老男性，低沉略沙哑，声音有岁月感，语气沉稳、有威严，但并不凶狠。']
t0 = time.time()
try:
    wavs, sr = gw.generate_voice_design(text=texts, instruct=instructs, language='Chinese', non_streaming_mode=True)
    print('gen ok sr', sr, 'calls', call_count[0], 'secs', round(time.time()-t0,1), flush=True)
except Exception as e:
    print('gen FAIL', repr(e)[:400], flush=True)
h.remove()
# save boundary if prefill captured
if 'kv_k' in captured:
    np.save(OUT + '/kv_k.npy', captured['kv_k'].float().numpy().astype(np.float32))
    np.save(OUT + '/kv_v.npy', captured['kv_v'].float().numpy().astype(np.float32))
    np.save(OUT + '/step0_inputs.npz', emb=captured['inputs_embeds'].float().numpy().astype(np.float32), pos=captured['position_ids'].float().numpy().astype(np.float32) if captured['position_ids'] is not None else np.array([]), cp=captured['cache_position'].long().numpy() if captured['cache_position'] is not None else np.array([]))
    manifest = {
        'kv_shape': list(captured['kv_k'].shape),
        'emb_shape': list(captured['inputs_embeds'].shape),
        'pos_shape': None if captured['position_ids'] is None else list(captured['position_ids'].shape),
        'cp': None if captured['cache_position'] is None else captured['cache_position'].tolist(),
        'num_base_calls': call_count[0],
        'sr': int(sr) if sr else None,
        'texts': texts, 'instructs': instructs,
        'kv_len': captured['kv_k'].shape[3],
    }
    with open(OUT + '/manifest.json', 'w') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print('BOUNDARY_SAVED', manifest, flush=True)
else:
    print('BOUNDARY_NOT_CAPTURED calls=' + str(call_count[0]), flush=True)
print('G4B0_DONE', flush=True)