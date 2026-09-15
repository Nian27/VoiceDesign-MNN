import os, torch, time, json
os.environ['HF_HUB_OFFLINE'] = '1'
MODEL = r'E:/AndroidStudioProjects/qwen3tts-mnn/models/qwen3-tts/Qwen3-TTS-12Hz-1.7B-VoiceDesign'
ART = r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/m22'
os.makedirs(ART + '/golden', exist_ok=True)
from qwen_tts import Qwen3TTSModel
print('loading...', flush=True)
gw = Qwen3TTSModel.from_pretrained(MODEL, dtype=torch.bfloat16, device_map='cuda' if torch.cuda.is_available() else 'cpu')
m = gw.model; talker = m.talker; base = talker.model; cp = talker.code_predictor
capture = {'decode_steps': [], 'cp_start': [], 'kv_shapes': []}
def base_hook(mod, args, kwargs, output=None):
    inputs_embeds = kwargs.get('inputs_embeds')
    if inputs_embeds is None: return
    cache_pos = kwargs.get('cache_position')
    step = int(cache_pos[0].item()) if cache_pos is not None else -1
    rec = {
        'step': step,
        'inputs_embeds': inputs_embeds.detach().float().cpu(),
        'position_ids': kwargs.get('position_ids').detach().float().cpu(),
        'cache_position': cache_pos.detach().cpu() if cache_pos is not None else None,
        'mask_shape': tuple(kwargs.get('attention_mask').shape) if kwargs.get('attention_mask') is not None else None,
    }
    # KV: capture shapes (flatten DynamicCache layers: 28L x 2 tensors)
    kv = kwargs.get('past_key_values')
    if kv is not None and hasattr(kv, '__len__') and len(kv) > 0:
        try:
            layer0 = kv[0]
            k = layer0[0] if isinstance(layer0, (tuple, list)) else layer0
            rec['kv_shape'] = tuple(k.shape) if torch.is_tensor(k) else str(type(k).__name__)
        except Exception as e:
            rec['kv_shape'] = 'ERR:' + str(e)[:80]
    if inputs_embeds.shape[1] == 1 and len(capture['decode_steps']) < 3:
        capture['decode_steps'].append(rec)
        print('G1_DECODE', json.dumps(rec, default=str), flush=True)
def cp_hook(mod, args, kwargs, output=None):
    ie = kwargs.get('inputs_embeds')
    if ie is not None and len(capture['cp_start']) < 2:
        capture['cp_start'].append({'inputs_embeds': ie.detach().float().cpu(), 'max_new_tokens': kwargs.get('max_new_tokens'), 'dosample': kwargs.get('do_sample')})
        print('G1_CP', 'emb', tuple(ie.shape), 'max_tokens', kwargs.get('max_new_tokens'), flush=True)
h1 = base.register_forward_hook(base_hook, with_kwargs=True)
h2 = cp.model.register_forward_hook(cp_hook, with_kwargs=True)
t0 = time.time()
try:
    wavs, sr = gw.generate_voice_design(text=['我在这个夏天，只想去看看海。'], instruct=['沉稳成熟的青年男声，音色偏沉'], language='Chinese', non_streaming_mode=True)
    print('GEN_OK', sr, len(wavs), flush=True)
except Exception as e:
    print('GEN_FAIL', repr(e)[:400], flush=True)
h1.remove(); h2.remove()
print('gen_s', round(time.time()-t0, 2), flush=True)
torch.save(capture, ART + '/golden/g1_decode_golden.pt')
with open(ART + '/golden/g1_decode_meta.json', 'w', encoding='utf-8') as f:
    json.dump({k: len(v) for k, v in capture.items()}, f)
print('G1_GOLDEN_SAVED', flush=True)