import os, json, torch, time
os.environ['HF_HUB_OFFLINE'] = '1'
MODEL = r'E:/AndroidStudioProjects/qwen3tts-mnn/models/qwen3-tts/Qwen3-TTS-12Hz-1.7B-VoiceDesign'
ART = r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/m22'
os.makedirs(ART + '/golden', exist_ok=True)
from qwen_tts import Qwen3TTSModel
print('loading...', flush=True)
gw = Qwen3TTSModel.from_pretrained(MODEL, dtype=torch.bfloat16, device_map='cuda' if torch.cuda.is_available() else 'cpu')
m = gw.model
talker = m.talker
base = talker.model
cp = talker.code_predictor
print('talker children:', [n for n,_ in talker.named_children()], flush=True)
print('base children:', [n for n,_ in base.named_children()], flush=True)
print('cp children:', [n for n,_ in cp.named_children()], flush=True)
events = {'prefill': [], 'decode': [], 'cp': []}
def tometa(t):
    if t is None: return None
    if torch.is_tensor(t):
        return {'dtype': str(t.dtype), 'shape': tuple(t.shape), 'sample': [round(float(x),5) for x in t.flatten()[:8]]}
    if isinstance(t, tuple): return 'tuple<' + str(len(t)) + '>'
    return str(type(t).__name__)
def base_hook(mod, args, kwargs, output=None):
    rec = {'kind': 'base'}
    rec['inputs_embeds'] = tometa(kwargs.get('inputs_embeds'))
    rec['input_ids'] = tometa(kwargs.get('input_ids'))
    rec['position_ids'] = tometa(kwargs.get('position_ids'))
    rec['past_kv'] = tometa(kwargs.get('past_key_values'))
    rec['cache_position'] = tometa(kwargs.get('cache_position'))
    rec['mask'] = tometa(kwargs.get('attention_mask'))
    if kwargs.get('past_key_values') is None:
        events['prefill'].append(rec)
        print('PREFILL:', json.dumps(rec, ensure_ascii=False), flush=True)
    elif len(events['decode']) < 3:
        events['decode'].append(rec)
        print('DECODE:', json.dumps(rec, ensure_ascii=False), flush=True)
def cp_hook(mod, args, kwargs, output=None):
    rec = {'kind': 'cp'}
    rec['inputs_embeds'] = tometa(kwargs.get('inputs_embeds'))
    rec['past_kv'] = tometa(kwargs.get('past_key_values'))
    if kwargs.get('past_key_values') is None and len(events['cp']) == 0:
        events['cp'].append(rec)
        print('CP first:', json.dumps(rec, ensure_ascii=False), flush=True)
h1 = base.register_forward_hook(base_hook, with_kwargs=True)
h2 = cp.model.register_forward_hook(cp_hook, with_kwargs=True)
wavs = None
sr = None
texts = [{'text': '我在这个夏天，只想去看看海。', 'instruct': '沉稳成熟的青年男声，音色偏沉', 'language': 'Chinese'}]
t0 = time.time()
try:
    wavs, sr = gw.generate_voice_design(text=[t['text'] for t in texts], instruct=[t['instruct'] for t in texts], language='Chinese', non_streaming_mode=True)
    print('GEN_OK sr=', sr, 'n=', len(wavs), flush=True)
except Exception as e:
    print('GEN_FAIL', repr(e)[:500], flush=True)
print('gen_s', round(time.time()-t0, 2), flush=True)
h1.remove(); h2.remove()
out = {'events': events}
with open(ART + '/golden/m22_golden.json', 'w', encoding='utf-8') as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print('M22_GOLDEN_SAVED', flush=True)