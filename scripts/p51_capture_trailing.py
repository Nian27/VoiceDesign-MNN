# P5.1b: capture trailing_text_hidden + EOS step code0 from official run
import os, json, numpy as np, torch, functools, traceback
os.environ['HF_HUB_OFFLINE'] = '1'
D = r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/device-smoke'
OUT = r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/device-smoke/p51_assets'
os.makedirs(OUT, exist_ok=True)
LOG = open(OUT + '/capture.log', 'w')
def log(s):
    LOG.write(s + '\n'); LOG.flush()
from qwen_tts import Qwen3TTSModel
gw = Qwen3TTSModel.from_pretrained(r'E:/AndroidStudioProjects/qwen3tts-mnn/models/qwen3-tts/Qwen3-TTS-12Hz-1.7B-VoiceDesign', dtype=torch.float32, device_map='cpu')
talker = gw.model.talker
log('LOADED')
orig_fwd = talker.forward
state = {'call': 0, 'trail': None, 'pad': None, 'decode_steps': []}
@functools.wraps(orig_fwd)
def patched_fwd(*args, **kwargs):
    state['call'] += 1
    if kwargs.get('trailing_text_hidden') is not None and state['trail'] is None:
        state['trail'] = kwargs['trailing_text_hidden'].detach().clone()
    if kwargs.get('tts_pad_embed') is not None and state['pad'] is None:
        state['pad'] = kwargs['tts_pad_embed'].detach().clone()
    cp = kwargs.get('cache_position')
    emb = kwargs.get('inputs_embeds')
    if emb is not None and emb.shape[1] <= 1 and cp is not None:
        state['decode_steps'].append(int(cp[0]))
    return orig_fwd(*args, **kwargs)
talker.forward = patched_fwd
texts = ['小家伙，修炼之事急不得，根基若是不稳，日后吃亏的还是你自己。']
instructs = ['苍老男性，低沉略沙哑，声音有岁月感，语气沉稳、有威严，但并不凶狠。']
try:
    wavs, sr = gw.generate_voice_design(text=texts, instruct=instructs, language='Chinese', non_streaming_mode=True)
    log('GEN OK wav ' + str(len(wavs[0])) + ' sr ' + str(sr))
except Exception as e:
    log('GEN ERR ' + repr(e)[:300])
    traceback.print_exc(file=LOG)
if state['trail'] is not None:
    np.save(OUT + '/trailing_text_hidden.npy', state['trail'].float().numpy().astype(np.float32))
    log('TRAIL saved ' + str(tuple(state['trail'].shape)))
if state['pad'] is not None:
    np.save(OUT + '/tts_pad_embed.npy', state['pad'].float().numpy().astype(np.float32))
    log('PAD saved ' + str(tuple(state['pad'].shape)))
log('decode steps: ' + str(state['decode_steps'][:5]) + ' ... ' + str(state['decode_steps'][-3:]) + ' total ' + str(len(state['decode_steps'])))
log('EOS step (last decode cp): ' + str(state['decode_steps'][-1] if state['decode_steps'] else None))
print('P51_CAPTURE_DONE', len(state['decode_steps']), flush=True)