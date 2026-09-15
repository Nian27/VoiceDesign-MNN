# P5.1: trace official generate_voice_design phase machine (TEXT/TRAILING/PAD/EOS)
import os, json, numpy as np, torch, time, traceback, functools
os.environ['HF_HUB_OFFLINE'] = '1'
D = r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/device-smoke'
LOG = open(D + '/p51_trace.log', 'w')
def log(s):
    LOG.write(s + '\n'); LOG.flush()
from qwen_tts import Qwen3TTSModel
gw = Qwen3TTSModel.from_pretrained(r'E:/AndroidStudioProjects/qwen3tts-mnn/models/qwen3-tts/Qwen3-TTS-12Hz-1.7B-VoiceDesign', dtype=torch.float32, device_map='cpu')
talker = gw.model.talker
log('LOADED')
# monkey-patch talker.forward to capture per-step state
orig_fwd = talker.forward
state = {'steps': [], 'call': 0}
@functools.wraps(orig_fwd)
def patched_fwd(*args, **kwargs):
    state['call'] += 1
    cp = kwargs.get('cache_position')
    gs = kwargs.get('generation_step')
    trail = kwargs.get('trailing_text_hidden')
    pad = kwargs.get('tts_pad_embed')
    emb = kwargs.get('inputs_embeds')
    if emb is not None and emb.shape[1] > 1:
        phase = 'PREFILL'
    else:
        phase = 'DECODE'
    rec = {'call': state['call'], 'phase': phase, 'cp': cp.tolist() if cp is not None else None, 'gen_step': gs, 'trail_len': trail.shape[1] if trail is not None else None, 'pad': bool(pad is not None), 'emb_shape': list(emb.shape) if emb is not None else None}
    # determine which emb source used (trail vs pad) from gen_step
    if trail is not None and gs is not None and gs < trail.shape[1]:
        rec['emb_src'] = 'TRAILING'
    elif pad is not None:
        rec['emb_src'] = 'PAD'
    else:
        rec['emb_src'] = 'OTHER'
    state['steps'].append(rec)
    log('call ' + str(state['call']) + ' ' + phase + ' cp ' + str(rec['cp']) + ' gs ' + str(gs) + ' trail_len ' + str(rec['trail_len']) + ' emb_src ' + rec['emb_src'])
    return orig_fwd(*args, **kwargs)
talker.forward = patched_fwd
texts = ['小家伙，修炼之事急不得，根基若是不稳，日后吃亏的还是你自己。']
instructs = ['苍老男性，低沉略沙哑，声音有岁月感，语气沉稳、有威严，但并不凶狠。']
t0 = time.time()
try:
    wavs, sr = gw.generate_voice_design(text=texts, instruct=instructs, language='Chinese', non_streaming_mode=True)
    log('GEN OK sr ' + str(sr) + ' secs ' + str(round(time.time()-t0,1)) + ' wav_len ' + str(len(wavs[0])))
except Exception as e:
    log('GEN ERR ' + repr(e)[:300])
    traceback.print_exc(file=LOG)
# analyze phases
phases = [s['emb_src'] for s in state['steps'] if s['phase'] == 'DECODE']
log('DECODE steps: ' + str(len(phases)))
# first TRAILING/PAD transition
if 'TRAILING' in phases:
    first_pad = phases.index('PAD') if 'PAD' in phases else -1
    log('first PAD at decode index ' + str(first_pad) + ' (trailing steps before)')
# EOS check: decode calls count
log('TOTAL calls: ' + str(len(state['steps'])) + ' decode: ' + str(len(phases)))
with open(D + '/p51_trace.json', 'w') as f:
    f.write(json.dumps(state['steps'], indent=2))
print('P51_TRACE_DONE', len(phases), flush=True)