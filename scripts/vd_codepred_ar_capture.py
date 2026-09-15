# VD codepred AR ground truth: capture official per-step codepred logits + prefix.
# Official semantics (modeling_qwen3_tts.py:1670-1687):
#   code_predictor.generate(inputs_embeds=cat(past_hidden, code0_emb), max_new_tokens=15, ...)
#   -> 15 AR steps; step g uses cp_emb[g-1](code_g) + KV, lm_head[g]
# We hook each lm_head[g] and code_predictor.model to get exact per-step logits + inputs.
import os, json, numpy as np, torch, functools, traceback
os.environ['HF_HUB_OFFLINE'] = '1'
D = r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/device-smoke'
OUT = D + '/codepred_ar'
os.makedirs(OUT, exist_ok=True)
LOG = open(OUT + '/capture.log', 'w')
def log(s):
    LOG.write(str(s) + '\n'); LOG.flush()
from qwen_tts import Qwen3TTSModel
gw = Qwen3TTSModel.from_pretrained(r'E:/AndroidStudioProjects/qwen3tts-mnn/models/qwen3-tts/Qwen3-TTS-12Hz-1.7B-VoiceDesign', dtype=torch.float32, device_map='cpu')
talker = gw.model.talker
cp = talker.code_predictor
log('LOADED')

lm_call_log = []   # each: {'g': int, 'logits': np}
cp_model_calls = []  # each: {'emb': np (seq,1024 post-proj is inside), 'shape': ...}

# hook lm_head[g]
for gi in range(len(cp.lm_head)):
    def make_hook(gi_):
        def h(mod, args, output):
            try:
                o = output.detach().float().cpu().numpy().astype(np.float32)
                lm_call_log.append({'g': gi_, 'logits': o, 'in_shape': list(args[0].shape) if args else None})
            except Exception as e:
                log('lmhook err ' + str(e)[:120])
        return h
    cp.lm_head[gi].register_forward_hook(make_hook(gi))

# hook code_predictor.model to capture (post-proj) inputs
def cp_model_hook(mod, args, kwargs, output):
    try:
        ie = kwargs.get('inputs_embeds')
        if ie is None and args:
            ie = args[0]
        if ie is not None:
            cp_model_calls.append({'shape': list(ie.shape), 'emb': ie.detach().float().cpu().numpy().astype(np.float32)})
    except Exception as e:
        log('cpmodelhook err ' + str(e)[:120])
cp.model.register_forward_hook(cp_model_hook, with_kwargs=True)

# hook talker.forward for per-frame past_hidden / input_ids / codec_ids
frames = []
orig = talker.forward
@functools.wraps(orig)
def patched(*args, **kwargs):
    emb = kwargs.get('inputs_embeds', args[0] if args else None)
    is_prefill = emb is not None and emb.shape[1] > 1
    out = orig(*args, **kwargs)
    if not is_prefill:
        iid = kwargs.get('input_ids')
        ph = kwargs.get('past_hidden')
        cids = None
        if hasattr(out, 'hidden_states') and out.hidden_states is not None and len(out.hidden_states) >= 2 and out.hidden_states[1] is not None:
            cids = out.hidden_states[1].detach().long().cpu().numpy().astype(np.int32)
        frames.append({
            'code0_in': int(iid[0,0].item()) if iid is not None else -1,
            'past_hidden': ph.detach().float().cpu().numpy().astype(np.float32) if ph is not None else None,
            'codec_ids': cids,
            'lm_mark': len(lm_call_log),
            'cpm_mark': len(cp_model_calls),
        })
    return out
talker.forward = patched

texts = ['小家伙，修炼之事急不得，根基若是不稳，日后吃亏的还是你自己。']
instructs = ['苍老男性，低沉略沙哑，声音有岁月感，语气沉稳、有威严，但并不凶狠。']
try:
    wavs, sr = gw.generate_voice_design(text=texts, instruct=instructs, language='Chinese', non_streaming_mode=True)
    log('GEN OK sr ' + str(sr) + ' wav ' + str(len(wavs[0])))
except Exception as e:
    log('GEN ERR ' + repr(e)[:300]); traceback.print_exc(file=LOG)
talker.forward = orig

log('frames=' + str(len(frames)) + ' lm_calls=' + str(len(lm_call_log)) + ' cpmodel_calls=' + str(len(cp_model_calls)))
# per-frame: 15 lm calls expected
if len(frames) > 0:
    marks = [f['lm_mark'] for f in frames] + [len(lm_call_log)]
    log('lm calls per frame (first 5): ' + str([marks[i+1]-marks[i] for i in range(min(5, len(frames)))]))
    log('cpmodel calls per frame (first 5): ' + str([frames[i+1]['cpm_mark']-frames[i]['cpm_mark'] if i+1 < len(frames) else len(cp_model_calls)-frames[i]['cpm_mark'] for i in range(min(5, len(frames)))]))
np.savez(OUT + '/frames.npz',
         n_frames=np.array([len(frames)]),
         code0_in=np.array([f['code0_in'] for f in frames], dtype=np.int32),
         past_hidden=np.stack([f['past_hidden'] for f in frames if f['past_hidden'] is not None]).astype(np.float32),
         codec_ids=np.stack([f['codec_ids'] for f in frames if f['codec_ids'] is not None]).astype(np.int32),
         lm_marks=np.array([f['lm_mark'] for f in frames], dtype=np.int64),
         cpm_marks=np.array([f['cpm_mark'] for f in frames], dtype=np.int64))
np.savez(OUT + '/lm_calls.npz',
         n=np.array([len(lm_call_log)]),
         g=np.array([c['g'] for c in lm_call_log], dtype=np.int32),
         logits=np.stack([c['logits'] for c in lm_call_log]).astype(np.float32))
np.savez(OUT + '/cpm_calls.npz',
         n=np.array([len(cp_model_calls)]),
         emb=np.stack([c['emb'] for c in cp_model_calls]).astype(np.float32))
print('CODEPRED_AR_CAPTURED', len(frames), len(lm_call_log), len(cp_model_calls), flush=True)
