# VD codepred AR ground truth v2: store LAST-POSITION logits per lm_head call (shape-uniform).
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

lm_g = []; lm_last = []; lm_seqlen = []
for gi in range(len(cp.lm_head)):
    def make_hook(gi_):
        def h(mod, args, output):
            try:
                o = output.detach().float()
                lm_g.append(gi_)
                lm_seqlen.append(int(o.shape[1]) if o.dim() == 3 else -1)
                lm_last.append(o[0, -1].cpu().numpy().astype(np.float32))
            except Exception as e:
                log('lmhook err ' + str(e)[:120])
        return h
    cp.lm_head[gi].register_forward_hook(make_hook(gi))

cpm_len = []
def cp_model_hook(mod, args, kwargs, output):
    ie = kwargs.get('inputs_embeds')
    if ie is None and args:
        ie = args[0]
    cpm_len.append(int(ie.shape[1]) if ie is not None and ie.dim() == 3 else -1)
cp.model.register_forward_hook(cp_model_hook, with_kwargs=True)

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
            'lm_mark': len(lm_g),
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

log('frames=%d lm_calls=%d' % (len(frames), len(lm_g)))
marks = [f['lm_mark'] for f in frames] + [len(lm_g)]
per = [marks[i+1]-marks[i] for i in range(len(frames))]
log('lm per frame: min=%d max=%d first5=%s' % (min(per), max(per), str(per[:5])))
log('g pattern frame0: ' + str(lm_g[0:16]))
log('seqlen pattern frame0: ' + str(lm_seqlen[0:16]))

np.savez(OUT + '/official_codepred.npz',
         code0_in=np.array([f['code0_in'] for f in frames], dtype=np.int32),
         past_hidden=np.stack([f['past_hidden'] for f in frames if f['past_hidden'] is not None]).astype(np.float32),
         codec_ids=np.stack([f['codec_ids'] for f in frames if f['codec_ids'] is not None]).astype(np.int32),
         lm_marks=np.array([f['lm_mark'] for f in frames], dtype=np.int64),
         lm_g=np.array(lm_g, dtype=np.int32),
         lm_seqlen=np.array(lm_seqlen, dtype=np.int32),
         lm_last_logits=np.stack(lm_last).astype(np.float32))
log('SAVED official_codepred.npz  logits %s' % str(np.stack(lm_last).shape))
print('CODEPRED_AR_V2_DONE', len(frames), len(lm_g), flush=True)
