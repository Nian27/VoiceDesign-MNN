import os, numpy as np, torch
os.environ['HF_HUB_OFFLINE'] = '1'
torch.set_num_threads(1)
from qwen_tts import Qwen3TTSModel
gw = Qwen3TTSModel.from_pretrained(r'E:/AndroidStudioProjects/qwen3tts-mnn/models/qwen3-tts/Qwen3-TTS-12Hz-1.7B-VoiceDesign', dtype=torch.float32, device_map='cpu')
cp = gw.model.talker.code_predictor; core = cp.model
print('gw.training', getattr(gw,'training',None), 'cp.training', cp.training, 'core.training', core.training)
D = r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/device-smoke'
past = torch.from_numpy(np.fromfile(D + '/vd_m5b/prefill_hidden_last.raw', dtype=np.float32).reshape(1,1,2048))
# use a FIXED fake code0 emb (avoid depending on MNN)
c0e = torch.from_numpy(np.fromfile(D + '/vd_cpr/codec_emb_weight.f32', dtype=np.float32).reshape(15,2048,2048)[0][1995].reshape(1,1,2048))
def once():
    with torch.no_grad():
        xx = cp.small_to_mtp_projection(torch.cat([past, c0e], dim=1))
        o = core(inputs_embeds=xx, use_cache=True)
        h = o.last_hidden_state
        lg = cp.lm_head[0](h[:, -1:])[0,0].numpy().astype(np.float32)
    return lg
a = once(); b = once(); c = once()
print('stage0 logits f4 run1', a[:4])
print('stage0 logits f4 run2', b[:4])
print('stage0 logits f4 run3', c[:4])
print('maxdiff a-b %.3e   a-c %.3e' % (float(np.abs(a-b).max()), float(np.abs(a-c).max())))
print('argmax a/b/c', int(np.argmax(a)), int(np.argmax(b)), int(np.argmax(c)))
gw.eval(); cp.eval(); core.eval()
d = once(); e = once()
print('after eval(): f4', d[:4], 'maxdiff a-d %.3e' % float(np.abs(a-d).max()))
print('DETERM_PROBE_DONE')
