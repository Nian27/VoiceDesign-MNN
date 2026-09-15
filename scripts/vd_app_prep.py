import os, numpy as np, torch
os.environ['HF_HUB_OFFLINE']='1'
D = r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/device-smoke'
OUT = D + '/app_models'; os.makedirs(OUT, exist_ok=True)
MAXPOS = 384
from qwen_tts import Qwen3TTSModel
gw = Qwen3TTSModel.from_pretrained(r'E:/AndroidStudioProjects/qwen3tts-mnn/models/qwen3-tts/Qwen3-TTS-12Hz-1.7B-VoiceDesign', dtype=torch.float32, device_map='cpu')
base = gw.model.talker.model
ms = base.layers[0].self_attn.rope_scaling['mrope_section']; MOD = 3
def host_merge(c3, s3):
    half = c3.shape[-1] // 2
    def merge(x):
        x_t = x[0].clone()
        for i, n in enumerate(ms[1:], 1):
            x_t[..., i:n*MOD:MOD] = x[i, ..., i:n*MOD:MOD]
        return x_t
    return torch.cat([merge(c3[..., :half])]*2, dim=-1), torch.cat([merge(s3[..., :half])]*2, dim=-1)
tc = np.zeros((MAXPOS,128), np.float32); ts = np.zeros((MAXPOS,128), np.float32)
with torch.no_grad():
    x = torch.zeros(1,1,2048)
    for p in range(MAXPOS):
        pos = torch.tensor([[[float(p)]],[[float(p)]],[[float(p)]]])
        c3, s3 = base.rotary_emb(x, pos)
        mc, msin = host_merge(c3, s3)
        tc[p] = mc.reshape(128).numpy(); ts[p] = msin.reshape(128).numpy()
tc.tofile(OUT + '/talker_rope_cos.f32'); ts.tofile(OUT + '/talker_rope_sin.f32')
cp = gw.model.talker.code_predictor; core = cp.model
with torch.no_grad():
    cc, ss = core.rotary_emb(torch.zeros(1,1,1024), torch.arange(16).unsqueeze(0))
np.asarray(cc, np.float32).reshape(16,128).tofile(OUT + '/codepred_rope_cos.f32')
np.asarray(ss, np.float32).reshape(16,128).tofile(OUT + '/codepred_rope_sin.f32')
pe = np.fromfile(D + '/vd_m5b/prefill_emb.raw', dtype=np.float32).reshape(-1, 2048)
pe.tofile(OUT + '/prompt_emb.f32')
print('prompt tokens =', pe.shape[0])
print('APP_MODELS_PREP_DONE')