# G2-B: rope table for cp = 62..361 (300 positions)
import os, numpy as np, torch
os.environ['HF_HUB_OFFLINE'] = '1'
D = r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/device-smoke'
OUT = D + '/vd_p2/g2'
os.makedirs(OUT, exist_ok=True)
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
    return torch.cat([merge(c3[..., :half])] * 2, dim=-1), torch.cat([merge(s3[..., :half])] * 2, dim=-1)
N = 300; CP0 = 62
cos_t = np.zeros((N,1,1,128), np.float32); sin_t = np.zeros((N,1,1,128), np.float32)
with torch.no_grad():
    x = torch.zeros((1,1,2048))
    for k in range(N):
        cp = CP0 + k
        pos = torch.tensor([[[float(cp)]],[[float(cp)]],[[float(cp)]]])
        c3, s3 = base.rotary_emb(x, pos)
        mc, msin = host_merge(c3, s3)
        cos_t[k] = mc.numpy(); sin_t[k] = msin.numpy()
cos_t.tofile(OUT + '/cos300.raw'); sin_t.tofile(OUT + '/sin300.raw')
print('rope table: cos', cos_t.shape, 'sin', sin_t.shape, 'cp', CP0, '..', CP0+N-1)
print('G2B_TABLE_DONE')
