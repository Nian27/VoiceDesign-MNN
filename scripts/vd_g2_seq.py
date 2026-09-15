# Generate per-step HOST inputs for G2-A: rope_cos/sin, attn_mask, slot_mask for cp = 62..81
import os, numpy as np, torch
os.environ['HF_HUB_OFFLINE'] = '1'
D = r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/device-smoke'
OUT = D + '/vd_p2/g2'
os.makedirs(OUT, exist_ok=True)
from qwen_tts import Qwen3TTSModel
gw = Qwen3TTSModel.from_pretrained(r'E:/AndroidStudioProjects/qwen3tts-mnn/models/qwen3-tts/Qwen3-TTS-12Hz-1.7B-VoiceDesign', dtype=torch.float32, device_map='cpu')
base = gw.model.talker.model
a0 = base.layers[0].self_attn
ms = a0.rope_scaling['mrope_section']; MOD = 3; MAXKV = 768
def host_merge(c3, s3):
    half = c3.shape[-1] // 2
    def merge(x):
        x_t = x[0].clone()
        for i, n in enumerate(ms[1:], 1):
            x_t[..., i:n*MOD:MOD] = x[i, ..., i:n*MOD:MOD]
        return x_t
    return torch.cat([merge(c3[..., :half])] * 2, dim=-1), torch.cat([merge(s3[..., :half])] * 2, dim=-1)
N = 20; CP0 = 62
cosall = np.zeros((N,1,1,128), np.float32); sinall = np.zeros((N,1,1,128), np.float32)
maskall = np.zeros((N,1,1,1,MAXKV), np.float32); slotall = np.zeros((N,1,1,MAXKV,1), np.float32)
with torch.no_grad():
    x = torch.zeros((1,1,2048))
    for k in range(N):
        cp = CP0 + k
        pos = torch.tensor([[[float(cp)]],[[float(cp)]],[[float(cp)]]])
        c3, s3 = base.rotary_emb(x, pos)
        mc, msin = host_merge(c3, s3)
        cosall[k] = mc.numpy(); sinall[k] = msin.numpy()
        m = np.zeros((1,1,1,MAXKV), np.float32); m[0,0,0,:cp+1] = 0.0; m[0,0,0,cp+1:] = -1e4
        maskall[k] = m
        s = np.zeros((1,1,MAXKV,1), np.float32); s[0,0,cp,0] = 1.0
        slotall[k] = s
cosall.tofile(OUT + '/cos20.raw'); sinall.tofile(OUT + '/sin20.raw')
maskall.tofile(OUT + '/mask20.raw'); slotall.tofile(OUT + '/slot20.raw')
print('seq saved: cos', cosall.shape, 'mask', maskall.shape, 'slot', slotall.shape)
print('cp range', CP0, '..', CP0+N-1)
print('G2_SEQ_DONE')
