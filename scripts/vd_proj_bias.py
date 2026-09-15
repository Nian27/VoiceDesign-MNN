import os, numpy as np, torch
os.environ['HF_HUB_OFFLINE']='1'
MDL = r'E:/AndroidStudioProjects/qwen3tts-mnn/models/qwen3-tts/Qwen3-TTS-12Hz-1.7B-VoiceDesign'
OUT = r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/device-smoke/app_models'
from qwen_tts import Qwen3TTSModel
gw = Qwen3TTSModel.from_pretrained(MDL, dtype=torch.float32, device_map='cpu')
tp = gw.model.talker.text_projection
b1 = tp.linear_fc1.bias.detach().float().numpy().astype(np.float32)
b2 = tp.linear_fc2.bias.detach().float().numpy().astype(np.float32)
print('b1', b1.shape, 'mean|.|=%.5f' % np.abs(b1).mean())
print('b2', b2.shape, 'mean|.|=%.5f' % np.abs(b2).mean())
b1.tofile(OUT + '/text_proj_fc1_bias.f32')
b2.tofile(OUT + '/text_proj_fc2_bias.f32')
print('bytes', os.path.getsize(OUT+'/text_proj_fc1_bias.f32'), os.path.getsize(OUT+'/text_proj_fc2_bias.f32'))
# 立刻在 PC 上验证带 bias 的复现
f1 = np.fromfile(OUT + '/text_proj_fc1.fp16', dtype=np.float16).astype(np.float32).reshape(2048,2048)
f2 = np.fromfile(OUT + '/text_proj_fc2.fp16', dtype=np.float16).astype(np.float32).reshape(2048,2048)
def silu(z): return z/(1+np.exp(-z))
def cos(a,b):
    a=a.astype(np.float64); b=b.astype(np.float64)
    return float(a@b/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12))
tid = 102635
with torch.no_grad():
    x = gw.model.talker.model.text_embedding.weight[tid:tid+1].float().numpy().astype(np.float32).reshape(-1)
    ref = tp(gw.model.talker.model.text_embedding.weight[tid:tid+1].float()).numpy().reshape(-1)
mine = (silu(x @ f1.T + b1)) @ f2.T + b2
print('WITH BIAS: PROJ cos = %.7f   maxabsdiff=%.6f' % (cos(mine, ref), float(np.abs(mine-ref).max())))
print('PROJ_FIXED_DONE')