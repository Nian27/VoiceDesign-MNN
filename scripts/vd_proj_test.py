import numpy as np, torch, os
os.environ['HF_HUB_OFFLINE']='1'
OUT = r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/device-smoke/app_models'
MDL = r'E:/AndroidStudioProjects/qwen3tts-mnn/models/qwen3-tts/Qwen3-TTS-12Hz-1.7B-VoiceDesign'
from qwen_tts import Qwen3TTSModel
gw = Qwen3TTSModel.from_pretrained(MDL, dtype=torch.float32, device_map='cpu')
tp = gw.model.talker.text_projection
# 1) 导出表 vs 模型权重
e_fp16 = np.fromfile(OUT + '/text_embedding.fp16', dtype=np.float16)
te = gw.model.talker.model.text_embedding.weight.detach().float().numpy().astype(np.float32).ravel()
print('text_embedding fp16 bytes=%d  expect=%d' % (e_fp16.size, te.size))
e32 = e_fp16.astype(np.float32)
d = np.abs(e32 - te)
print('  embed maxabsdiff(fp16 vs fp32) = %.5f  (fp16 量化误差量级)' % d.max())
f1 = np.fromfile(OUT + '/text_proj_fc1.fp16', dtype=np.float16).astype(np.float32).reshape(2048,2048)
f2 = np.fromfile(OUT + '/text_proj_fc2.fp16', dtype=np.float16).astype(np.float32).reshape(2048,2048)
w1 = tp.linear_fc1.weight.detach().float().numpy(); w2 = tp.linear_fc2.weight.detach().float().numpy()
print('  fc1 maxabsdiff=%.5f  fc2 maxabsdiff=%.5f' % (np.abs(f1-w1).max(), np.abs(f2-w2).max()))
# 2) 手工复现 projection 与 torch 对比
def silu(x): return x/(1+np.exp(-x))
def myproj(x): return silu(x @ f1.T) @ f2.T
with torch.no_grad():
    tid = 102635
    x = gw.model.talker.model.text_embedding.weight[tid:tid+1].float().numpy().astype(np.float32)
    ref = tp(torch.tensor(x)).numpy().reshape(-1)
    mine = myproj(x).reshape(-1)
a=mine.astype(np.float64); b=ref.astype(np.float64)
print('PROJ cos = %.7f   maxabsdiff=%.5f' % (float(a@b/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12)), float(np.abs(a-b).max())))
print('ref mean|.|=%.4f  mine mean|.|=%.4f' % (np.abs(b).mean(), np.abs(a).mean()))
print('PROJ_TEST_DONE')