import numpy as np, torch, os
os.environ['HF_HUB_OFFLINE']='1'
OUT = r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/device-smoke/app_models'
MDL = r'E:/AndroidStudioProjects/qwen3tts-mnn/models/qwen3-tts/Qwen3-TTS-12Hz-1.7B-VoiceDesign'
from qwen_tts import Qwen3TTSModel
gw = Qwen3TTSModel.from_pretrained(MDL, dtype=torch.float32, device_map='cpu')
tp = gw.model.talker.text_projection
print('tp repr:', tp)
print('fc1 shape', tuple(tp.linear_fc1.weight.shape), 'fc2 shape', tuple(tp.linear_fc2.weight.shape))
print('has bias:', tp.linear_fc1.bias is not None, tp.linear_fc2.bias is not None)
print('act:', tp.act_fn)
w1 = tp.linear_fc1.weight.detach().float().numpy(); w2 = tp.linear_fc2.weight.detach().float().numpy()
f1 = np.fromfile(OUT + '/text_proj_fc1.fp16', dtype=np.float16).astype(np.float32).reshape(2048,2048)
f2 = np.fromfile(OUT + '/text_proj_fc2.fp16', dtype=np.float16).astype(np.float32).reshape(2048,2048)
print('fc1 export vs model: maxdiff=%.6f  cos=%.7f' % (np.abs(f1-w1).max(), float((f1.ravel()*w1.ravel()).sum()/(np.linalg.norm(f1.ravel())*np.linalg.norm(w1.ravel())))))
print('fc2 export vs model: maxdiff=%.6f' % np.abs(f2-w2).max())
tid = 102635
with torch.no_grad():
    x = gw.model.talker.model.text_embedding.weight[tid:tid+1].float()
    h1 = tp.linear_fc1(x)
    h2 = tp.act_fn(h1)
    ref = tp(x).numpy().reshape(-1)
    xn = x.numpy().astype(np.float32).reshape(-1)
    h1n = h1.numpy().astype(np.float32).reshape(-1); h2n = h2.numpy().astype(np.float32).reshape(-1)
def cos(a,b): 
    a=a.astype(np.float64); b=b.astype(np.float64)
    return float(a@b/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12))
def silu(z): return z/(1+np.exp(-z))
c1a = xn @ f1.T; c1b = xn @ f1
print('fc1 stage:  x@f1.T vs torch  cos=%.7f   x@f1 vs torch cos=%.7f' % (cos(c1a,h1n), cos(c1b,h1n)))
mid = silu(c1a)
r1 = mid @ f2.T; r2 = mid @ f2
print('full: silu(x@f1.T)@f2.T cos=%.7f   @f2 cos=%.7f' % (cos(r1,ref), cos(r2,ref)))
print('PROJ2_DONE')