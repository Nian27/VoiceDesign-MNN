import os, math, inspect, numpy as np, torch, MNN
os.environ['HF_HUB_OFFLINE'] = '1'
torch.set_num_threads(1)
D = r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/device-smoke'; CR = D + '/vd_cpr'
CAP, HD = 16, 128
def cos(a,b):
    a=np.asarray(a,np.float64).ravel(); b=np.asarray(b,np.float64).ravel()
    return float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12))
from qwen_tts import Qwen3TTSModel
import qwen_tts.core.models.modeling_qwen3_tts as MF
gw = Qwen3TTSModel.from_pretrained(r'E:/AndroidStudioProjects/qwen3tts-mnn/models/qwen3-tts/Qwen3-TTS-12Hz-1.7B-VoiceDesign', dtype=torch.float32, device_map='cpu')
cp = gw.model.talker.code_predictor; core = cp.model
cfg = core.config
print('partial_rotary_factor =', getattr(cfg,'partial_rotary_factor','N/A'))
print('rope_scaling =', getattr(cfg,'rope_scaling',None))
print('head_dim', cfg.head_dim, 'hidden', cfg.hidden_size)
src = inspect.getsource(MF.Qwen3TTSAttention.forward)
i = src.find('apply_rotary') 
j = src.find('cos, sin')
print('--- attention rope usage ---')
k = src.find('cos')
print(src[:60])
seg = src
for kw in ['rotary_emb','apply_rotary_pos_emb','cos','sin']:
    p = seg.find(kw)
print('has apply_rotary_pos_emb:', 'apply_rotary_pos_emb' in src)
p = src.find('apply_rotary_pos_emb')
print(src[max(0,p-300):p+200] if p>0 else 'NOT FOUND in attention.forward')
