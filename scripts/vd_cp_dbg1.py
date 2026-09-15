# Debug: locate the divergence between manual codepred forward and official cpm.
import os, numpy as np, torch
os.environ['HF_HUB_OFFLINE'] = '1'
import torch.nn.functional as F
def cos(a,b):
    a=np.asarray(a,np.float64).ravel(); b=np.asarray(b,np.float64).ravel()
    return float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12))
from qwen_tts import Qwen3TTSModel
import inspect
from qwen_tts.core.models import modeling_qwen3_tts as M
gw = Qwen3TTSModel.from_pretrained(r'E:/AndroidStudioProjects/qwen3tts-mnn/models/qwen3-tts/Qwen3-TTS-12Hz-1.7B-VoiceDesign', dtype=torch.float32, device_map='cpu')
cp = gw.model.talker.code_predictor
cpm = cp.model
print('apply_rotary_pos_emb sig:', inspect.signature(M.apply_rotary_pos_emb))
print('source of apply_rotary_pos_emb used by attention:')
src = inspect.getsource(M.Qwen3TTSAttention.forward)
print(src[:1200])
