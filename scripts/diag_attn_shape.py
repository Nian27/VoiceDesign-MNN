import os, torch
os.environ['HF_HUB_OFFLINE'] = '1'
from qwen_tts import Qwen3TTSModel
gw = Qwen3TTSModel.from_pretrained(r'E:/AndroidStudioProjects/qwen3tts-mnn/models/qwen3-tts/Qwen3-TTS-12Hz-1.7B-VoiceDesign', dtype=torch.float32, device_map='cpu')
t = gw.model.talker
attn = t.model.layers[0].self_attn
print('config._attn_implementation:', t.config._attn_implementation)
x = torch.randn(1, 1, 2048)
pos = torch.tensor([[[34.0]],[[34.0]],[[34.0]]])
rot = t.model.rotary_emb(x, pos)
print('rot cos/sin shapes:', tuple(rot[0].shape), tuple(rot[1].shape))
import torch.nn.functional as F
hd = attn.head_dim
hs = x.shape[:-1]
q = attn.q_norm(attn.q_proj(x).view((*hs, -1, hd))).transpose(1,2)
k = attn.k_norm(attn.k_proj(x).view((*hs, -1, hd))).transpose(1,2)
v = attn.v_proj(x).view((*hs, -1, hd)).transpose(1,2)
print('q', tuple(q.shape), 'k', tuple(k.shape), 'v', tuple(v.shape))
print('q_proj out last:', attn.q_proj(x).shape, 'head_dim', hd)
print('DONE', flush=True)