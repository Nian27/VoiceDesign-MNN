import os, torch
os.environ['HF_HUB_OFFLINE'] = '1'
from qwen_tts import Qwen3TTSModel
gw = Qwen3TTSModel.from_pretrained(r'E:/AndroidStudioProjects/qwen3tts-mnn/models/qwen3-tts/Qwen3-TTS-12Hz-1.7B-VoiceDesign', dtype=torch.float32, device_map='cpu')
t = gw.model.talker
attn = t.model.layers[0].self_attn
x = torch.randn(1, 1, 2048)
pos = torch.tensor([[[34.0]],[[34.0]],[[34.0]]])
rot = t.model.rotary_emb(x, pos)
hd = attn.head_dim
hs = x.shape[:-1]
q = attn.q_norm(attn.q_proj(x).view((*hs, -1, hd))).transpose(1,2)
k = attn.k_norm(attn.k_proj(x).view((*hs, -1, hd))).transpose(1,2)
v = attn.v_proj(x).view((*hs, -1, hd)).transpose(1,2)
from transformers.modeling_attn_utils import eager_attention_forward as eaf
import transformers
print('has eager:', hasattr(transformers.modeling_attn_utils if hasattr(transformers,'modeling_attn_utils') else None, 'eager_attention_forward'))
# try sdpa directly
from transformers.modeling_attn_utils import sdpa_attention_forward
try:
    out, w = sdpa_attention_forward(attn, q, k, v, None, scaling=attn.scaling)
    print('sdpa out:', tuple(out.shape))
except Exception as e:
    print('sdpa err:', str(e)[:200])
print('DONE', flush=True)