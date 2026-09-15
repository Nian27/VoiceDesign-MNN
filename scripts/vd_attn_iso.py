# Isolate: compare official layer0 attention output vs manual, and rope stage.
import os, numpy as np, torch
os.environ['HF_HUB_OFFLINE'] = '1'
import torch.nn.functional as F
def cos(a,b):
    a=np.asarray(a,np.float64).ravel(); b=np.asarray(b,np.float64).ravel()
    return float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12))
from qwen_tts import Qwen3TTSModel
from qwen_tts.core.models.modeling_qwen3_tts import apply_rotary_pos_emb
gw = Qwen3TTSModel.from_pretrained(r'E:/AndroidStudioProjects/qwen3tts-mnn/models/qwen3-tts/Qwen3-TTS-12Hz-1.7B-VoiceDesign', dtype=torch.float32, device_map='cpu')
cp = gw.model.talker.code_predictor
cpm = cp.model
torch.manual_seed(0)
n = 5
xt = torch.randn(1, n, 1024) * 0.5
cap = {}
def hk(mod, args, kwargs, output):
    cap['attn'] = (output[0] if isinstance(output, tuple) else output).detach().numpy()
    cap['q'] = kwargs.get('position_embeddings')
cpm.layers[0].self_attn.register_forward_hook(hk, with_kwargs=True)
with torch.no_grad():
    cpm(inputs_embeds=xt, use_cache=False)
a = cpm.layers[0].self_attn
attn = a
with torch.no_grad():
    hn = cpm.layers[0].input_layernorm(xt)
    hs = hn.shape[:-1]; hd = attn.head_dim
    shp = (*hs, -1, hd)
    q = attn.q_norm(attn.q_proj(hn).view(shp)).transpose(1,2)
    k = attn.k_norm(attn.k_proj(hn).view(shp)).transpose(1,2)
    v = attn.v_proj(hn).view(shp).transpose(1,2)
    pos = torch.arange(n, dtype=torch.long).unsqueeze(0)
    cosv, sinv = cpm.rotary_emb(xt, pos)
    q2, k2 = apply_rotary_pos_emb(q, k, cosv, sinv)
    # variant A: is_causal=True
    g = attn.num_key_value_groups
    kk = k2.repeat_interleave(g, dim=1); vv = v.repeat_interleave(g, dim=1)
    aoA = F.scaled_dot_product_attention(q2, kk, vv, dropout_p=0.0, is_causal=True)
    aoA = attn.o_proj(aoA.reshape(*hs,-1).contiguous()).numpy()
    # variant B: explicit causal bool mask, is_causal=False
    mask = torch.tril(torch.ones(n, n, dtype=torch.bool))
    aoB = F.scaled_dot_product_attention(q2, kk, vv, attn_mask=mask, dropout_p=0.0, is_causal=False)
    aoB = attn.o_proj(aoB.reshape(*hs,-1).contiguous()).numpy()
    # variant C: GQA via enable_gqa
    try:
        aoC = F.scaled_dot_product_attention(q2, k2, v, dropout_p=0.0, is_causal=True, enable_gqa=True)
        aoC = attn.o_proj(aoC.reshape(*hs,-1).contiguous()).numpy()
    except Exception as e:
        aoC = None; print('enable_gqa err', str(e)[:80])
print('official attn out shape', cap['attn'].shape)
print('A is_causal=True        cos=%.6f' % cos(aoA, cap['attn']))
print('B bool-mask is_causal=F cos=%.6f' % cos(aoB, cap['attn']))
if aoC is not None: print('C enable_gqa            cos=%.6f' % cos(aoC, cap['attn']))
print('ISO_DONE')
