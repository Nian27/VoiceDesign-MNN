# Debug v2: hook official layers to find where manual diverges.
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
print('attn_impl =', getattr(cpm.config, '_attn_implementation', 'MISSING'))
print('Qwen3TTSAttention type:', type(cpm.layers[0].self_attn).__name__)
a0 = cpm.layers[0].self_attn
print('q_proj', a0.q_proj, '| k_proj', a0.k_proj, '| v_proj', a0.v_proj, '| o_proj', a0.o_proj)
print('q_norm', a0.q_norm, '| k_norm', a0.k_norm)
print('rotary type:', type(cpm.rotary_emb).__name__)

torch.manual_seed(0)
n = 5
xt = torch.randn(1, n, 1024) * 0.5
caps = {}
hooks = []
for i, layer in enumerate(cpm.layers):
    def mk(i_):
        def h(mod, args, output):
            caps['L%d' % i_] = (output[0] if isinstance(output, tuple) else output).detach().numpy()
        return h
    hooks.append(cpm.layers[i].register_forward_hook(mk(i)))
hooks.append(cpm.norm.register_forward_hook(lambda m,a,o: caps.__setitem__('norm', o.detach().numpy())))
with torch.inference_mode():
    off = cpm(inputs_embeds=xt, use_cache=False).last_hidden_state.numpy()
for h in hooks: h.remove()

# manual with per-layer capture
import torch as _t
with _t.no_grad():
  pass
pos = torch.arange(n, dtype=torch.long).unsqueeze(0)
cosv, sinv = cpm.rotary_emb(xt, pos)
print('cos shape', tuple(cosv.shape), 'sin shape', tuple(sinv.shape))
h = xt
for i, layer in enumerate(cpm.layers):
    residual = h
    hn = layer.input_layernorm(h)
    attn = layer.self_attn
    hs = hn.shape[:-1]; hd = attn.head_dim
    shp = (*hs, -1, hd)
    q = attn.q_norm(attn.q_proj(hn).view(shp)).transpose(1,2)
    k = attn.k_norm(attn.k_proj(hn).view(shp)).transpose(1,2)
    v = attn.v_proj(hn).view(shp).transpose(1,2)
    q2, k2 = apply_rotary_pos_emb(q, k, cosv, sinv)
    g = attn.num_key_value_groups
    kk = k2.repeat_interleave(g, dim=1); vv = v.repeat_interleave(g, dim=1)
    ao = F.scaled_dot_product_attention(q2, kk, vv, dropout_p=0.0, is_causal=True)
    ao = ao.reshape(*hs, -1).contiguous()
    ao = attn.o_proj(ao)
    h = residual + ao
    r2 = h
    h = layer.post_attention_layernorm(h)
    h = layer.mlp(h)
    h = r2 + h
    print('manual L%d cos vs official = %.6f' % (i, cos(h.numpy(), caps['L%d'%i])))
print('manual norm cos = %.6f' % cos(cpm.norm(h).detach().numpy(), caps['norm']))
print('DBG2_DONE')
