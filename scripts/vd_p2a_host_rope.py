# P2-A: host-side merged_cos/sin + contiguous RoPE  ==  official interleaved mRoPE ?
import os, numpy as np, torch
os.environ['HF_HUB_OFFLINE'] = '1'
D = r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/device-smoke'
OUT = D + '/vd_p2'
os.makedirs(OUT, exist_ok=True)
def cos(a,b):
    a=np.asarray(a,np.float64).ravel(); b=np.asarray(b,np.float64).ravel()
    return float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12))
from qwen_tts import Qwen3TTSModel
from qwen_tts.core.models.modeling_qwen3_tts import apply_multimodal_rotary_pos_emb
gw = Qwen3TTSModel.from_pretrained(r'E:/AndroidStudioProjects/qwen3tts-mnn/models/qwen3-tts/Qwen3-TTS-12Hz-1.7B-VoiceDesign', dtype=torch.float32, device_map='cpu')
base = gw.model.talker.model
a0 = base.layers[0].self_attn
ms = a0.rope_scaling['mrope_section']          # [24,20,20]
MOD = 3
print('mrope_section', ms)

# real step0 input
emb = np.fromfile(D + '/vd_m5a_inputs/step0_emb.raw', dtype=np.float32).reshape(1,1,2048)
pos = np.fromfile(D + '/vd_m5a_inputs/step0_pos.raw', dtype=np.float32).reshape(3,1,1)
with torch.no_grad():
    hidden = torch.from_numpy(emb)
    pos_t = torch.from_numpy(pos)
    c3, s3 = base.rotary_emb(hidden, pos_t)
    print('rotary cos shape', tuple(c3.shape), 'sin', tuple(s3.shape))
    norm_out = base.layers[0].input_layernorm(hidden)
    attn = base.layers[0].self_attn
    hd = attn.head_dim; hs = norm_out.shape[:-1]; shp = (*hs,-1,hd)
    q = attn.q_norm(attn.q_proj(norm_out).view(shp)).transpose(1,2)
    k = attn.k_norm(attn.k_proj(norm_out).view(shp)).transpose(1,2)
    # official
    q_off, k_off = apply_multimodal_rotary_pos_emb(q, k, c3, s3, ms, a0.rope_scaling['interleaved'])
    # host merge
    dim = c3.shape[-1]; half = dim // 2
    def merge(x):
        x_t = x[0].clone()
        for i, n in enumerate(ms[1:], 1):
            beg = i; end = n * MOD
            x_t[..., beg:end:MOD] = x[i, ..., beg:end:MOD]
        return x_t
    mc = torch.cat([merge(c3[..., :half])] * 2, dim=-1)
    msin = torch.cat([merge(s3[..., :half])] * 2, dim=-1)
    print('merged cos shape', tuple(mc.shape))
    def rotate_half(x):
        x1 = x[..., : x.shape[-1]//2]; x2 = x[..., x.shape[-1]//2:]
        return torch.cat((-x2, x1), dim=-1)
    cv = mc.unsqueeze(1); sv = msin.unsqueeze(1)
    q_host = q * cv + rotate_half(q) * sv
    k_host = k * cv + rotate_half(k) * sv
    print('q_rope  host vs official cos=%.8f maxdiff=%.3e' % (cos(q_host.numpy(), q_off.numpy()), float(np.abs(q_host.numpy()-q_off.numpy()).max())))
    print('k_rope  host vs official cos=%.8f maxdiff=%.3e' % (cos(k_host.numpy(), k_off.numpy()), float(np.abs(k_host.numpy()-k_off.numpy()).max())))
    # also layer0_out equivalence
    kvk = torch.from_numpy(np.fromfile(D+'/vd_m5a_inputs/step0_kvk.raw', dtype=np.float32).reshape(28,1,8,768,128))
    kvv = torch.from_numpy(np.fromfile(D+'/vd_m5a_inputs/step0_kvv.raw', dtype=np.float32).reshape(28,1,8,768,128))
    cp = 62
    kvk2 = kvk.clone(); kvv2 = kvv.clone()
    kvk2[0,:,:,cp,:] = k_host[:,:,0,:]; kvv2[0,:,:,cp,:] = k[:,:,0,:]
    g = attn.num_key_value_groups
    k_all = kvk2[0].repeat_interleave(g, dim=1); v_all = kvv2[0].repeat_interleave(g, dim=1)
    positions = torch.arange(768).view(1,1,1,768)
    amask = torch.where(positions <= cp, torch.tensor(0.0), torch.tensor(float('-inf')))
    import torch.nn.functional as F
    ao = F.scaled_dot_product_attention(q_host, k_all, v_all, attn_mask=amask, dropout_p=0.0, is_causal=False)
    ref_layer0 = np.fromfile(D + '/vd_p1/ref_layer0_out.raw', dtype=np.float32)
    o_proj = attn.o_proj(ao.reshape(*hs,-1).contiguous())
    res1 = hidden + o_proj; r2 = res1
    l0 = r2 + base.layers[0].mlp(base.layers[0].post_attention_layernorm(r2))
    print('layer0_out(host rope) vs official ref cos=%.8f' % cos(l0.numpy(), ref_layer0))
    mc[0].numpy().astype(np.float32).tofile(OUT + '/merged_cos.raw')
    msin[0].numpy().astype(np.float32).tofile(OUT + '/merged_sin.raw')
print('P2A_DONE')
