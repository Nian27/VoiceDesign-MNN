# P2-A(fix) + P2-B export: 1L DIAG v2 with host-provided merged rope_cos/rope_sin
import os, numpy as np, torch
os.environ['HF_HUB_OFFLINE'] = '1'
import torch.nn.functional as F
D = r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/device-smoke'
OUT = D + '/vd_p2'
os.makedirs(OUT, exist_ok=True)
def cos(a,b):
    a=np.asarray(a,np.float64).ravel(); b=np.asarray(b,np.float64).ravel()
    return float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12))
from qwen_tts import Qwen3TTSModel
gw = Qwen3TTSModel.from_pretrained(r'E:/AndroidStudioProjects/qwen3tts-mnn/models/qwen3-tts/Qwen3-TTS-12Hz-1.7B-VoiceDesign', dtype=torch.float32, device_map='cpu')
base = gw.model.talker.model
a0 = base.layers[0].self_attn
ms = a0.rope_scaling['mrope_section']; MOD = 3

def rotate_half(x):
    x1 = x[..., : x.shape[-1]//2]; x2 = x[..., x.shape[-1]//2:]
    return torch.cat((-x2, x1), dim=-1)

def host_merge(c3, s3):
    dim = c3.shape[-1]; half = dim // 2
    def merge(x):
        x_t = x[0].clone()
        for i, n in enumerate(ms[1:], 1):
            beg = i; end = n * MOD
            x_t[..., beg:end:MOD] = x[i, ..., beg:end:MOD]
        return x_t
    return torch.cat([merge(c3[..., :half])] * 2, dim=-1), torch.cat([merge(s3[..., :half])] * 2, dim=-1)

emb = np.fromfile(D + '/vd_m5a_inputs/step0_emb.raw', dtype=np.float32).reshape(1,1,2048)
pos = np.fromfile(D + '/vd_m5a_inputs/step0_pos.raw', dtype=np.float32).reshape(3,1,1)
with torch.no_grad():
    hidden = torch.from_numpy(emb)
    c3, s3 = base.rotary_emb(hidden, torch.from_numpy(pos))
    mc, msin = host_merge(c3, s3)          # (1,1,128) each
    mc.numpy().tofile(OUT + '/merged_cos.raw'); msin.numpy().tofile(OUT + '/merged_sin.raw')
    print('merged cos', tuple(mc.shape), 'saved 512B each')
    # layer0_out with correct v
    norm_out = base.layers[0].input_layernorm(hidden)
    attn = base.layers[0].self_attn
    hd = attn.head_dim; hs = norm_out.shape[:-1]; shp = (*hs,-1,hd)
    q = attn.q_norm(attn.q_proj(norm_out).view(shp)).transpose(1,2)
    k = attn.k_norm(attn.k_proj(norm_out).view(shp)).transpose(1,2)
    v = attn.v_proj(norm_out).view(shp).transpose(1,2)
    q_r = q * mc + rotate_half(q) * msin
    k_r = k * mc + rotate_half(k) * msin
    kvk = torch.from_numpy(np.fromfile(D+'/vd_m5a_inputs/step0_kvk.raw', dtype=np.float32).reshape(28,1,8,768,128)).clone()
    kvv = torch.from_numpy(np.fromfile(D+'/vd_m5a_inputs/step0_kvv.raw', dtype=np.float32).reshape(28,1,8,768,128)).clone()
    cp = 62
    kvk[0,:,:,cp,:] = k_r[:,:,0,:]; kvv[0,:,:,cp,:] = v[:,:,0,:]
    g = attn.num_key_value_groups
    k_all = kvk[0].repeat_interleave(g, dim=1); v_all = kvv[0].repeat_interleave(g, dim=1)
    positions = torch.arange(768).view(1,1,1,768)
    amask = torch.where(positions <= cp, torch.tensor(0.0), torch.tensor(float('-inf')))
    ao = F.scaled_dot_product_attention(q_r, k_all, v_all, attn_mask=amask, dropout_p=0.0, is_causal=False)
    res1 = hidden + attn.o_proj(ao.reshape(*hs,-1).contiguous())
    l0 = res1 + base.layers[0].mlp(base.layers[0].post_attention_layernorm(res1))
    ref = np.fromfile(D + '/vd_p1/ref_layer0_out.raw', dtype=np.float32)
    print('layer0_out(host rope) vs official ref  cos=%.8f maxdiff=%.3e' % (cos(l0.numpy(), ref), float(np.abs(l0.numpy()-ref).max())))

# ---- P2-B export: 1L DIAG v2 (host rope) ----
class GraphB1LDiagV2(torch.nn.Module):
    def __init__(self, base, max_kv, li):
        super().__init__(); self.base = base; self.max_kv = max_kv; self.li = li
    def forward(self, inputs_embeds, cache_position, kv_k, kv_v, rope_cos, rope_sin):
        hidden = inputs_embeds
        kvk_out = kv_k.clone(); kvv_out = kv_v.clone()
        layer = self.base.layers[self.li]
        residual = hidden
        norm_out = layer.input_layernorm(hidden)
        attn = layer.self_attn
        hd = attn.head_dim; hs = norm_out.shape[:-1]; shp = (*hs, -1, hd)
        q_raw = attn.q_proj(norm_out).view(shp)
        k_raw = attn.k_proj(norm_out).view(shp)
        v_raw = attn.v_proj(norm_out).view(shp)
        q_norm = attn.q_norm(q_raw); k_norm = attn.k_norm(k_raw)
        q = q_norm.transpose(1,2); k = k_norm.transpose(1,2); v = v_raw.transpose(1,2)
        q_rope = q * rope_cos + rotate_half(q) * rope_sin
        k_rope = k * rope_cos + rotate_half(k) * rope_sin
        kvk_out[self.li, :, :, cache_position, :] = k_rope[:, :, 0, :].unsqueeze(2)
        kvv_out[self.li, :, :, cache_position, :] = v[:, :, 0, :].unsqueeze(2)
        g = attn.num_key_value_groups
        k_all = kvk_out[self.li].repeat_interleave(g, dim=1); v_all = kvv_out[self.li].repeat_interleave(g, dim=1)
        positions = torch.arange(self.max_kv).view(1,1,1,self.max_kv)
        amask = torch.where(positions <= cache_position, torch.tensor(0.0), torch.tensor(float('-inf')))
        ao = F.scaled_dot_product_attention(q_rope, k_all, v_all, attn_mask=amask, dropout_p=0.0, is_causal=False)
        attn_out = ao.reshape(*hs, -1).contiguous()
        o_proj_out = attn.o_proj(attn_out)
        res1 = residual + o_proj_out
        post_norm = layer.post_attention_layernorm(res1)
        mlp_out = layer.mlp(post_norm)
        layer0_out = res1 + mlp_out
        return (norm_out, q_raw, k_raw, v_raw, q_norm, k_norm, q_rope, k_rope,
                attn_out, o_proj_out, res1, post_norm, mlp_out, layer0_out, kvk_out, kvv_out)
core = GraphB1LDiagV2(base, 768, 0).eval()
names = ['norm_out','q_raw','k_raw','v_raw','q_norm','k_norm','q_rope','k_rope','attn_out','o_proj_out','res1','post_norm','mlp_out','layer0_out','kv_k_out','kv_v_out']
e = torch.zeros((1,1,2048)); cp_ = torch.tensor([62], dtype=torch.int64)
kk = torch.zeros((28,1,8,768,128)); vv = torch.zeros((28,1,8,768,128))
mc2 = torch.zeros((1,1,128)); ms2 = torch.zeros((1,1,128))
with torch.inference_mode():
    outs = core(e, cp_, kk, vv, mc2, ms2)
torch.onnx.export(core, (e, cp_, kk, vv, mc2, ms2), OUT + '/graphb_1l_diag_v2.onnx',
                  input_names=['inputs_embeds','cache_position','kv_k','kv_v','rope_cos','rope_sin'],
                  output_names=names, opset_version=18, dynamo=False)
import onnx
mm = onnx.load(OUT + '/graphb_1l_diag_v2.onnx', load_external_data=False)
ops = {}
for n in mm.graph.node: ops[n.op_type] = ops.get(n.op_type,0)+1
print('v2 IsNaN', ops.get('IsNaN',0), 'size', os.path.getsize(OUT + '/graphb_1l_diag_v2.onnx'))
print('has ScatterND:', ops.get('ScatterND',0), 'Slice:', ops.get('Slice',0))
print('P2AB_DONE')
