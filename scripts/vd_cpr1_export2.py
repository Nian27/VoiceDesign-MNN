# CP-R1 (v2): per-token one-hot mask writes, no einsum/scatter
import os, math, numpy as np, torch
os.environ['HF_HUB_OFFLINE'] = '1'
D = r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/device-smoke'
OUT = D + '/vd_cpr'
os.makedirs(OUT, exist_ok=True)
from qwen_tts import Qwen3TTSModel
gw = Qwen3TTSModel.from_pretrained(r'E:/AndroidStudioProjects/qwen3tts-mnn/models/qwen3-tts/Qwen3-TTS-12Hz-1.7B-VoiceDesign', dtype=torch.float32, device_map='cpu')
cp = gw.model.talker.code_predictor; core = cp.model
NL, NH, NKV, HD, CAP = 5, 16, 8, 128, 16
SCALE = 1.0 / math.sqrt(HD)
def rot_half(x):
    x1 = x[..., : x.shape[-1]//2]; x2 = x[..., x.shape[-1]//2:]
    return torch.cat((-x2, x1), dim=-1)

def do_layer(i, h, cos, sin, kvk, kvv, writes, attn_mask):
    """writes: list of (mask(1,1,CAP,1), k(1,8,1,128), v(1,8,1,128))"""
    layer = self.core.layers[i]
    residual = h
    hn = layer.input_layernorm(h)
    at = layer.self_attn
    hs = hn.shape[:-1]; shp = (*hs, -1, HD)
    q = at.q_norm(at.q_proj(hn).view(shp)).transpose(1,2)
    k = at.k_norm(at.k_proj(hn).view(shp)).transpose(1,2)
    v = at.v_proj(hn).view(shp).transpose(1,2)
    qr = q * cos + rot_half(q) * sin
    kr = k * cos + rot_half(k) * sin
    keep = 1.0
    add_k = 0.0; add_v = 0.0
    for (m, ks, vs) in writes:
        keep = keep - m
        add_k = add_k + ks * m
        add_v = add_v + vs * m
    kvk_i = kvk[i] * keep + add_k
    kvv_i = kvv[i] * keep + add_v
    g = NH // NKV
    k_all = kvk_i.repeat_interleave(g, dim=1)
    v_all = kvv_i.repeat_interleave(g, dim=1)
    logits = torch.matmul(qr, k_all.transpose(-1,-2)) * SCALE
    probs = torch.softmax(logits + attn_mask, dim=-1)
    ao = torch.matmul(probs, v_all).reshape(*hs, -1).contiguous()
    h2 = residual + at.o_proj(ao); r2 = h2
    h2 = layer.post_attention_layernorm(h2)
    return r2 + layer.mlp(h2), kvk_i, kvv_i

def empty_kv():
    return torch.zeros(NL,1,NKV,CAP,HD), torch.zeros(NL,1,NKV,CAP,HD)

class Prefill(torch.nn.Module):
    def forward(self, past_hidden, code0_emb, rope_cos, rope_sin, slot_mask0, slot_mask1, attn_mask):
        x = self.cp.small_to_mtp_projection(torch.cat([past_hidden, code0_emb], dim=1))
        kvk, kvv = empty_kv(); ok=[]; ov=[]; h = x
        for i in range(NL):
            at = self.core.layers[i].self_attn
            hs = self.core.layers[i].input_layernorm(x).shape[:-1]
            # recompute k,v slices inside do_layer is not possible; pass slices via closure-free approach:
            h, ki, vi = do_layer(i, h, rope_cos, rope_sin, kvk, kvv, PREF_WRITES[0], attn_mask)
            ok.append(ki.unsqueeze(0)); ov.append(vi.unsqueeze(0))
        return self.core.norm(h)[:, -1:, :], torch.cat(ok,0), torch.cat(ov,0)

# The write slices depend on the layer's own k/v, so handle prefill by a dedicated impl:
class Prefill2(torch.nn.Module):
    def __init__(self, cp, core):
        super().__init__(); self.cp = cp; self.core = core
    def forward(self, past_hidden, code0_emb, rope_cos, rope_sin, slot_mask0, slot_mask1, attn_mask):
        x = self.cp.small_to_mtp_projection(torch.cat([past_hidden, code0_emb], dim=1))
        kvk, kvv = empty_kv(); ok=[]; ov=[]; h = x
        for i in range(NL):
            layer = self.core.layers[i]; residual = h
            hn = layer.input_layernorm(h); at = layer.self_attn
            hs = hn.shape[:-1]; shp = (*hs,-1,HD)
            q = at.q_norm(at.q_proj(hn).view(shp)).transpose(1,2)
            k = at.k_norm(at.k_proj(hn).view(shp)).transpose(1,2)
            v = at.v_proj(hn).view(shp).transpose(1,2)
            qr = q*rope_cos + rot_half(q)*rope_sin
            kr = k*rope_cos + rot_half(k)*rope_sin
            k0 = kr[:,:,0:1,:]; k1 = kr[:,:,1:2,:]
            v0 = v[:,:,0:1,:];  v1 = v[:,:,1:2,:]
            kvk_i = kvk[i]*(1.0-slot_mask0-slot_mask1) + k0*slot_mask0 + k1*slot_mask1
            kvv_i = kvv[i]*(1.0-slot_mask0-slot_mask1) + v0*slot_mask0 + v1*slot_mask1
            g = NH//NKV
            k_all = kvk_i.repeat_interleave(g, dim=1); v_all = kvv_i.repeat_interleave(g, dim=1)
            logits = torch.matmul(qr, k_all.transpose(-1,-2))*SCALE
            probs = torch.softmax(logits + attn_mask, dim=-1)
            ao = torch.matmul(probs, v_all).transpose(1,2).reshape(*hs,-1).contiguous()
            h2 = residual + at.o_proj(ao); r2 = h2
            h2 = layer.post_attention_layernorm(h2)
            h = r2 + layer.mlp(h2)
            ok.append(kvk_i.unsqueeze(0)); ov.append(kvv_i.unsqueeze(0))
        return self.core.norm(h)[:, -1:, :], torch.cat(ok,0), torch.cat(ov,0)

class Step2(torch.nn.Module):
    def __init__(self, cp, core):
        super().__init__(); self.cp = cp; self.core = core
    def forward(self, token_emb, kv_k, kv_v, rope_cos, rope_sin, slot_mask, attn_mask):
        x = self.cp.small_to_mtp_projection(token_emb)
        ok=[]; ov=[]; h = x
        for i in range(NL):
            layer = self.core.layers[i]; residual = h
            hn = layer.input_layernorm(h); at = layer.self_attn
            hs = hn.shape[:-1]; shp = (*hs,-1,HD)
            q = at.q_norm(at.q_proj(hn).view(shp)).transpose(1,2)
            k = at.k_norm(at.k_proj(hn).view(shp)).transpose(1,2)
            v = at.v_proj(hn).view(shp).transpose(1,2)
            qr = q*rope_cos + rot_half(q)*rope_sin
            kr = k*rope_cos + rot_half(k)*rope_sin
            kvk_i = kv_k[i]*(1.0-slot_mask) + kr*slot_mask
            kvv_i = kv_v[i]*(1.0-slot_mask) + v*slot_mask
            g = NH//NKV
            k_all = kvk_i.repeat_interleave(g, dim=1); v_all = kvv_i.repeat_interleave(g, dim=1)
            logits = torch.matmul(qr, k_all.transpose(-1,-2))*SCALE
            probs = torch.softmax(logits + attn_mask, dim=-1)
            ao = torch.matmul(probs, v_all).transpose(1,2).reshape(*hs,-1).contiguous()
            h2 = residual + at.o_proj(ao); r2 = h2
            h2 = layer.post_attention_layernorm(h2)
            h = r2 + layer.mlp(h2)
            ok.append(kvk_i.unsqueeze(0)); ov.append(kvv_i.unsqueeze(0))
        return self.core.norm(h), torch.cat(ok,0), torch.cat(ov,0)

pref = Prefill2(cp, core).eval(); stepm = Step2(cp, core).eval()
ph = torch.zeros(1,1,2048); c0e = torch.zeros(1,1,2048)
pc = torch.zeros(1,1,2,HD); ps = torch.zeros(1,1,2,HD)
m0 = torch.zeros(1,1,CAP,1); m0[0,0,0,0]=1
m1 = torch.zeros(1,1,CAP,1); m1[0,0,1,0]=1
am2 = torch.full((1,1,2,CAP), -1e4); am2[0,0,0,0]=0; am2[0,0,1,0]=0; am2[0,0,0,1]=0; am2[0,0,1,1]=0
kk0 = torch.zeros(NL,1,NKV,CAP,HD); vv0 = torch.zeros(NL,1,NKV,CAP,HD)
with torch.inference_mode():
    o = pref(ph, c0e, pc, ps, m0, m1, am2)
print('prefill out', [tuple(t.shape) for t in o], 'finite', bool(torch.isfinite(o[0]).all()))
torch.onnx.export(pref, (ph, c0e, pc, ps, m0, m1, am2), OUT + '/codepred_prefill.onnx',
    input_names=['past_hidden','code0_emb','rope_cos','rope_sin','slot_mask0','slot_mask1','attn_mask'],
    output_names=['hidden_last','kv_k_out','kv_v_out'], opset_version=18, dynamo=False)
ts = torch.zeros(1,1,2048); sc = torch.zeros(1,1,1,HD); ss = torch.zeros(1,1,1,HD)
sm1 = torch.zeros(1,1,CAP,1); sm1[0,0,2,0]=1
am1 = torch.full((1,1,1,CAP), -1e4)
for j in range(3): am1[0,0,0,j]=0
with torch.inference_mode():
    o2 = stepm(ts, kk0, vv0, sc, ss, sm1, am1)
print('step out', [tuple(t.shape) for t in o2], 'finite', bool(torch.isfinite(o2[0]).all()))
torch.onnx.export(stepm, (ts, kk0, vv0, sc, ss, sm1, am1), OUT + '/codepred_step.onnx',
    input_names=['token_emb','kv_k','kv_v','rope_cos','rope_sin','slot_mask','attn_mask'],
    output_names=['hidden_last','kv_k_out','kv_v_out'], opset_version=18, dynamo=False)
lw = np.stack([cp.lm_head[g].weight.detach().numpy().astype(np.float32) for g in range(15)])
ew = np.stack([core.codec_embedding[g].weight.detach().numpy().astype(np.float32) for g in range(15)])
lw.tofile(OUT + '/lm_head_weight.f32'); ew.tofile(OUT + '/codec_emb_weight.f32')
print('lm_head', lw.shape, 'codec_emb', ew.shape)
import onnx
for f in ['codepred_prefill.onnx','codepred_step.onnx']:
    m = onnx.load(OUT + '/' + f, load_external_data=False)
    ops = {}
    for n in m.graph.node: ops[n.op_type] = ops.get(n.op_type,0)+1
    print('%-24s IsNaN=%d ArgMax=%d ScatterND=%d nodes=%d size=%d' % (f, ops.get('IsNaN',0), ops.get('ArgMax',0), ops.get('ScatterND',0), len(m.graph.node), os.path.getsize(OUT+'/'+f)))
print('CPR1_EXPORT_DONE')
