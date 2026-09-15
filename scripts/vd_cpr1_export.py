# CP-R1: export codepred_prefill + codepred_step (host sampling, KV cache, no ArgMax unroll)
import os, math, numpy as np, torch
os.environ['HF_HUB_OFFLINE'] = '1'
D = r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/device-smoke'
OUT = D + '/vd_cpr'
os.makedirs(OUT, exist_ok=True)
from qwen_tts import Qwen3TTSModel
gw = Qwen3TTSModel.from_pretrained(r'E:/AndroidStudioProjects/qwen3tts-mnn/models/qwen3-tts/Qwen3-TTS-12Hz-1.7B-VoiceDesign', dtype=torch.float32, device_map='cpu')
cp = gw.model.talker.code_predictor
core = cp.model
NL, NH, NKV, HD, CAP = 5, 16, 8, 128, 16
SCALE = 1.0 / math.sqrt(HD)

def rot_half(x):
    x1 = x[..., : x.shape[-1]//2]; x2 = x[..., x.shape[-1]//2:]
    return torch.cat((-x2, x1), dim=-1)

def layer_attn(core, i, h, cos, sin, kvk, kvv, slot_mask, attn_mask, groups):
    layer = core.layers[i]
    residual = h
    hn = layer.input_layernorm(h)
    at = layer.self_attn
    hs = hn.shape[:-1]; shp = (*hs, -1, HD)
    q = at.q_norm(at.q_proj(hn).view(shp)).transpose(1,2)
    k = at.k_norm(at.k_proj(hn).view(shp)).transpose(1,2)
    v = at.v_proj(hn).view(shp).transpose(1,2)
    qr = q * cos + rot_half(q) * sin
    kr = k * cos + rot_half(k) * sin
    kvk_i = kvk[i] * (1.0 - slot_mask) + kr * slot_mask
    kvv_i = kvv[i] * (1.0 - slot_mask) + v * slot_mask
    k_all = kvk_i.repeat_interleave(groups, dim=1)
    v_all = kvv_i.repeat_interleave(groups, dim=1)
    logits = torch.matmul(qr, k_all.transpose(-1,-2)) * SCALE
    probs = torch.softmax(logits + attn_mask, dim=-1)
    ao = torch.matmul(probs, v_all).reshape(*hs, -1).contiguous()
    h2 = residual + at.o_proj(ao)
    r2 = h2
    h2 = layer.post_attention_layernorm(h2)
    return r2 + layer.mlp(h2), kvk_i, kvv_i

class Prefill(torch.nn.Module):
    def __init__(self, cp, core, cap, scale):
        super().__init__(); self.cp=cp; self.core=core; self.cap=cap; self.scale=scale
    def forward(self, past_hidden, code0_emb, rope_cos, rope_sin, slot_mask, attn_mask):
        x = self.cp.small_to_mtp_projection(torch.cat([past_hidden, code0_emb], dim=1))  # (1,2,1024)
        kvk = torch.zeros(NL,1,NKV,self.cap,HD); kvv = torch.zeros(NL,1,NKV,self.cap,HD)
        outk = []; outv = []
        h = x
        for i in range(NL):
            h, ki, vi = layer_attn(self.core, i, h, rope_cos, rope_sin, kvk, kvv, slot_mask, attn_mask, NH//NKV)
            outk.append(ki.unsqueeze(0)); outv.append(vi.unsqueeze(0))
        h = self.core.norm(h)
        return h[:, -1:, :], torch.cat(outk,0), torch.cat(outv,0)

class Step(torch.nn.Module):
    def __init__(self, cp, core, cap, scale):
        super().__init__(); self.cp=cp; self.core=core; self.cap=cap; self.scale=scale
    def forward(self, token_emb, kv_k, kv_v, rope_cos, rope_sin, slot_mask, attn_mask):
        x = self.cp.small_to_mtp_projection(token_emb)     # (1,1,1024)
        outk = []; outv = []
        h = x
        for i in range(NL):
            h, ki, vi = layer_attn(self.core, i, h, rope_cos, rope_sin, kv_k, kv_v, slot_mask, attn_mask, NH//NKV)
            outk.append(ki.unsqueeze(0)); outv.append(vi.unsqueeze(0))
        h = self.core.norm(h)
        return h, torch.cat(outk,0), torch.cat(outv,0)

pref = Prefill(cp, core, CAP, SCALE).eval()
stepm = Step(cp, core, CAP, SCALE).eval()

# dummy IO
ph = torch.zeros(1,1,2048); c0e = torch.zeros(1,1,2048)
pc = torch.zeros(1,1,2,HD); ps = torch.zeros(1,1,2,HD)
sm2 = torch.zeros(1,1,CAP,1); sm2[0,0,0,0]=1; sm2[0,0,1,0]=1
am2 = torch.full((1,1,2,CAP), -1e4); am2[0,0,0,0]=0; am2[0,0,1,0]=0; am2[0,0,0,1]=0; am2[0,0,1,1]=0
kk0 = torch.zeros(NL,1,NKV,CAP,HD); vv0 = torch.zeros(NL,1,NKV,CAP,HD)
with torch.inference_mode():
    o = pref(ph, c0e, pc, ps, sm2, am2)
print('prefill out', [tuple(t.shape) for t in o])
torch.onnx.export(pref, (ph, c0e, pc, ps, sm2, am2), OUT + '/codepred_prefill.onnx',
    input_names=['past_hidden','code0_emb','rope_cos','rope_sin','slot_mask','attn_mask'],
    output_names=['hidden_last','kv_k_out','kv_v_out'], opset_version=18, dynamo=False)

ts = torch.zeros(1,1,2048); sc = torch.zeros(1,1,1,HD); ss = torch.zeros(1,1,1,HD)
sm1 = torch.zeros(1,1,CAP,1); sm1[0,0,2,0]=1
am1 = torch.full((1,1,1,CAP), -1e4); am1[0,0,0,0]=0; am1[0,0,0,1]=0; am1[0,0,0,2]=0
with torch.inference_mode():
    o2 = stepm(ts, kk0, vv0, sc, ss, sm1, am1)
print('step out', [tuple(t.shape) for t in o2])
torch.onnx.export(stepm, (ts, kk0, vv0, sc, ss, sm1, am1), OUT + '/codepred_step.onnx',
    input_names=['token_emb','kv_k','kv_v','rope_cos','rope_sin','slot_mask','attn_mask'],
    output_names=['hidden_last','kv_k_out','kv_v_out'], opset_version=18, dynamo=False)

# host-side lm_head + embedding weights
lw = np.stack([cp.lm_head[g].weight.detach().numpy().astype(np.float32) for g in range(15)])
ew = np.stack([core.codec_embedding[g].weight.detach().numpy().astype(np.float32) for g in range(15)])
lw.tofile(OUT + '/lm_head_weight.f32')      # (15,2048,1024)
ew.tofile(OUT + '/codec_emb_weight.f32')    # (15,2048,2048)
print('lm_head', lw.shape, 'codec_emb', ew.shape)
import onnx
for f in ['codepred_prefill.onnx','codepred_step.onnx']:
    m = onnx.load(OUT + '/' + f, load_external_data=False)
    ops = {}
    for n in m.graph.node: ops[n.op_type] = ops.get(n.op_type,0)+1
    print('%-24s IsNaN=%d ArgMax=%d ScatterND=%d nodes=%d size=%d' % (f, ops.get('IsNaN',0), ops.get('ArgMax',0), ops.get('ScatterND',0), len(m.graph.node), os.path.getsize(OUT+'/'+f)))
print('CPR1_EXPORT_DONE')
