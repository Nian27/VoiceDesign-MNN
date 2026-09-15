# debug export: expose layer-0 internals of the prefill graph
import os, math, numpy as np, torch
os.environ['HF_HUB_OFFLINE'] = '1'
D = r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/device-smoke'; CR = D + '/vd_cpr'
from qwen_tts import Qwen3TTSModel
gw = Qwen3TTSModel.from_pretrained(r'E:/AndroidStudioProjects/qwen3tts-mnn/models/qwen3-tts/Qwen3-TTS-12Hz-1.7B-VoiceDesign', dtype=torch.float32, device_map='cpu')
cp = gw.model.talker.code_predictor; core = cp.model
NL, NH, NKV, HD, CAP = 5, 16, 8, 128, 16
SCALE = 1.0/math.sqrt(HD)
def rh(x):
    x1 = x[..., : x.shape[-1]//2]; x2 = x[..., x.shape[-1]//2:]
    return torch.cat((-x2, x1), dim=-1)
class Dbg(torch.nn.Module):
    def __init__(self, cp, core):
        super().__init__(); self.cp = cp; self.core = core
    def forward(self, past_hidden, code0_emb, rope_cos, rope_sin, slot_mask0, slot_mask1, attn_mask):
        x = self.cp.small_to_mtp_projection(torch.cat([past_hidden, code0_emb], dim=1))
        layer = self.core.layers[0]
        residual = x
        hn = layer.input_layernorm(x)
        at = layer.self_attn
        hs = hn.shape[:-1]; shp = (*hs,-1,HD)
        q = at.q_norm(at.q_proj(hn).view(shp)).transpose(1,2)
        k = at.k_norm(at.k_proj(hn).view(shp)).transpose(1,2)
        v = at.v_proj(hn).view(shp).transpose(1,2)
        qr = q*rope_cos + rh(q)*rope_sin
        kr = k*rope_cos + rh(k)*rope_sin
        kvk = torch.zeros(1,NKV,CAP,HD); kvv = torch.zeros(1,NKV,CAP,HD)
        kvk = kvk*(1.0-slot_mask0-slot_mask1) + kr[:,:,0:1,:]*slot_mask0 + kr[:,:,1:2,:]*slot_mask1
        kvv = kvv*(1.0-slot_mask0-slot_mask1) + v[:,:,0:1,:]*slot_mask0 + v[:,:,1:2,:]*slot_mask1
        g = NH//NKV
        k_all = kvk.repeat_interleave(g, dim=1); v_all = kvv.repeat_interleave(g, dim=1)
        scores = torch.matmul(qr, k_all.transpose(-1,-2)) * SCALE
        masked = scores + attn_mask
        probs = torch.softmax(masked, dim=-1)
        ao = torch.matmul(probs, v_all)
        ao_f = ao.reshape(*hs,-1).contiguous()
        oo = at.o_proj(ao_f)
        h1 = residual + oo
        r2 = h1
        h1n = layer.post_attention_layernorm(h1)
        h2 = r2 + layer.mlp(h1n)
        return (q, k, kvk, scores, probs, ao_f, oo, h1, h2, self.core.norm(h2))
m = Dbg(cp, core).eval()
ph = torch.zeros(1,1,2048); c0e = torch.zeros(1,1,2048)
pc = torch.zeros(1,1,2,HD); ps = torch.zeros(1,1,2,HD)
sm0 = torch.zeros(1,1,CAP,1); sm0[0,0,0,0]=1
sm1 = torch.zeros(1,1,CAP,1); sm1[0,0,1,0]=1
am = torch.full((1,1,2,CAP), -1e4); am[0,0,0,0]=0; am[0,0,1,0]=0; am[0,0,1,1]=0
with torch.inference_mode():
    o = m(ph, c0e, pc, ps, sm0, sm1, am)
names = ['q','k','kvk','scores','probs','ao_f','oo','h1','h2','out']
torch.onnx.export(m, (ph,c0e,pc,ps,sm0,sm1,am), CR + '/dbg_prefill.onnx',
    input_names=['past_hidden','code0_emb','rope_cos','rope_sin','slot_mask0','slot_mask1','attn_mask'],
    output_names=names, opset_version=18, dynamo=False)
print('dbg export ok', [tuple(t.shape) for t in o][:5])
print('DBG_EXPORT_DONE')
