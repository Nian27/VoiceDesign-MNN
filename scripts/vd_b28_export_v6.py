# 28L GraphB v6: per-layer slice OVERWRITE (kv*(1-m)+new*m), outputs hidden_states + cur_k/cur_v
import os, numpy as np, torch, math
os.environ['HF_HUB_OFFLINE'] = '1'
D = r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/device-smoke'
OUT = D + '/vd_p2'
from qwen_tts import Qwen3TTSModel
gw = Qwen3TTSModel.from_pretrained(r'E:/AndroidStudioProjects/qwen3tts-mnn/models/qwen3-tts/Qwen3-TTS-12Hz-1.7B-VoiceDesign', dtype=torch.float32, device_map='cpu')
base = gw.model.talker.model
a0 = base.layers[0].self_attn
SCALE = 1.0/math.sqrt(a0.head_dim); MAXKV = 768
def rotate_half(x):
    x1 = x[..., : x.shape[-1]//2]; x2 = x[..., x.shape[-1]//2:]
    return torch.cat((-x2, x1), dim=-1)

class GraphB28V6(torch.nn.Module):
    def __init__(self, base, max_kv, scale):
        super().__init__(); self.base=base; self.max_kv=max_kv; self.scale=scale
    def forward(self, inputs_embeds, cache_position, kv_k, kv_v, rope_cos, rope_sin, attn_mask, slot_mask):
        hidden = inputs_embeds
        one_minus = 1.0 - slot_mask                     # (1,1,768,1)
        cur_ks = []; cur_vs = []
        for i in range(28):
            layer = self.base.layers[i]
            residual = hidden
            h = layer.input_layernorm(hidden)
            attn = layer.self_attn
            hd = attn.head_dim; hs = h.shape[:-1]; shp = (*hs, -1, hd)
            q = attn.q_norm(attn.q_proj(h).view(shp)).transpose(1,2)
            k = attn.k_norm(attn.k_proj(h).view(shp)).transpose(1,2)
            v = attn.v_proj(h).view(shp).transpose(1,2)
            q_rope = q * rope_cos + rotate_half(q) * rope_sin
            k_rope = k * rope_cos + rotate_half(k) * rope_sin
            # ---- per-layer slice OVERWRITE (no scatter) ----
            kvk_i = kv_k[i] * one_minus + k_rope * slot_mask      # (1,8,768,128)
            kvv_i = kv_v[i] * one_minus + v * slot_mask
            cur_ks.append(k_rope.reshape(1,1,8,128))
            cur_vs.append(v.reshape(1,1,8,128))
            g = attn.num_key_value_groups
            k_all = kvk_i.repeat_interleave(g, dim=1)
            v_all = kvv_i.repeat_interleave(g, dim=1)
            logits = torch.matmul(q_rope, k_all.transpose(-1,-2)) * self.scale
            probs = torch.softmax(logits + attn_mask, dim=-1)
            ao = torch.matmul(probs, v_all).reshape(*hs, -1).contiguous()
            h2 = residual + attn.o_proj(ao)
            r2 = h2
            h2 = layer.post_attention_layernorm(h2)
            hidden = r2 + layer.mlp(h2)
        hidden = self.base.norm(hidden)
        return hidden, torch.cat(cur_ks, dim=0), torch.cat(cur_vs, dim=0)

core = GraphB28V6(base, MAXKV, SCALE).eval()
emb = torch.zeros((1,1,2048)); cp_ = torch.tensor([62], dtype=torch.int64)
kk = torch.zeros((28,1,8,MAXKV,128)); vv = torch.zeros((28,1,8,MAXKV,128))
mc = torch.zeros((1,1,128)); ms = torch.zeros((1,1,128))
am = torch.zeros((1,1,1,MAXKV)); sm = torch.zeros((1,1,MAXKV,1))
with torch.inference_mode():
    h, ck, cv = core(emb, cp_, kk, vv, mc, ms, am, sm)
print('shapes', tuple(h.shape), tuple(ck.shape), tuple(cv.shape))
torch.onnx.export(core, (emb, cp_, kk, vv, mc, ms, am, sm), OUT + '/graphb28_v6.onnx',
                  input_names=['inputs_embeds','cache_position','kv_k','kv_v','rope_cos','rope_sin','attn_mask','slot_mask'],
                  output_names=['hidden_states','cur_k','cur_v'], opset_version=18, dynamo=False)
import onnx
mm = onnx.load(OUT + '/graphb28_v6.onnx', load_external_data=False)
ops = {}
for n in mm.graph.node: ops[n.op_type] = ops.get(n.op_type,0)+1
print('v6 ScatterND', ops.get('ScatterND',0), 'IsNaN', ops.get('IsNaN',0), 'size', os.path.getsize(OUT + '/graphb28_v6.onnx'))
print('V6_EXPORT_DONE')
