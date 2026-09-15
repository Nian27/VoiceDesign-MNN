# A0: GraphB-1L-DIAG export + torch boundary references for layer 0
import os, numpy as np, torch
os.environ['HF_HUB_OFFLINE'] = '1'
import torch.nn.functional as F
D = r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/device-smoke'
OUT = D + '/vd_p1'
os.makedirs(OUT, exist_ok=True)
from qwen_tts import Qwen3TTSModel
from qwen_tts.core.models.modeling_qwen3_tts import apply_multimodal_rotary_pos_emb
gw = Qwen3TTSModel.from_pretrained(r'E:/AndroidStudioProjects/qwen3tts-mnn/models/qwen3-tts/Qwen3-TTS-12Hz-1.7B-VoiceDesign', dtype=torch.float32, device_map='cpu')
base = gw.model.talker.model
MAX_KV = 768
LAYER = 0

class GraphB1LDiag(torch.nn.Module):
    def __init__(self, base, max_kv, li):
        super().__init__(); self.base = base; self.max_kv = max_kv; self.li = li
    def forward(self, inputs_embeds, position_ids, cache_position, kv_k, kv_v):
        hidden = inputs_embeds
        pos_emb = self.base.rotary_emb(hidden, position_ids)
        kvk_out = kv_k.clone(); kvv_out = kv_v.clone()
        layer = self.base.layers[self.li]
        residual = hidden
        norm_out = layer.input_layernorm(hidden)
        attn = layer.self_attn
        hd = attn.head_dim; hs = norm_out.shape[:-1]; shp = (*hs, -1, hd)
        q_raw = attn.q_proj(norm_out).view(shp)
        k_raw = attn.k_proj(norm_out).view(shp)
        v_raw = attn.v_proj(norm_out).view(shp)
        q_norm = attn.q_norm(q_raw)
        k_norm = attn.k_norm(k_raw)
        q = q_norm.transpose(1,2); k = k_norm.transpose(1,2); v = v_raw.transpose(1,2)
        cos, sin = pos_emb
        q_rope, k_rope = apply_multimodal_rotary_pos_emb(q, k, cos, sin, attn.rope_scaling['mrope_section'], attn.rope_scaling['interleaved'])
        kvk_out[self.li, :, :, cache_position, :] = k_rope[:, :, 0, :].unsqueeze(2)
        kvv_out[self.li, :, :, cache_position, :] = v[:, :, 0, :].unsqueeze(2)
        g = attn.num_key_value_groups
        k_all = kvk_out[self.li].repeat_interleave(g, dim=1)
        v_all = kvv_out[self.li].repeat_interleave(g, dim=1)
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

core = GraphB1LDiag(base, MAX_KV, LAYER).eval()
emb = torch.zeros((1,1,2048)); pos = torch.zeros((3,1,1)); pos[0,0,0]=62; pos[1,0,0]=62; pos[2,0,0]=62
cp_ = torch.tensor([62], dtype=torch.int64)
kk = torch.zeros((28,1,8,MAX_KV,128)); vv = torch.zeros((28,1,8,MAX_KV,128))
names = ['norm_out','q_raw','k_raw','v_raw','q_norm','k_norm','q_rope','k_rope','attn_out','o_proj_out','res1','post_norm','mlp_out','layer0_out','kv_k_out','kv_v_out']
with torch.inference_mode():
    outs = core(emb,pos,cp_,kk,vv)
print('n_out', len(outs), [tuple(o.shape) for o in outs[:4]])
torch.onnx.export(core, (emb,pos,cp_,kk,vv), OUT + '/graphb_1l_diag.onnx',
                  input_names=['inputs_embeds','position_ids','cache_position','kv_k','kv_v'],
                  output_names=names, opset_version=18, dynamo=False)
import onnx
mm = onnx.load(OUT + '/graphb_1l_diag.onnx', load_external_data=False)
ops = {}
for n in mm.graph.node: ops[n.op_type] = ops.get(n.op_type,0)+1
print('IsNaN', ops.get('IsNaN',0), 'size', os.path.getsize(OUT + '/graphb_1l_diag.onnx'))

# torch boundary reference with the REAL step0 input
emb_r = np.fromfile(D + '/vd_m5a_inputs/step0_emb.raw', dtype=np.float32).reshape(1,1,2048)
pos_r = np.fromfile(D + '/vd_m5a_inputs/step0_pos.raw', dtype=np.float32).reshape(3,1,1)
cp_r = int(np.fromfile(D + '/vd_m5a_inputs/step0_cp.raw', dtype=np.int32)[0])
kvk_r = np.fromfile(D + '/vd_m5a_inputs/step0_kvk.raw', dtype=np.float32).reshape(28,1,8,768,128)
kvv_r = np.fromfile(D + '/vd_m5a_inputs/step0_kvv.raw', dtype=np.float32).reshape(28,1,8,768,128)
with torch.inference_mode():
    refs = core(torch.from_numpy(emb_r), torch.from_numpy(pos_r), torch.tensor([cp_r]),
                torch.from_numpy(kvk_r), torch.from_numpy(kvv_r))
for nm, t in zip(names, refs):
    arr = t.detach().numpy().astype(np.float32)
    arr.tofile(OUT + '/ref_' + nm + '.raw')
    print('ref %-12s shape %-20s maxabs %.4f f4 %s' % (nm, str(arr.shape), np.abs(arr).max(), arr.ravel()[:3]))
print('A0_DONE')
