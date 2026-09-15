# B core fix: replace strided scatter with broadcast multiply-add (slot_mask from host)
import os, numpy as np, torch, math
os.environ['HF_HUB_OFFLINE'] = '1'
D = r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/device-smoke'
OUT = D + '/vd_p2'
MASK_NEG = -1e4
from qwen_tts import Qwen3TTSModel
gw = Qwen3TTSModel.from_pretrained(r'E:/AndroidStudioProjects/qwen3tts-mnn/models/qwen3-tts/Qwen3-TTS-12Hz-1.7B-VoiceDesign', dtype=torch.float32, device_map='cpu')
base = gw.model.talker.model
attn0 = base.layers[0].self_attn
ms = attn0.rope_scaling['mrope_section']; MOD = 3; MAXKV = 768
LI = 0
def rotate_half(x):
    x1 = x[..., : x.shape[-1]//2]; x2 = x[..., x.shape[-1]//2:]
    return torch.cat((-x2, x1), dim=-1)
def host_merge(c3, s3):
    half = c3.shape[-1] // 2
    def merge(x):
        x_t = x[0].clone()
        for i, n in enumerate(ms[1:], 1):
            x_t[..., i:n*MOD:MOD] = x[i, ..., i:n*MOD:MOD]
        return x_t
    return torch.cat([merge(c3[..., :half])] * 2, dim=-1), torch.cat([merge(s3[..., :half])] * 2, dim=-1)

cp0 = 62
slot_mask = np.zeros((1,1,MAXKV,1), np.float32); slot_mask[0,0,cp0,0] = 1.0
slot_mask.tofile(OUT + '/slot_mask.raw')
print('slot_mask saved, ones at', cp0)

class GraphB1LV5(torch.nn.Module):
    def __init__(self, base, max_kv, li, scale):
        super().__init__(); self.base=base; self.max_kv=max_kv; self.li=li; self.scale=scale
        lm = torch.zeros(28,1,1,1,1); lm[li] = 1.0
        self.register_buffer('layer_onehot', lm)
    def forward(self, inputs_embeds, cache_position, kv_k, kv_v, rope_cos, rope_sin, attn_mask, slot_mask):
        hidden = inputs_embeds
        layer = self.base.layers[self.li]
        residual = hidden
        norm_out = layer.input_layernorm(hidden)
        a = layer.self_attn
        hd = a.head_dim; hs = norm_out.shape[:-1]; shp = (*hs,-1,hd)
        q_raw = a.q_proj(norm_out).view(shp); k_raw = a.k_proj(norm_out).view(shp); v_raw = a.v_proj(norm_out).view(shp)
        q_norm = a.q_norm(q_raw); k_norm = a.k_norm(k_raw)
        q = q_norm.transpose(1,2); k = k_norm.transpose(1,2); v = v_raw.transpose(1,2)
        q_rope = q * rope_cos + rotate_half(q) * rope_sin          # (1,16,1,128)
        k_rope = k * rope_cos + rotate_half(k) * rope_sin          # (1,8,1,128)
        # ---- B core: broadcast multiply-add instead of scatter ----
        kvk_out = kv_k + (k_rope.view(1,1,8,1,128) * self.layer_onehot * slot_mask)
        kvv_out = kv_v + (v.view(1,1,8,1,128) * self.layer_onehot * slot_mask)
        g = a.num_key_value_groups
        k_all = kvk_out[self.li].repeat_interleave(g, dim=1)        # (1,16,768,128)
        v_all = kvv_out[self.li].repeat_interleave(g, dim=1)
        k_T = k_all.transpose(-1,-2)
        logits = torch.matmul(q_rope, k_T) * self.scale
        masked = logits + attn_mask
        probs = torch.softmax(masked, dim=-1)
        attn_pre = torch.matmul(probs, v_all)
        attn_out = attn_pre.reshape(*hs,-1).contiguous()
        o_proj_out = a.o_proj(attn_out)
        res1 = residual + o_proj_out
        post_norm = layer.post_attention_layernorm(res1)
        mlp_out = layer.mlp(post_norm)
        layer0_out = res1 + mlp_out
        kvk_slot = kvk_out[:,:,:,cache_position[0],:]               # (28,1,8,128)
        kvv_slot = kvv_out[:,:,:,cache_position[0],:]
        kvk_old0 = kvk_out[0,:,:,:cache_position[0],:]              # (1,8,62,128)
        kvv_old0 = kvv_out[0,:,:,:cache_position[0],:]
        return (norm_out,q_rope,k_rope,k_all,k_T,logits,masked,probs,attn_out,layer0_out,
                kvk_slot,kvv_slot,kvk_old0,kvv_old0)
names = ['norm_out','q_rope','k_rope','k_all','k_T','qk_logits','qk_masked','attn_probs','attn_out','layer0_out',
         'kv_k_slot','kv_v_slot','kv_k_old0','kv_v_old0']
core = GraphB1LV5(base, MAXKV, LI, 1.0/math.sqrt(attn0.head_dim)).eval()
emb = np.fromfile(D + '/vd_m5a_inputs/step0_emb.raw', dtype=np.float32).reshape(1,1,2048)
pos = np.fromfile(D + '/vd_m5a_inputs/step0_pos.raw', dtype=np.float32).reshape(3,1,1)
kvk_in = np.fromfile(D + '/vd_m5a_inputs/step0_kvk.raw', dtype=np.float32).reshape(28,1,8,MAXKV,128)
kvv_in = np.fromfile(D + '/vd_m5a_inputs/step0_kvv.raw', dtype=np.float32).reshape(28,1,8,MAXKV,128)
mask1 = np.where(np.arange(MAXKV) <= cp0, 0.0, MASK_NEG).astype(np.float32).reshape(1,1,1,MAXKV)
mask1.tofile(OUT + '/attn_mask_1e4.raw')
with torch.no_grad():
    hidden = torch.from_numpy(emb)
    c3, s3 = base.rotary_emb(hidden, torch.from_numpy(pos))
    mc, msin = host_merge(c3, s3)
    refs = core(hidden, torch.tensor([cp0]), torch.from_numpy(kvk_in), torch.from_numpy(kvv_in),
                mc, msin, torch.from_numpy(mask1), torch.from_numpy(slot_mask))
for nm, t in zip(names, refs):
    t.detach().numpy().astype(np.float32).tofile(D + '/vd_p1/ref_' + nm + '.raw')
print('refs saved:', len(names))
# sanity: layer0_out must still match the official reference
ref0 = np.fromfile(D + '/vd_p1/ref_layer0_out.raw', dtype=np.float32)
a = refs[names.index('layer0_out')].numpy()
print('layer0_out(v5) vs earlier ref cos=%.8f' % float(np.dot(a.ravel().astype(np.float64), ref0.astype(np.float64))/(np.linalg.norm(a.ravel())*np.linalg.norm(ref0)+1e-12)))
e = torch.zeros((1,1,2048)); cp_ = torch.tensor([cp0], dtype=torch.int64)
kk = torch.zeros((28,1,8,MAXKV,128)); vv = torch.zeros((28,1,8,MAXKV,128))
mc2 = torch.zeros((1,1,128)); ms2 = torch.zeros((1,1,128)); am = torch.zeros((1,1,1,MAXKV)); sm = torch.zeros((1,1,MAXKV,1))
with torch.inference_mode():
    core(e, cp_, kk, vv, mc2, ms2, am, sm)
torch.onnx.export(core, (e, cp_, kk, vv, mc2, ms2, am, sm), OUT + '/graphb_1l_v5.onnx',
                  input_names=['inputs_embeds','cache_position','kv_k','kv_v','rope_cos','rope_sin','attn_mask','slot_mask'],
                  output_names=names, opset_version=18, dynamo=False)
import onnx
mm = onnx.load(OUT + '/graphb_1l_v5.onnx', load_external_data=False)
ops = {}
for n in mm.graph.node: ops[n.op_type] = ops.get(n.op_type,0)+1
print('v5 ScatterND', ops.get('ScatterND',0), 'IsNaN', ops.get('IsNaN',0), 'size', os.path.getsize(OUT + '/graphb_1l_v5.onnx'))
print('B_V5_EXPORT_DONE')
