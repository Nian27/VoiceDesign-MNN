# G1 torch reference: 28L v6 with the real frozen step0 input
import os, gc, numpy as np, torch, math
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
emb = np.fromfile(D + '/vd_m5a_inputs/step0_emb.raw', dtype=np.float32).reshape(1,1,2048)
kvk = np.fromfile(D + '/vd_m5a_inputs/step0_kvk.raw', dtype=np.float32).reshape(28,1,8,MAXKV,128)
kvv = np.fromfile(D + '/vd_m5a_inputs/step0_kvv.raw', dtype=np.float32).reshape(28,1,8,MAXKV,128)
mc = np.fromfile(OUT + '/merged_cos.raw', dtype=np.float32).reshape(1,1,128)
ms = np.fromfile(OUT + '/merged_sin.raw', dtype=np.float32).reshape(1,1,128)
am = np.fromfile(OUT + '/attn_mask_1e4.raw', dtype=np.float32).reshape(1,1,1,MAXKV)
sm = np.fromfile(OUT + '/slot_mask.raw', dtype=np.float32).reshape(1,1,MAXKV,1)
with torch.no_grad():
    hidden = torch.from_numpy(emb)
    kk = torch.from_numpy(kvk); vv = torch.from_numpy(kvv)
    rc = torch.from_numpy(mc); rs = torch.from_numpy(ms)
    attn_mask = torch.from_numpy(am); slot_mask = torch.from_numpy(sm)
    one_minus = 1.0 - slot_mask
    cks = []; cvs = []
    for i in range(28):
        layer = base.layers[i]
        residual = hidden
        h = layer.input_layernorm(hidden)
        a = layer.self_attn
        hd = a.head_dim; hs = h.shape[:-1]; shp = (*hs,-1,hd)
        q = a.q_norm(a.q_proj(h).view(shp)).transpose(1,2)
        k = a.k_norm(a.k_proj(h).view(shp)).transpose(1,2)
        v = a.v_proj(h).view(shp).transpose(1,2)
        q_r = q * rc + rotate_half(q) * rs
        k_r = k * rc + rotate_half(k) * rs
        kvk_i = kk[i] * one_minus + k_r * slot_mask
        kvv_i = vv[i] * one_minus + v * slot_mask
        gc.collect()
        cks.append(k_r.reshape(1,1,8,128)); cvs.append(v.reshape(1,1,8,128))
        g = a.num_key_value_groups
        k_all = kvk_i.repeat_interleave(g, dim=1); v_all = kvv_i.repeat_interleave(g, dim=1)
        logits = torch.matmul(q_r, k_all.transpose(-1,-2)) * SCALE
        probs = torch.softmax(logits + attn_mask, dim=-1)
        ao = torch.matmul(probs, v_all).reshape(*hs,-1).contiguous()
        h2 = residual + a.o_proj(ao); r2 = h2
        h2 = layer.post_attention_layernorm(h2)
        hidden = r2 + layer.mlp(h2)
    hidden = base.norm(hidden)
    ck = torch.cat(cks, dim=0); cv = torch.cat(cvs, dim=0)
for nm, t in [('hidden_states',hidden),('cur_k',ck),('cur_v',cv)]:
    arr = t.numpy().astype(np.float32); arr.tofile(OUT + '/ref28_' + nm + '.raw')
    print('ref28 %-13s shape %-20s maxabs %.5f' % (nm, str(tuple(arr.shape)), float(np.abs(arr).max())))
print('G1_REF_DONE')
