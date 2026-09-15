# CPU-side reference for G2-B margin: replicate g2b_loop2 EXACTLY on PC (torch),
# including the graph's greedy-prefix codepred semantics, with the same LCG.
import os, numpy as np, torch, math
os.environ['HF_HUB_OFFLINE'] = '1'
D = r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/device-smoke'
OUT = D + '/vd_p2'; G2 = OUT + '/g2'
MASK64 = (1 << 64) - 1
class LCG:
    def __init__(self, seed=123456789): self.s = seed & MASK64
    def u(self):
        self.s = (self.s * 6364136223846793005 + 1442695040888963407) & MASK64
        return ((self.s >> 11) & ((1 << 53) - 1)) / float(1 << 53)
def sample_topk(lg, temp, topk, suppress, u):
    idx = np.arange(lg.size)
    if suppress:
        keep = ~((idx >= 2048) & (idx != 2150)); idx = idx[keep]; vals = lg[keep] / temp
    else:
        vals = lg / temp
    order = np.argsort(-vals, kind='stable')
    idx = idx[order][:topk]; vals = vals[order][:topk]
    mx = vals.max(); p = np.exp(vals - mx); p /= p.sum()
    acc = np.cumsum(p)
    j = int(np.searchsorted(acc, u, side='left'))
    if j >= p.size: j = p.size - 1
    lo = float(acc[j] - p[j]); hi = float(acc[j])
    return int(idx[j]), lo, hi
from qwen_tts import Qwen3TTSModel
gw = Qwen3TTSModel.from_pretrained(r'E:/AndroidStudioProjects/qwen3tts-mnn/models/qwen3-tts/Qwen3-TTS-12Hz-1.7B-VoiceDesign', dtype=torch.float32, device_map='cpu')
talker = gw.model.talker; base = talker.model
cp = talker.code_predictor
a0 = base.layers[0].self_attn
SCALE = 1.0/math.sqrt(a0.head_dim); MAXKV = 768
def rotate_half(x):
    x1 = x[..., : x.shape[-1]//2]; x2 = x[..., x.shape[-1]//2:]
    return torch.cat((-x2, x1), dim=-1)
import onnxruntime as ort
so_fe = ort.InferenceSession(D + '/vd_m5b/frame_emb.onnx', providers=['CPUExecutionProvider'])
hidden = np.fromfile(D + '/vd_m5b/prefill_hidden_last.raw', dtype=np.float32)
kvk = np.fromfile(G2 + '/init_kv_k.raw', dtype=np.float32).reshape(28,1,8,MAXKV,128)
kvv = np.fromfile(G2 + '/init_kv_v.raw', dtype=np.float32).reshape(28,1,8,MAXKV,128)
cos_t = np.fromfile(G2 + '/cos300.raw', dtype=np.float32).reshape(300,1,1,128)
sin_t = np.fromfile(G2 + '/sin300.raw', dtype=np.float32).reshape(300,1,1,128)
rng = LCG()
lines = []
with torch.no_grad():
    hv = torch.from_numpy(hidden.copy())
    for n in range(3):
        cp_pos = 62 + n
        lg = talker.codec_head(hv.view(1,1,2048))[0,0].numpy().astype(np.float32)
        u0 = rng.u(); c0, lo0, hi0 = sample_topk(lg, 0.9, 50, True, u0)
        c0e = talker.get_input_embeddings()(torch.tensor([[c0]]))
        seq = torch.cat([hv.view(1,1,2048), c0e], dim=1)
        codes = [c0]; us = [u0]; los = [lo0]; his = [hi0]
        lg15acc = np.zeros((15,2048), np.float32)
        for g in range(15):
            h = cp.model(inputs_embeds=cp.small_to_mtp_projection(seq), use_cache=False).last_hidden_state
            lgg = cp.lm_head[g](h[:, -1:])[0,0].numpy().astype(np.float32)
            ug = rng.u()
            cg, lo, hi = sample_topk(lgg, 0.9, 50, False, ug)
            lg15acc[g] = lgg
            codes.append(cg); us.append(ug); los.append(lo); his.append(hi)
            if g < 14:
                greedy = int(np.argmax(lgg))          # ← 复现图里的 ArgMax 贪心前缀
                seq = torch.cat([seq, cp.model.codec_embedding[g](torch.tensor([[greedy]]))], dim=1)
        lg15acc.tofile(OUT + '/cpu_step%d_lg15.raw' % n)
        emb = so_fe.run(None, {'codes': np.array([codes], dtype=np.int64)})[0].astype(np.float32)
        h = torch.from_numpy(emb)
        rc = torch.from_numpy(cos_t[n]); rs = torch.from_numpy(sin_t[n])
        am = np.full((1,1,1,MAXKV), -1e4, np.float32); am[0,0,0,:cp_pos+1] = 0.0
        sm = np.zeros((1,1,MAXKV,1), np.float32); sm[0,0,cp_pos,0] = 1.0
        attn_mask = torch.from_numpy(am); slot_mask = torch.from_numpy(sm); one_minus = 1.0 - slot_mask
        cks = []; cvs = []
        kk = torch.from_numpy(kvk.copy()); vv = torch.from_numpy(kvv.copy())
        for i in range(28):
            layer = base.layers[i]; residual = h
            hn = layer.input_layernorm(h); at = layer.self_attn
            hd = at.head_dim; hs = hn.shape[:-1]; shp = (*hs,-1,hd)
            q = at.q_norm(at.q_proj(hn).view(shp)).transpose(1,2)
            k = at.k_norm(at.k_proj(hn).view(shp)).transpose(1,2)
            v = at.v_proj(hn).view(shp).transpose(1,2)
            q_r = q * rc + rotate_half(q) * rs
            k_r = k * rc + rotate_half(k) * rs
            kvk_i = kk[i] * one_minus + k_r * slot_mask
            kvv_i = vv[i] * one_minus + v * slot_mask
            cks.append(k_r.reshape(1,1,8,128)); cvs.append(v.reshape(1,1,8,128))
            g2 = at.num_key_value_groups
            k_all = kvk_i.repeat_interleave(g2, dim=1); v_all = kvv_i.repeat_interleave(g2, dim=1)
            logits = torch.matmul(q_r, k_all.transpose(-1,-2)) * SCALE
            probs = torch.softmax(logits + attn_mask, dim=-1)
            ao = torch.matmul(probs, v_all).reshape(*hs,-1).contiguous()
            h2 = residual + at.o_proj(ao); r2 = h2
            h2 = layer.post_attention_layernorm(h2)
            h = r2 + layer.mlp(h2)
        h = base.norm(h)
        ck = torch.cat(cks, dim=0).numpy(); cvt = torch.cat(cvs, dim=0).numpy()
        for l in range(28):
            for hh in range(8):
                kvk[l,0,hh,cp_pos,:] = ck[l,0,hh,:]; kvv[l,0,hh,cp_pos,:] = cvt[l,0,hh,:]
        hv = h.view(2048)
        line = 'CPU step %d cp %d code0 %d %s | u %s | cdf %s' % (
            n, cp_pos, c0, ' '.join(str(x) for x in codes),
            ' '.join('%.9f' % x for x in us),
            ' '.join('%.6f %.6f' % (los[i], his[i]) for i in range(16)))
        lines.append(line); print(line[:180]); print('   ...(truncated)')
with open(OUT + '/cpu_margin_ref.txt', 'w') as f:
    f.write('\n'.join(lines) + '\n')
# margin table
print()
print('=== CPU margins (step1) ===')
ln = lines[1]
cdf_part = ln.split(' | cdf ')[1].split()
u_part = ln.split(' | u ')[1].split(' | ')[0].split()
for i in range(16):
    lo = float(cdf_part[2*i]); hi = float(cdf_part[2*i+1]); u = float(u_part[i])
    d = min(u - lo, hi - u)
    if i <= 3: print('  code_index %2d  u=%.9f  [%.6f, %.6f)  dist=%.9f' % (i, u, lo, hi, d))
print('CPU_MARGIN_DONE')
