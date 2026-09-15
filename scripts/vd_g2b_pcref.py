# G2-B PC reference: replicate g2b_loop2's exact loop (same LCG) with torch
import os, numpy as np, torch, math
os.environ['HF_HUB_OFFLINE'] = '1'
D = r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/device-smoke'
OUT = D + '/vd_p2'
G2 = OUT + '/g2'
MASK64 = (1 << 64) - 1

class LCG:
    def __init__(self, seed=123456789):
        self.s = seed & MASK64
    def u(self):
        self.s = (self.s * 6364136223846793005 + 1442695040888963407) & MASK64
        return ((self.s >> 11) & ((1 << 53) - 1)) / float(1 << 53)

def sample_topk(lg, temp, topk, suppress, u):
    idx = np.arange(lg.size)
    if suppress:
        keep = ~((idx >= 2048) & (idx != 2150))
        idx = idx[keep]; vals = lg[keep] / temp
    else:
        vals = lg / temp
    order = np.argsort(-vals, kind='stable')
    idx = idx[order][:topk]; vals = vals[order][:topk]
    mx = vals.max(); p = np.exp(vals - mx); p /= p.sum()
    acc = np.cumsum(p)
    j = int(np.searchsorted(acc, u, side='left'))
    if j >= p.size: j = p.size - 1
    return int(idx[j]), float(acc[j] - p[j]), float(acc[j])

from qwen_tts import Qwen3TTSModel
gw = Qwen3TTSModel.from_pretrained(r'E:/AndroidStudioProjects/qwen3tts-mnn/models/qwen3-tts/Qwen3-TTS-12Hz-1.7B-VoiceDesign', dtype=torch.float32, device_map='cpu')
base = gw.model.talker.model
a0 = base.layers[0].self_attn
SCALE = 1.0/math.sqrt(a0.head_dim); MAXKV = 768
def rotate_half(x):
    x1 = x[..., : x.shape[-1]//2]; x2 = x[..., x.shape[-1]//2:]
    return torch.cat((-x2, x1), dim=-1)

# codec_head / frame_emb via onnxruntime (already verified == MNN)
import onnxruntime as ort
so_ch = ort.InferenceSession(D + '/codec_head.onnx', providers=['CPUExecutionProvider']) if os.path.exists(D + '/codec_head.onnx') else None
so_fe = ort.InferenceSession(OUT + '/frame_emb.onnx', providers=['CPUExecutionProvider'])
codec_head_w = base.codec_head if hasattr(base, 'codec_head') else None
talker = gw.model.talker

hidden = np.fromfile(D + '/vd_m5b/prefill_hidden_last.raw', dtype=np.float32)
kvk = np.fromfile(G2 + '/init_kv_k.raw', dtype=np.float32).reshape(28,1,8,MAXKV,128)
kvv = np.fromfile(G2 + '/init_kv_v.raw', dtype=np.float32).reshape(28,1,8,MAXKV,128)
cos_t = np.fromfile(G2 + '/cos300.raw', dtype=np.float32).reshape(300,1,1,128)
sin_t = np.fromfile(G2 + '/sin300.raw', dtype=np.float32).reshape(300,1,1,128)
rng = LCG()
recs = []
with torch.no_grad():
    hv = torch.from_numpy(hidden.copy())
    for n in range(20):
        cp = 62 + n
        # codec_head
        lg = talker.codec_head(hv.view(1,1,2048))[0,0].numpy().astype(np.float32)
        u0 = rng.u()
        code0, lo0, hi0 = sample_topk(lg, 0.9, 50, True, u0)
        if code0 == 2150:
            recs.append((n, cp, code0, [code0], u0)); break
        # codepred
        c0e = talker.get_input_embeddings()(torch.tensor([[code0]]))            # (1,1,2048)
        past = torch.cat([hv.view(1,1,2048), c0e], dim=1)                        # (1,2,2048)
        cp_out = talker.code_predictor.generate(inputs_embeds=past, max_new_tokens=15,
            do_sample=True, top_k=50, top_p=1.0, temperature=0.9,
            eos_token_id=talker.config.code_predictor_config.eos_token_id if hasattr(talker.config,'code_predictor_config') else None,
            pad_token_id=0, return_dict_in_generate=True, output_hidden_states=True)
        seq = cp_out.sequences[0, past.shape[1]:]                                # (15,)
        codes15 = [int(x) for x in seq.tolist()]
        us = [u0]
        for g in range(15): us.append(rng.u())   # consume same count as device
        frame = [code0] + codes15
        emb = so_fe.run(None, {'codes': np.array([frame], dtype=np.int64)})[0].astype(np.float32)
        # GraphB v6 forward
        h = torch.from_numpy(emb)
        rc = torch.from_numpy(cos_t[n]); rs = torch.from_numpy(sin_t[n])
        am = np.full((1,1,1,MAXKV), -1e4, np.float32); am[0,0,0,:cp+1] = 0.0
        sm = np.zeros((1,1,MAXKV,1), np.float32); sm[0,0,cp,0] = 1.0
        attn_mask = torch.from_numpy(am); slot_mask = torch.from_numpy(sm)
        one_minus = 1.0 - slot_mask
        cks = []; cvs = []
        kk = torch.from_numpy(kvk.copy()); vv = torch.from_numpy(kvv.copy())
        for i in range(28):
            layer = base.layers[i]
            residual = h
            hn = layer.input_layernorm(h)
            at = layer.self_attn
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
        # host KV write
        ck = torch.cat(cks, dim=0).numpy(); cvt = torch.cat(cvs, dim=0).numpy()
        for l in range(28):
            for hh in range(8):
                kvk[l,0,hh,cp,:] = ck[l,0,hh,:]
                kvv[l,0,hh,cp,:] = cvt[l,0,hh,:]
        hv = h.view(2048)
        recs.append((n, cp, code0, frame, u0, lo0, hi0))
        print('PC step %02d cp=%3d code0=%4d codes=%s' % (n, cp, code0, frame[1:4]))
with open(OUT + '/pc_records.txt', 'w') as f:
    for r in recs:
        if len(r) >= 4:
            f.write('step %d cp %d code0 %d %s\n' % (r[0], r[1], r[2], ' '.join(str(x) for x in r[3])))
print('PC_REF_DONE steps=%d' % len(recs))
