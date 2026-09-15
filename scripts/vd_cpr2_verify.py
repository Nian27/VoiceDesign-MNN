# CP-R2: new host-driven MNN (prefill+step) vs official Torch, same RNG, per-stage exact
import os, math, numpy as np, torch, MNN
os.environ['HF_HUB_OFFLINE'] = '1'
D = r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/device-smoke'
CR = D + '/vd_cpr'
CAP, HD, NL, NKV = 16, 128, 5, 8
MASK64 = (1 << 64) - 1
class LCG:
    def __init__(self, s=123456789): self.s = s & MASK64
    def u(self):
        self.s = (self.s * 6364136223846793005 + 1442695040888963407) & MASK64
        return ((self.s >> 11) & ((1 << 53) - 1)) / float(1 << 53)
def sample_topk(lg, temp, topk, suppress, u):
    idx = np.arange(lg.size)
    if suppress:
        keep = ~((idx >= 2048) & (idx != 2150)); idx = idx[keep]; vals = lg[keep]/temp
    else:
        vals = lg/temp
    order = np.argsort(-vals, kind='stable')
    idx = idx[order][:topk]; vals = vals[order][:topk]
    mx = vals.max(); p = np.exp(vals-mx); p /= p.sum(); acc = np.cumsum(p)
    j = int(np.searchsorted(acc, u, side='left')); j = min(j, p.size-1)
    return int(idx[j]), float(acc[j]-p[j]), float(acc[j])
def cos(a,b):
    a=np.asarray(a,np.float64).ravel(); b=np.asarray(b,np.float64).ravel()
    return float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12))

torch.set_num_threads(1)
try:
    torch.use_deterministic_algorithms(True, warn_only=True)
except Exception:
    pass
from qwen_tts import Qwen3TTSModel
gw = Qwen3TTSModel.from_pretrained(r'E:/AndroidStudioProjects/qwen3tts-mnn/models/qwen3-tts/Qwen3-TTS-12Hz-1.7B-VoiceDesign', dtype=torch.float32, device_map='cpu')
cp = gw.model.talker.code_predictor; core = cp.model; talker = gw.model.talker
lw = np.fromfile(CR + '/lm_head_weight.f32', dtype=np.float32).reshape(15,2048,1024)
ew = np.fromfile(CR + '/codec_emb_weight.f32', dtype=np.float32).reshape(15,2048,2048)

# ---- MNN sessions ----
def mk(path):
    it = MNN.Interpreter(path); s = it.createSession()
    return it, s
def run1(it, s, inName, data, outNames):
    t = it.getSessionInput(s, inName)
    dims = [d if d > 0 else 1 for d in t.getShape()]
    ht = MNN.Tensor(dims, MNN.Halide_Type_Float, np.ascontiguousarray(data, np.float32).ravel(), MNN.Tensor_DimensionType_Caffe)
    t.copyFromHostTensor(ht)
    it.runSession(s)
    outs = []
    for on in outNames:
        o = it.getSessionOutput(s, on)
        sh = [d if d > 0 else 1 for d in o.getShape()]
        ho = MNN.Tensor(sh, MNN.Halide_Type_Float, MNN.Tensor_DimensionType_Caffe)
        o.copyToHostTensor(ho)
        outs.append(np.asarray(ho.getNumpyData()).reshape(sh).astype(np.float32))
    return outs
pre_it, pre_s = mk(CR + '/codepred_prefill_fp16.mnn')
stp_it, stp_s = mk(CR + '/codepred_step_fp16.mnn')
ce_it, ce_s = mk(D + '/vd_m5b/codec_emb_fp16.mnn')

# ---- rope for positions 0..15 ----
x0 = torch.zeros(1,1,1024)
pos = torch.arange(CAP).unsqueeze(0)
with torch.no_grad():
    c, si = core.rotary_emb(x0, pos)
print('rope cos shape', tuple(c.shape))
c = c.numpy(); si = si.numpy()
if c.ndim == 3: c = c[:, None, :, :]; si = si[:, None, :, :]


# ---- test input: real prefill hidden ----
past = np.fromfile(D + '/vd_m5b/prefill_hidden_last.raw', dtype=np.float32).reshape(1,1,2048)
with torch.no_grad():
    lg0 = talker.codec_head(torch.from_numpy(past))[0,0].numpy().astype(np.float32)
rng = LCG()
u0 = rng.u()
code0, lo0, hi0 = sample_topk(lg0, 0.9, 50, True, u0)
print('code0 =', code0)
c0e = run1(ce_it, ce_s, 'codes', np.array([[code0]], np.int32), ['emb'])[0]
c0e = c0e.reshape(1,1,2048)
print('code0 emb shape', c0e.shape)

# ---- TORCH official ----
torch_codes = []; torch_logits = []
with torch.no_grad():
    xx = cp.small_to_mtp_projection(torch.cat([torch.from_numpy(past), torch.from_numpy(c0e)], dim=1))
    o = core(inputs_embeds=xx, use_cache=True)
    cache = o.past_key_values; h = o.last_hidden_state
    cur_logits = cp.lm_head[0](h[:, -1:])[0,0].numpy().astype(np.float32)
    torch_logits.append(cur_logits)
    torch_codes.append(code0)
    for g in range(15):
        torch_codes.append(None)
    torch_codes = [code0]
    for g in range(15):
        u = rng.u()
        cd, _, _ = sample_topk(torch_logits[g], 0.9, 50, False, u)
        torch_codes.append(cd)
        if g < 14:
            e = cp.model.codec_embedding[g](torch.tensor([[cd]]))
            xx = cp.small_to_mtp_projection(e)
            o = core(inputs_embeds=xx, past_key_values=cache, use_cache=True,
                     cache_position=torch.tensor([2+g]))
            cache = o.past_key_values; h = o.last_hidden_state
            torch_logits.append(cp.lm_head[g+1](h)[0,0].numpy().astype(np.float32))
print('torch codes =', torch_codes[1:])

# ---- MNN host-driven ----
rng2 = LCG()
u0b = rng2.u()
sm0 = np.zeros((1,1,CAP,1), np.float32); sm0[0,0,0,0]=1
sm1 = np.zeros((1,1,CAP,1), np.float32); sm1[0,0,1,0]=1
am2 = np.full((1,1,2,CAP), -1e4, np.float32)
am2[0,0,0,0]=0; am2[0,0,1,0]=0; am2[0,0,0,1]=0; am2[0,0,1,1]=0
# feed all prefill inputs one by one then run
preit = pre_it; pres = pre_s
for nm, dat in [('past_hidden', past), ('code0_emb', c0e),
                ('rope_cos', c[0,:2].reshape(1,1,2,HD)),
                ('rope_sin', si[0,:2].reshape(1,1,2,HD)),
                ('slot_mask0', sm0), ('slot_mask1', sm1), ('attn_mask', am2)]:
    t = preit.getSessionInput(pres, nm)
    dims = [d if d>0 else 1 for d in t.getShape()]
    arr = np.ascontiguousarray(dat.reshape(-1), np.float32)
    if arr.size != int(np.prod(dims)):
        print('  NOTE shape mismatch for', nm, 'expect', dims, 'got', dat.shape)
        dims = list(dat.shape)
    ht = MNN.Tensor(dims, MNN.Halide_Type_Float, dat.astype(np.float32).ravel(), MNN.Tensor_DimensionType_Caffe)
    t.copyFromHostTensor(ht)
preit.runSession(pres)
def getout(it, s, nm):
    o = it.getSessionOutput(s, nm); sh = [d if d>0 else 1 for d in o.getShape()]
    ho = MNN.Tensor(sh, MNN.Halide_Type_Float, MNN.Tensor_DimensionType_Caffe)
    o.copyToHostTensor(ho); return np.asarray(ho.getNumpyData()).reshape(sh).astype(np.float32)
hl = getout(preit, pres, 'hidden_last'); kvk = getout(preit, pres, 'kv_k_out'); kvv = getout(preit, pres, 'kv_v_out')
print('prefill hl', hl.shape, 'kv', kvk.shape)

mnn_logits = []; mnn_codes = []
lg = hl.reshape(1024) @ lw[0].T
mnn_logits.append(lg.astype(np.float32))
cd, _, _ = sample_topk(mnn_logits[0], 0.9, 50, False, u0b)
mnn_codes.append(cd)
for g in range(1, 15):
    tok = mnn_codes[-1]
    e = ew[g-1][tok].reshape(1,1,2048)
    cp_pos = 1 + g
    sm = np.zeros((1,1,CAP,1), np.float32); sm[0,0,cp_pos,0]=1
    am = np.full((1,1,1,CAP), -1e4, np.float32); am[0,0,0,:cp_pos+1]=0
    rc = c[0,cp_pos].reshape(1,1,1,HD)
    rs = si[0,cp_pos].reshape(1,1,1,HD)
    for nm, dat in [('token_emb', e), ('kv_k', kvk), ('kv_v', kvv), ('rope_cos', rc), ('rope_sin', rs),
                    ('slot_mask', sm), ('attn_mask', am)]:
        t = stp_it.getSessionInput(stp_s, nm)
        dims = [d if d>0 else 1 for d in t.getShape()]
        ht = MNN.Tensor(dims, MNN.Halide_Type_Float, np.ascontiguousarray(dat, np.float32).ravel(), MNN.Tensor_DimensionType_Caffe)
        t.copyFromHostTensor(ht)
    stp_it.runSession(stp_s)
    hl2 = getout(stp_it, stp_s, 'hidden_last'); kvk = getout(stp_it, stp_s, 'kv_k_out'); kvv = getout(stp_it, stp_s, 'kv_v_out')
    u = rng2.u()
    lg = hl2.reshape(1024) @ lw[g].T
    mnn_logits.append(lg.astype(np.float32))
    cd, _, _ = sample_topk(lg, 0.9, 50, False, u)
    mnn_codes.append(cd)
print('mnn   codes =', mnn_codes)
print()
match = sum(1 for i in range(15) if torch_codes[1+i] == mnn_codes[i])
print('15-code exact = %d/15' % match)
for i in range(15):
    cl = cos(torch_logits[i], mnn_logits[i])
    ta = int(np.argmax(torch_logits[i])); tb = int(np.argmax(mnn_logits[i]))
    oa=set(np.argsort(-torch_logits[i])[:50].tolist()); ob=set(np.argsort(-mnn_logits[i])[:50].tolist())
    print('  stage %2d logits_cos=%.7f top1=%d/%d top50ovl=%d torch=%d mnn=%d %s' % (
        i, cl, ta, tb, len(oa&ob), torch_codes[1+i], mnn_codes[i], 'OK' if torch_codes[1+i]==mnn_codes[i] else 'DIFF'))
print('CPR2_DONE')
