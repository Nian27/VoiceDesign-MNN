# CP-R2 (clean): host-driven MNN (prefill+step) vs official Torch, same RNG, per-stage
import os, math, numpy as np, torch, MNN
os.environ['HF_HUB_OFFLINE'] = '1'
torch.set_num_threads(1)
D = r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/device-smoke'
CR = D + '/vd_cpr'
CAP, HD, NL, NKV, NH = 16, 128, 5, 8, 16
SCALE = 1.0 / math.sqrt(HD)
MASK64 = (1 << 64) - 1
class LCG:
    def __init__(self, s=123456789): self.s = s & MASK64
    def u(self):
        self.s = (self.s * 6364136223846793005 + 1442695040888963407) & MASK64
        return ((self.s >> 11) & ((1 << 53) - 1)) / float(1 << 53)
def sample(lg, u, temp=0.9, topk=50, suppress=False):
    idx = np.arange(lg.size)
    if suppress:
        keep = ~((idx >= 2048) & (idx != 2150)); idx = idx[keep]; vals = lg[keep]/temp
    else:
        vals = lg/temp
    order = np.argsort(-vals, kind='stable')
    idx = idx[order][:topk]; vals = vals[order][:topk]
    mx = vals.max(); p = np.exp(vals-mx); p /= p.sum(); acc = np.cumsum(p)
    j = min(int(np.searchsorted(acc, u, side='left')), p.size-1)
    return int(idx[j])
def cos(a,b):
    a=np.asarray(a,np.float64).ravel(); b=np.asarray(b,np.float64).ravel()
    return float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12))

tew = np.fromfile(CR + '/talker_codec_emb_weight.f32', dtype=np.float32).reshape(3072,2048)
lw  = np.fromfile(CR + '/lm_head_weight.f32', dtype=np.float32).reshape(15,2048,1024)
ew  = np.fromfile(CR + '/codec_emb_weight.f32', dtype=np.float32).reshape(15,2048,2048)

from qwen_tts import Qwen3TTSModel
gw = Qwen3TTSModel.from_pretrained(r'E:/AndroidStudioProjects/qwen3tts-mnn/models/qwen3-tts/Qwen3-TTS-12Hz-1.7B-VoiceDesign', dtype=torch.float32, device_map='cpu')
cp = gw.model.talker.code_predictor; core = cp.model; talker = gw.model.talker

# ---- MNN helpers ----
def mk(path):
    it = MNN.Interpreter(path); return it, it.createSession()
def feed(it, s, name, arr):
    t = it.getSessionInput(s, name)
    shp = [d if d > 0 else 1 for d in t.getShape()]
    flat = np.ascontiguousarray(arr, np.float32).ravel()
    ht = MNN.Tensor(shp, MNN.Halide_Type_Float, flat, MNN.Tensor_DimensionType_Caffe)
    t.copyFromHostTensor(ht)
def getout(it, s, nm):
    o = it.getSessionOutput(s, nm); sh = [d if d > 0 else 1 for d in o.getShape()]
    ho = MNN.Tensor(sh, MNN.Halide_Type_Float, MNN.Tensor_DimensionType_Caffe)
    o.copyToHostTensor(ho)
    return np.asarray(ho.getNumpyData()).reshape(sh).astype(np.float32)
pre_it, pre_s = mk(CR + '/codepred_prefill_fp16.mnn')
stp_it, stp_s = mk(CR + '/codepred_step_fp16.mnn')

# ---- rope ----
x0 = torch.zeros(1,1,1024); pos = torch.arange(CAP).unsqueeze(0)
with torch.no_grad():
    c, si = core.rotary_emb(x0, pos)
c = np.asarray(c, np.float32); si = np.asarray(si, np.float32)
assert c.shape == (1, CAP, HD), c.shape
print('rope ok', c.shape)

past_np = np.fromfile(D + '/vd_m5b/prefill_hidden_last.raw', dtype=np.float32).reshape(1,1,2048)
with torch.no_grad():
    lg0 = talker.codec_head(torch.from_numpy(past_np))[0,0].numpy().astype(np.float32)
rng = LCG(); code0 = sample(lg0, rng.u(), suppress=True)
print('code0 =', code0)

# ---- TORCH official ----
codes_t = [code0]; lg_t = []
with torch.no_grad():
    xx = cp.small_to_mtp_projection(torch.cat([torch.from_numpy(past_np), torch.from_numpy(tew[code0].reshape(1,1,2048))], dim=1))
    o = core(inputs_embeds=xx, use_cache=True)
    cache = o.past_key_values; h = o.last_hidden_state
    lg_t.append(cp.lm_head[0](h[:, -1:])[0,0].numpy().astype(np.float32))
    for g in range(15):
        cd = sample(lg_t[g], rng.u())
        codes_t.append(cd)
        if g < 14:
            e = cp.model.codec_embedding[g](torch.tensor([[cd]]))
            o = core(inputs_embeds=cp.small_to_mtp_projection(e), past_key_values=cache,
                     use_cache=True, cache_position=torch.tensor([2+g]))
            cache = o.past_key_values; h = o.last_hidden_state
            lg_t.append(cp.lm_head[g+1](h)[0,0].numpy().astype(np.float32))
print('torch codes =', codes_t[1:])

# ---- MNN host-driven ----
rng2 = LCG(); code0b = sample(lg0, rng2.u(), suppress=True)
assert code0b == code0
sm0 = np.zeros((1,1,CAP,1), np.float32); sm0[0,0,0,0] = 1
sm1 = np.zeros((1,1,CAP,1), np.float32); sm1[0,0,1,0] = 1
am2 = np.full((1,1,2,CAP), -1e4, np.float32)
am2[0,0,0,0]=0; am2[0,0,1,0]=0; am2[0,0,1,1]=0   # causal: q0 only sees slot0, q1 sees slots 0..1
feed(pre_it, pre_s, 'past_hidden', past_np)
feed(pre_it, pre_s, 'code0_emb', tew[code0].reshape(1,1,2048))
feed(pre_it, pre_s, 'rope_cos', c[:, :2].reshape(1,1,2,HD))
feed(pre_it, pre_s, 'rope_sin', si[:, :2].reshape(1,1,2,HD))
feed(pre_it, pre_s, 'slot_mask0', sm0); feed(pre_it, pre_s, 'slot_mask1', sm1)
feed(pre_it, pre_s, 'attn_mask', am2)
pre_it.runSession(pre_s)
hl = getout(pre_it, pre_s, 'hidden_last').reshape(1024)
kvk = getout(pre_it, pre_s, 'kv_k_out'); kvv = getout(pre_it, pre_s, 'kv_v_out')
codes_m = []; lg_m = [hl @ lw[0].T]
for g in range(15):
    cd = sample(lg_m[g], rng2.u())
    codes_m.append(cd)
    if g < 14:
        e = ew[g][cd].reshape(1,1,2048)
        cp_pos = 2 + g
        sm = np.zeros((1,1,CAP,1), np.float32); sm[0,0,cp_pos,0] = 1
        am = np.full((1,1,1,CAP), -1e4, np.float32); am[0,0,0,:cp_pos+1] = 0
        feed(stp_it, stp_s, 'token_emb', e)
        feed(stp_it, stp_s, 'kv_k', kvk); feed(stp_it, stp_s, 'kv_v', kvv)
        feed(stp_it, stp_s, 'rope_cos', c[:, cp_pos].reshape(1,1,1,HD))
        feed(stp_it, stp_s, 'rope_sin', si[:, cp_pos].reshape(1,1,1,HD))
        feed(stp_it, stp_s, 'slot_mask', sm); feed(stp_it, stp_s, 'attn_mask', am)
        stp_it.runSession(stp_s)
        hl = getout(stp_it, stp_s, 'hidden_last').reshape(1024)
        kvk = getout(stp_it, stp_s, 'kv_k_out'); kvv = getout(stp_it, stp_s, 'kv_v_out')
        lg_m.append(hl @ lw[g+1].T)
print('mnn   codes =', codes_m)
print()
ok = sum(1 for i in range(15) if codes_t[1+i] == codes_m[i])
print('15-code exact = %d/15' % ok)
worst = 1.0
for i in range(15):
    cl = cos(lg_t[i], lg_m[i]); worst = min(worst, cl)
    oa = set(np.argsort(-lg_t[i])[:50].tolist()); ob = set(np.argsort(-lg_m[i])[:50].tolist())
    print('  stage %2d cos=%.7f top1=%d/%d top50ovl=%2d torch=%4d mnn=%4d %s' % (
        i, cl, int(np.argmax(lg_t[i])), int(np.argmax(lg_m[i])), len(oa&ob),
        codes_t[1+i], codes_m[i], 'OK' if codes_t[1+i]==codes_m[i] else 'DIFF'))
print('worst logits cos = %.7f' % worst)
print('CPR2_DONE')
