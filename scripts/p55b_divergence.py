# P5.5b: decisive first-divergence on frame 0 — pre-sampling logits, no RNG.
# Official codepred (torch, autoregressive cat(past_hidden, code0_emb))
#   vs
# Deployed codepred (MNN, parallel 15 heads fed proj(hidden_out))
# Also verify GraphB MNN hidden and codec_head MNN logits vs official.
import os, json, numpy as np, torch, MNN, traceback
os.environ['HF_HUB_OFFLINE'] = '1'
D = r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/device-smoke'
B = r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/device-smoke/g4b_boundary'
OFF = D + '/p55_official'
LOG = open(D + '/p55b_divergence.log', 'w')
def log(s):
    LOG.write(s + '\n'); LOG.flush()
def cos_fn(a, b):
    a = np.asarray(a, dtype=np.float32).ravel(); b = np.asarray(b, dtype=np.float32).ravel()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
log('loading...')
from qwen_tts import Qwen3TTSModel
gw = Qwen3TTSModel.from_pretrained(r'E:/AndroidStudioProjects/qwen3tts-mnn/models/qwen3-tts/Qwen3-TTS-12Hz-1.7B-VoiceDesign', dtype=torch.float32, device_map='cpu')
talker = gw.model.talker
cpm = talker.code_predictor.model
lm = talker.code_predictor.lm_head
cp_emb = talker.code_predictor.get_input_embeddings()
codec_emb = talker.get_input_embeddings()
proj = talker.code_predictor.small_to_mtp_projection
log('LOADED')

traj = np.load(OFF + '/official_traj.npz')
prefill_hidden = traj['prefill_hidden']          # (1,1,2048)
hidden_out = traj['hidden_out']                   # (111,1,1,2048)
ph_in = traj['past_hidden_in']                    # (111,1,1,2048)
cids = traj['codec_ids']                          # (111,1,16)
lg_off = traj['logits']                           # (111,1,1,3072)
code0_in = traj['code0_in']                       # (111,)
log('traj loaded: hidden_out ' + str(hidden_out.shape) + ' cids ' + str(cids.shape))

# --- GraphB MNN ---
g = MNN.Interpreter(D + '/graphb_deploy_hand_fp16.mnn')
g.setExternalFile(D + '/graphb_deploy_hand_fp16.mnn.weight')
gs = g.createSession()
def g_feed(name, arr, dt):
    t = g.getSessionInput(gs, name)
    ht = MNN.Tensor(tuple(arr.shape), dt, arr.ravel(), MNN.Tensor_DimensionType_Caffe)
    t.copyFromHostTensor(ht)
def g_pull(name):
    t = g.getSessionOutput(gs, name)
    osh = [d if d > 0 else 1 for d in t.getShape()]
    ho = MNN.Tensor(osh, MNN.Halide_Type_Float, MNN.Tensor_DimensionType_Caffe)
    t.copyToHostTensor(ho)
    return np.asarray(ho.getNumpyData()).reshape(osh).astype(np.float32)

# --- codec_head MNN ---
ch = MNN.Interpreter(D + '/codec_head_fp16.mnn')
chs = ch.createSession()
def codec_head(h):
    t = ch.getSessionInput(chs, 'hidden_states')
    ht = MNN.Tensor((1,1,2048), MNN.Halide_Type_Float, h.ravel(), MNN.Tensor_DimensionType_Caffe)
    t.copyFromHostTensor(ht)
    ch.runSession(chs)
    ot = ch.getSessionOutput(chs, 'logits')
    osh = [d if d > 0 else 3072 for d in ot.getShape()]
    ho = MNN.Tensor(osh, MNN.Halide_Type_Float, MNN.Tensor_DimensionType_Caffe)
    ot.copyToHostTensor(ho)
    return np.asarray(ho.getNumpyData()).reshape(osh).astype(np.float32)

# --- codepred MNN 15 (deployed) ---
cps = []
for gg in range(15):
    ci = MNN.Interpreter(D + '/codepred_g' + str(gg) + '_fp16.mnn')
    css = ci.createSession()
    cps.append((ci, css))
def cp_run_mnn(cp_in, gg):
    ci, css = cps[gg]
    t = ci.getSessionInput(css, 'inputs_embeds')
    ht = MNN.Tensor((1,1,1024), MNN.Halide_Type_Float, cp_in.ravel(), MNN.Tensor_DimensionType_Caffe)
    t.copyFromHostTensor(ht)
    ci.runSession(css)
    ot = ci.getSessionOutput(css, 'logits')
    osh = [d if d > 0 else 2048 for d in ot.getShape()]
    ho = MNN.Tensor(osh, MNN.Halide_Type_Float, MNN.Tensor_DimensionType_Caffe)
    ot.copyToHostTensor(ho)
    return np.asarray(ho.getNumpyData()).reshape(osh).astype(np.float32)

# ---- frame 0 start ----
kvk62 = np.load(B + '/kv_k_decode.npy').astype(np.float32)
kvv62 = np.load(B + '/kv_v_decode.npy').astype(np.float32)
z0 = np.load(B + '/step_decode_inputs.npz')
emb0 = z0['emb'].astype(np.float32)
cp0 = int(z0['cp'][0])
MAX_KV = 768
kvk = np.zeros((28,1,8,MAX_KV,128), dtype=np.float32)
kvv = np.zeros((28,1,8,MAX_KV,128), dtype=np.float32)
kvk[:,:,:,:62,:] = kvk62[:,:,:,:62,:]
kvv[:,:,:,:62,:] = kvv62[:,:,:,:62,:]

# 1) GraphB MNN hidden vs official hidden_out[0]
pos = np.array([[[float(cp0)]],[[float(cp0)]],[[float(cp0)]]], dtype=np.float32)
g_feed('inputs_embeds', emb0, MNN.Halide_Type_Float)
g_feed('position_ids', pos, MNN.Halide_Type_Float)
g_feed('cache_position', np.array([cp0], dtype=np.int32), MNN.Halide_Type_Int)
g_feed('kv_k', kvk, MNN.Halide_Type_Float)
g_feed('kv_v', kvv, MNN.Halide_Type_Float)
g.runSession(gs)
h_mnn = g_pull('hidden_states')
cos_h = cos_fn(h_mnn, hidden_out[0])
log('GraphB hidden: cos(MNN, official) = %.7f  (MNN maxabs %.3f official maxabs %.3f)' % (
    cos_h, float(np.abs(h_mnn).max()), float(np.abs(hidden_out[0]).max())))

# 2) codec_head MNN logits vs official lg[0]
lg_mnn = codec_head(h_mnn)
cos_lg = cos_fn(lg_mnn, lg_off[0])
a0_mnn = int(np.argmax(lg_mnn[0,0])); a0_off = int(np.argmax(lg_off[0,0]))
log('codec_head logits: cos(MNN, official) = %.7f  argmax %d vs %d (official code0_in=%d)' % (
    cos_lg, a0_mnn, a0_off, int(code0_in[0])))

# 3) codepred g=0: official AR vs deployed parallel (pre-sampling logits)
#    official: cat(prefill_hidden, codec_emb(code0)) -> proj -> cpm -> lm[0] (last token)
#    deployed: proj(hidden_out) -> cpm -> lm[0]
code0 = int(code0_in[0])
ph = torch.from_numpy(ph_in[0])                    # official past_hidden for frame0
c0e = codec_emb(torch.tensor([[code0]]))
seq_off = torch.cat([ph, c0e], dim=1)              # (1,2,2048)
with torch.inference_mode():
    seq1024 = proj(seq_off)                        # (1,2,1024)
    out_off = cpm(inputs_embeds=seq1024, use_cache=False)
    lg0_off_ar = lm[0](out_off.last_hidden_state[:, -1:])  # (1,1,2048) last token
lg0_off_ar_np = lg0_off_ar.detach().float().numpy().astype(np.float32)

with torch.no_grad():
    pj_dep = proj(torch.from_numpy(hidden_out[0])).clone().numpy().astype(np.float32)
lg0_dep = cp_run_mnn(pj_dep, 0)

cos_g0 = cos_fn(lg0_off_ar_np, lg0_dep)
ag0_off = int(torch.argmax(lg0_off_ar[0,0])); ag0_dep = int(np.argmax(lg0_dep[0,0]))
log('codepred g=0 logits: cos(AR-official, deployed) = %.7f  argmax %d vs %d  (official cids g0=%d)' % (
    cos_g0, ag0_off, ag0_dep, int(cids[0,0,0])))

# 4) also compare official AR vs official trajectory codes (validate my AR understanding)
#    official cids[0] = [code0, c1..c15]; AR should reproduce c1..c15 if greedy matches
seq_ar = seq_off.clone()
codes_ar = []
for gg in range(15):
    with torch.inference_mode():
        s1024 = proj(seq_ar)
        o = cpm(inputs_embeds=s1024, use_cache=False)
        lgg = lm[gg](o.last_hidden_state[:, -1:])
    c = int(torch.argmax(lgg[0,0]))
    codes_ar.append(c)
    seq_ar = torch.cat([seq_ar, cp_emb[gg](torch.tensor([[c]]))], dim=1)
off_codes = [int(cids[0,0,gg+1]) for gg in range(15)]
log('AR codes (greedy): ' + str(codes_ar))
log('official codes : ' + str(off_codes))
log('AR vs official exact: ' + str(codes_ar == off_codes))

log('')
log('FIRST_DIVERGENCE: step=0 tensor=codepred_g0_logits cos=%.7f' % cos_g0)
log('  -> deployed feeds proj(hidden_out) (current-frame), official feeds cat(past_hidden, code0_emb)')
log('  -> plus official is autoregressive (15 sequential steps), deployed is 15 independent heads')
print('P55B_DONE cos_h=%.7f cos_lg=%.7f cos_g0=%.7f' % (cos_h, cos_lg, cos_g0), flush=True)
