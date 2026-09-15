# P5.5: official torch vs MNN greedy, same start, first N steps - find first divergence
import os, json, numpy as np, torch, MNN, time, traceback
os.environ['HF_HUB_OFFLINE'] = '1'
D = r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/device-smoke'
B = r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/device-smoke/g4b_boundary'
P51 = r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/device-smoke/p51_assets'
MAX_KV = 768
N = 30
LOG = open(D + '/p55_div.log', 'w')
def log(s):
    LOG.write(s + '\n'); LOG.flush()
cos_fn = lambda a,b: float(np.dot(a.ravel(),b.ravel())/(np.linalg.norm(a)*np.linalg.norm(b)+1e-9))
from qwen_tts import Qwen3TTSModel
gw = Qwen3TTSModel.from_pretrained(r'E:/AndroidStudioProjects/qwen3tts-mnn/models/qwen3-tts/Qwen3-TTS-12Hz-1.7B-VoiceDesign', dtype=torch.float32, device_map='cpu')
talker = gw.model.talker
base = talker.model
cp_emb = talker.code_predictor.get_input_embeddings()
codec_emb = talker.get_input_embeddings()
proj = talker.code_predictor.small_to_mtp_projection
log('LOADED')
# --- official torch path (greedy, from G4-B0 boundary) ---
from transformers.cache_utils import DynamicCache
kv_t = DynamicCache()
kvk62 = torch.from_numpy(np.load(B + '/kv_k_decode.npy')).float()
kvv62 = torch.from_numpy(np.load(B + '/kv_v_decode.npy')).float()
for i in range(28):
    kv_t.update(kvk62[i], kvv62[i], i)
z0 = np.load(B + '/step_decode_inputs.npz')
emb_t = torch.from_numpy(z0['emb']).float()
cp0 = int(z0['cp'][0])
trailing_t = torch.from_numpy(np.load(P51 + '/trailing_text_hidden.npy')).float()
pad_t = torch.from_numpy(np.load(P51 + '/tts_pad_embed.npy')).float()
torch_codes = []; torch_hiddens = []
for step in range(N):
    cp_val = cp0 + step
    pos = torch.tensor([[[float(cp_val)]],[[float(cp_val)]],[[float(cp_val)]]])
    with torch.inference_mode():
        out = base(inputs_embeds=emb_t, position_ids=pos, past_key_values=kv_t, cache_position=torch.tensor([cp_val]), use_cache=True, output_hidden_states=True)
    h_t = out.last_hidden_state
    lg0 = talker.codec_head(h_t)
    code0 = int(torch.argmax(lg0[0,0]))
    with torch.no_grad():
        pj = proj(h_t).clone()
    codes15 = []
    hid_t2 = pj
    for g in range(15):
        with torch.inference_mode():
            o = talker.code_predictor.model(inputs_embeds=hid_t2, use_cache=False)
            lgg = talker.code_predictor.lm_head[g](o.last_hidden_state)
        codes15.append(int(torch.argmax(lgg[0,0])))
    torch_codes.append([code0] + codes15)
    torch_hiddens.append(h_t.detach().float().numpy())
    # next emb (official semantics)
    c0e = codec_emb(torch.tensor([[code0]])).squeeze(0)
    c15e = torch.stack([cp_emb[g](torch.tensor([[codes15[g]]])).squeeze(0) for g in range(15)], dim=0)
    src = trailing_t[0] if step == 0 else pad_t[0]
    emb_t = (c0e + c15e.sum(0) + src).unsqueeze(0)
log('TORCH done ' + str(len(torch_codes)) + ' steps')
# --- MNN greedy path (same start) ---
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
ch = MNN.Interpreter(D + '/codec_head_fp16.mnn')
chs = ch.createSession()
def ch_run(hidden_np):
    t = ch.getSessionInput(chs, 'hidden_states')
    ht = MNN.Tensor((1,1,2048), MNN.Halide_Type_Float, hidden_np.ravel(), MNN.Tensor_DimensionType_Caffe)
    t.copyFromHostTensor(ht)
    ch.runSession(chs)
    ot = ch.getSessionOutput(chs, 'logits')
    osh = [d if d > 0 else 3072 for d in ot.getShape()]
    ho = MNN.Tensor(osh, MNN.Halide_Type_Float, MNN.Tensor_DimensionType_Caffe)
    ot.copyToHostTensor(ho)
    return np.asarray(ho.getNumpyData()).reshape(osh).astype(np.float32)
cps = []
for gg in range(15):
    ci = MNN.Interpreter(D + '/codepred_g' + str(gg) + '_fp16.mnn')
    css = ci.createSession()
    cps.append((ci, css))
def cp_run(cp_in, gg):
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
kvk_buf = np.zeros((28,1,8,MAX_KV,128), dtype=np.float32)
kvv_buf = np.zeros((28,1,8,MAX_KV,128), dtype=np.float32)
kvk_buf[:,:,:,:62,:] = np.load(B + '/kv_k_decode.npy').astype(np.float32)[:,:,:,:62,:]
kvv_buf[:,:,:,:62,:] = np.load(B + '/kv_v_decode.npy').astype(np.float32)[:,:,:,:62,:]
emb_m = z0['emb'].astype(np.float32)
first_div = None
for step in range(N):
    cp_val = cp0 + step
    pos = np.array([[[float(cp_val)]],[[float(cp_val)]],[[float(cp_val)]]], dtype=np.float32)
    g_feed('inputs_embeds', emb_m, MNN.Halide_Type_Float)
    g_feed('position_ids', pos, MNN.Halide_Type_Float)
    g_feed('cache_position', np.array([cp_val], dtype=np.int32), MNN.Halide_Type_Int)
    g_feed('kv_k', kvk_buf, MNN.Halide_Type_Float)
    g_feed('kv_v', kvv_buf, MNN.Halide_Type_Float)
    g.runSession(gs)
    h_m = g_pull('hidden_states')
    kvk_buf = g_pull('kv_k_out'); kvv_buf = g_pull('kv_v_out')
    lg0 = ch_run(h_m)
    lg0m = lg0[0,0].copy()
    lg0m[2048:3072] = -1e30
    code0_m = int(np.argmax(lg0m))
    with torch.no_grad():
        pj_m = proj(torch.from_numpy(h_m)).clone().numpy().astype(np.float32)
    codes15_m = [int(np.argmax(cp_run(pj_m, gg)[0,0])) for gg in range(15)]
    # compare
    h_t = torch_hiddens[step]
    cos_h = cos_fn(h_t, h_m)
    t_frame = torch_codes[step]
    m_frame = [code0_m] + codes15_m
    same = t_frame == m_frame
    log('step ' + str(step) + ' cos_h ' + str(round(cos_h,6)) + ' codes_match ' + str(same) + ' t0 ' + str(t_frame[0]) + ' m0 ' + str(m_frame[0]))
    if not same and first_div is None:
        first_div = step
        log('>>> FIRST DIVERGENCE at step ' + str(step) + ' torch ' + str(t_frame[:4]) + ' mnn ' + str(m_frame[:4]))
    # next emb (mnn, official semantics)
    with torch.no_grad():
        c0e_m = codec_emb(torch.tensor([[code0_m]])).squeeze(0)
        c15e_m = torch.stack([cp_emb[g](torch.tensor([[codes15_m[g]]])).squeeze(0) for g in range(15)], dim=0)
        src_m = trailing_t[0] if step == 0 else pad_t[0]
        emb_m = (c0e_m + c15e_m.sum(0) + src_m).unsqueeze(0).detach().numpy().astype(np.float32)
log('FIRST_DIV ' + str(first_div))
print('P55_DIV_DONE', first_div, flush=True)