# P5 minimal: verify suppress_tokens sampling produces EOS-eligible code0
import os, json, numpy as np, torch, MNN, time, traceback
os.environ['HF_HUB_OFFLINE'] = '1'
D = r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/device-smoke'
B = r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/device-smoke/g4b_boundary'
MAX_KV = 768
LOG = open(D + '/p5_sample_test.log', 'w')
def log(s):
    LOG.write(s + '\n'); LOG.flush()
from qwen_tts import Qwen3TTSModel
gw = Qwen3TTSModel.from_pretrained(r'E:/AndroidStudioProjects/qwen3tts-mnn/models/qwen3-tts/Qwen3-TTS-12Hz-1.7B-VoiceDesign', dtype=torch.float32, device_map='cpu')
talker = gw.model.talker
cp_emb = talker.code_predictor.get_input_embeddings()
codec_emb = talker.get_input_embeddings()
proj = talker.code_predictor.small_to_mtp_projection
log('LOADED')
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
def sample_code(logits_np, seed):
    np.random.seed(seed)
    lg = logits_np.ravel().astype(np.float64) / 0.9
    lg[2048:3072] = float('-inf')
    lg[2150] = logits_np.ravel()[2150] / 0.9
    idx = np.argsort(lg)[::-1][:50]
    topv = lg[idx]
    p = np.exp(topv - topv.max())
    p = p / p.sum()
    pick = np.random.choice(idx, p=p)
    return int(pick)
kvk_buf = np.zeros((28,1,8,MAX_KV,128), dtype=np.float32)
kvv_buf = np.zeros((28,1,8,MAX_KV,128), dtype=np.float32)
kvk62 = np.load(B + '/kv_k_decode.npy').astype(np.float32)
kvv62 = np.load(B + '/kv_v_decode.npy').astype(np.float32)
kvk_buf[:,:,:,:62,:] = kvk62[:,:,:,:62,:]
kvv_buf[:,:,:,:62,:] = kvv62[:,:,:,:62,:]
z0 = np.load(B + '/step_decode_inputs.npz')
emb = z0['emb'].astype(np.float32)
cp0 = int(z0['cp'][0])
tts_pad_id = 2148
with torch.no_grad():
    tts_pad_embed = talker.text_projection(talker.get_text_embeddings()(torch.tensor([[tts_pad_id]])))
log('init cp ' + str(cp0))
# run 50 steps, check code0 distribution + EOS hits
eos_hits = 0
code0_hist = {}
try:
    for step in range(50):
        cp_val = cp0 + step
        pos = np.array([[[float(cp_val)]],[[float(cp_val)]],[[float(cp_val)]]], dtype=np.float32)
        g_feed('inputs_embeds', emb, MNN.Halide_Type_Float)
        g_feed('position_ids', pos, MNN.Halide_Type_Float)
        g_feed('cache_position', np.array([cp_val], dtype=np.int32), MNN.Halide_Type_Int)
        g_feed('kv_k', kvk_buf, MNN.Halide_Type_Float)
        g_feed('kv_v', kvv_buf, MNN.Halide_Type_Float)
        g.runSession(gs)
        h = g_pull('hidden_states')
        kvk_buf = g_pull('kv_k_out'); kvv_buf = g_pull('kv_v_out')
        lg0 = ch_run(h)
        code0 = sample_code(lg0[0,0], 42 + step)
        code0_hist[code0] = code0_hist.get(code0, 0) + 1
        if code0 == 2150:
            eos_hits += 1
            log('EOS hit at step ' + str(step))
        with torch.no_grad():
            pj = proj(torch.from_numpy(h)).clone().numpy().astype(np.float32)
        codes15 = [sample_code(g.getSessionOutput(gs, 'x') if False else np.zeros(2048), 0) for _ in range(15)] if False else [0]*15
        with torch.no_grad():
            c0e = codec_emb(torch.tensor([[code0]])).squeeze(0)
            emb = (c0e + tts_pad_embed.squeeze(0)).unsqueeze(0).detach().numpy().astype(np.float32)
        if step < 3 or step % 10 == 0:
            log('step ' + str(step) + ' code0 ' + str(code0))
except Exception as ex:
    log('ERR: ' + str(ex)[:300])
    traceback.print_exc(file=LOG)
log('EOS_HITS ' + str(eos_hits) + ' distinct_code0 ' + str(len(code0_hist)) + ' top_code0 ' + str(sorted(code0_hist.items(), key=lambda x:-x[1])[:5]))
print('P5_SAMPLE_TEST_DONE eos', eos_hits, flush=True)