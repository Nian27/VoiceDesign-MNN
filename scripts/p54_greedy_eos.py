# P5.4: greedy (argmax) self-driven - does EOS trigger deterministically?
import os, json, numpy as np, torch, MNN, time, traceback
os.environ['HF_HUB_OFFLINE'] = '1'
D = r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/device-smoke'
B = r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/device-smoke/g4b_boundary'
P51 = r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/device-smoke/p51_assets'
MAX_KV = 768
MAX_STEPS = 600
LOG = open(D + '/p54_greedy.log', 'w')
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
def greedy_code(logits_np, suppress=False):
    lg = logits_np.ravel().copy()
    if suppress and lg.size >= 3072:
        lg[2048:3072] = -1e30
    return int(np.argmax(lg))
kvk_buf = np.zeros((28,1,8,MAX_KV,128), dtype=np.float32)
kvv_buf = np.zeros((28,1,8,MAX_KV,128), dtype=np.float32)
kvk62 = np.load(B + '/kv_k_decode.npy').astype(np.float32)
kvv62 = np.load(B + '/kv_v_decode.npy').astype(np.float32)
kvk_buf[:,:,:,:62,:] = kvk62[:,:,:,:62,:]
kvv_buf[:,:,:,:62,:] = kvv62[:,:,:,:62,:]
z0 = np.load(B + '/step_decode_inputs.npz')
emb = z0['emb'].astype(np.float32)
cp0 = int(z0['cp'][0])
trailing = np.load(P51 + '/trailing_text_hidden.npy').astype(np.float32)
pad_emb = np.load(P51 + '/tts_pad_embed.npy').astype(np.float32)
log('init cp ' + str(cp0))
traj = []; frame_codes = []; stop = False; stop_reason = 'max_steps'
for step in range(MAX_STEPS):
    if stop: break
    cp_val = cp0 + step
    try:
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
        code0 = greedy_code(lg0[0,0], suppress=True)
        with torch.no_grad():
            pj = proj(torch.from_numpy(h)).clone().numpy().astype(np.float32)
        codes15 = [greedy_code(cp_run(pj, gg)[0,0]) for gg in range(15)]
        frame = [code0] + codes15
        frame_codes.append(frame)
        traj.append({'step': step, 'cp': cp_val, 'frame': frame})
        if step % 10 == 0 or step < 3:
            log('step ' + str(step) + ' cp ' + str(cp_val) + ' code0 ' + str(code0))
        if code0 == 2150:
            stop = True; stop_reason = 'eos'; log('EOS at step ' + str(step) + ' cp ' + str(cp_val))
        else:
            with torch.no_grad():
                c0e = codec_emb(torch.tensor([[code0]])).squeeze(0)
                c15e = torch.stack([cp_emb[gg](torch.tensor([[frame[gg+1]]])).squeeze(0) for gg in range(15)], dim=0)
                src_emb = trailing[0] if step == 0 else pad_emb[0]
                emb = (c0e + c15e.sum(0) + torch.from_numpy(src_emb)).unsqueeze(0).detach().numpy().astype(np.float32)
    except Exception as ex:
        log('STEP ERR ' + str(step) + ': ' + str(ex)[:300])
        traceback.print_exc(file=LOG)
        raise
log('TOTAL ' + str(len(frame_codes)) + ' stop ' + stop_reason)
np.save(D + '/p54_greedy_codes.npy', np.array(frame_codes, dtype=np.int32))
print('P54_GREEDY_DONE', len(frame_codes), stop_reason, flush=True)