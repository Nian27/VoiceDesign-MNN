# P4 codes check: MNN deploy hidden -> codepred 15 codes vs authority (exact)
import os, json, numpy as np, torch, MNN
os.environ['HF_HUB_OFFLINE'] = '1'
D = r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/device-smoke'
T = r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/device-smoke/p1_traj'
MAX_KV = 768
NUM_STEPS = 20
LOG = open(D + '/p4_codes.log', 'w')
def log(s):
    LOG.write(s + '\n'); LOG.flush()
from qwen_tts import Qwen3TTSModel
gw = Qwen3TTSModel.from_pretrained(r'E:/AndroidStudioProjects/qwen3tts-mnn/models/qwen3-tts/Qwen3-TTS-12Hz-1.7B-VoiceDesign', dtype=torch.float32, device_map='cpu')
talker = gw.model.talker
cpm = talker.code_predictor.model
lm = talker.code_predictor.lm_head
proj = talker.code_predictor.small_to_mtp_projection
log('LOADED')
interp = MNN.Interpreter(D + '/graphb_deploy_hand_fp16.mnn')
interp.setExternalFile(D + '/graphb_deploy_hand_fp16.mnn.weight')
s2 = interp.createSession()
def feed(name, arr, dt):
    t = interp.getSessionInput(s2, name)
    ht = MNN.Tensor(tuple(arr.shape), dt, arr.ravel(), MNN.Tensor_DimensionType_Caffe)
    t.copyFromHostTensor(ht)
def pull_output(name):
    t = interp.getSessionOutput(s2, name)
    osh = [d if d > 0 else 1 for d in t.getShape()]
    ho = MNN.Tensor(osh, MNN.Halide_Type_Float, MNN.Tensor_DimensionType_Caffe)
    t.copyToHostTensor(ho)
    return np.asarray(ho.getNumpyData()).reshape(osh).astype(np.float32)
kvk_buf = np.zeros((28,1,8,MAX_KV,128), dtype=np.float32)
kvv_buf = np.zeros((28,1,8,MAX_KV,128), dtype=np.float32)
all_exact = True
for step in range(NUM_STEPS):
    z = np.load(T + '/step' + str(step) + '.npz')
    emb_in = z['emb_in'].astype(np.float32)
    pos = z['pos'].astype(np.float32)
    cp_val = int(z['cp'][0])
    kvk_cur = z['kvk'].astype(np.float32)
    kvv_cur = z['kvv'].astype(np.float32)
    codes_ref = z['codes'].tolist()
    n = kvk_cur.shape[3]
    kvk_buf[:,:,:,:n-1,:] = kvk_cur[:,:,:,:n-1,:]
    kvv_buf[:,:,:,:n-1,:] = kvv_cur[:,:,:,:n-1,:]
    feed('inputs_embeds', emb_in, MNN.Halide_Type_Float)
    feed('position_ids', pos, MNN.Halide_Type_Float)
    feed('cache_position', np.array([cp_val], dtype=np.int32), MNN.Halide_Type_Int)
    feed('kv_k', kvk_buf, MNN.Halide_Type_Float)
    feed('kv_v', kvv_buf, MNN.Halide_Type_Float)
    interp.runSession(s2)
    h = pull_output('hidden_states')
    kvk_buf = pull_output('kv_k_out'); kvv_buf = pull_output('kv_v_out')
    # codepred 15 via torch (proj from MNN hidden)
    with torch.no_grad():
        pj = proj(torch.from_numpy(h)).clone()
    codes_mnn = []
    hid_t = pj
    for g in range(15):
        with torch.inference_mode():
            o = cpm(inputs_embeds=hid_t, use_cache=False)
            lg = lm[g](o.last_hidden_state)
        codes_mnn.append(int(torch.argmax(lg[0,0])))
    exact = codes_ref == codes_mnn
    all_exact = all_exact and exact
    log('step ' + str(step) + ' exact15 ' + str(exact) + ' ref ' + str(codes_ref[:5]) + ' mnn ' + str(codes_mnn[:5]))
log('P4_CODES_ALL_EXACT ' + str(all_exact))
print('P4_CODES_DONE', all_exact, flush=True)