# P4 closure: GraphB deploy MNN 20-step -> codes -> decoder MNN -> PCM wav
import os, json, numpy as np, torch, MNN, time, traceback
os.environ['HF_HUB_OFFLINE'] = '1'
D = r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/device-smoke'
B = r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/device-smoke/g4b_boundary'
MAX_KV = 768
NUM_STEPS = 20
LOG = open(D + '/p4_closure.log', 'w')
def log(s):
    LOG.write(s + '\n'); LOG.flush()
from qwen_tts import Qwen3TTSModel
gw = Qwen3TTSModel.from_pretrained(r'E:/AndroidStudioProjects/qwen3tts-mnn/models/qwen3-tts/Qwen3-TTS-12Hz-1.7B-VoiceDesign', dtype=torch.float32, device_map='cpu')
talker = gw.model.talker
cpm = talker.code_predictor.model
lm = talker.code_predictor.lm_head
cp_emb = talker.code_predictor.get_input_embeddings()
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
kvk62 = np.load(B + '/kv_k_decode.npy').astype(np.float32)
kvv62 = np.load(B + '/kv_v_decode.npy').astype(np.float32)
kvk_buf[:,:,:,:62,:] = kvk62[:,:,:,:62,:]
kvv_buf[:,:,:,:62,:] = kvv62[:,:,:,:62,:]
z0 = np.load(B + '/step_decode_inputs.npz')
emb = z0['emb'].astype(np.float32)
cp0 = int(z0['cp'][0])
log('init kv62 cp ' + str(cp0))
traj_codes = []
for step in range(NUM_STEPS):
    cp_val = cp0 + step
    try:
        pos = np.array([[[float(cp_val)]],[[float(cp_val)]],[[float(cp_val)]]], dtype=np.float32)
        feed('inputs_embeds', emb, MNN.Halide_Type_Float)
        feed('position_ids', pos, MNN.Halide_Type_Float)
        feed('cache_position', np.array([cp_val], dtype=np.int32), MNN.Halide_Type_Int)
        feed('kv_k', kvk_buf, MNN.Halide_Type_Float)
        feed('kv_v', kvv_buf, MNN.Halide_Type_Float)
        t0 = time.time()
        interp.runSession(s2)
        t1 = time.time()
        h = pull_output('hidden_states')
        kvk_buf = pull_output('kv_k_out')
        kvv_buf = pull_output('kv_v_out')
        with torch.no_grad():
            pj = proj(torch.from_numpy(h)).clone()
        codes15 = []
        hid_t = pj
        for g in range(15):
            with torch.inference_mode():
                o = cpm(inputs_embeds=hid_t, use_cache=False)
                lg = lm[g](o.last_hidden_state)
            codes15.append(int(torch.argmax(lg[0,0])))
        traj_codes.append({'step': step, 'cp': cp_val, 'codes15': codes15})
        log('step ' + str(step) + ' cp ' + str(cp_val) + ' gb_ms ' + str(round((t1-t0)*1000,1)) + ' codes ' + str(codes15[:4]) + '...')
        codec_h = [torch.from_numpy(h)] + [cp_emb[g](torch.tensor([[codes15[g]]])) for g in range(15)]
        emb = torch.cat(codec_h, dim=1).sum(1, keepdim=True).detach().float().numpy().astype(np.float32)
    except Exception as ex:
        log('STEP ERR: ' + str(ex)[:400])
        traceback.print_exc(file=LOG)
        raise
with open(D + '/p4_closure_codes.json', 'w') as f:
    f.write(json.dumps({'steps': traj_codes, 'num_steps': NUM_STEPS}, indent=2))
log('P4_CLOSURE_CODES_SAVED ' + str(len(traj_codes)) + ' steps')
print('P4_CLOSURE_DONE', len(traj_codes), flush=True)