# P5 first shot: GraphB MNN hidden -> codec_head (torch) vs official, verify logits/code0
import os, json, numpy as np, torch, MNN, time, traceback
os.environ['HF_HUB_OFFLINE'] = '1'
D = r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/device-smoke'
B = r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/device-smoke/g4b_boundary'
T = r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/device-smoke/p1_traj'
MAX_KV = 768
N = 5
LOG = open(D + '/p5_codec.log', 'w')
def log(s):
    LOG.write(s + '\n'); LOG.flush()
from qwen_tts import Qwen3TTSModel
gw = Qwen3TTSModel.from_pretrained(r'E:/AndroidStudioProjects/qwen3tts-mnn/models/qwen3-tts/Qwen3-TTS-12Hz-1.7B-VoiceDesign', dtype=torch.float32, device_map='cpu')
talker = gw.model.talker
base = talker.model
cpm = talker.code_predictor.model
lm = talker.code_predictor.lm_head
cp_emb = talker.code_predictor.get_input_embeddings()
proj = talker.code_predictor.small_to_mtp_projection
ch = talker.codec_head
log('LOADED codec_head ' + str(tuple(ch.weight.shape)))
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
log('init cp ' + str(cp0))
traj = []
for step in range(N):
    cp_val = cp0 + step
    try:
        pos = np.array([[[float(cp_val)]],[[float(cp_val)]],[[float(cp_val)]]], dtype=np.float32)
        feed('inputs_embeds', emb, MNN.Halide_Type_Float)
        feed('position_ids', pos, MNN.Halide_Type_Float)
        feed('cache_position', np.array([cp_val], dtype=np.int32), MNN.Halide_Type_Int)
        feed('kv_k', kvk_buf, MNN.Halide_Type_Float)
        feed('kv_v', kvv_buf, MNN.Halide_Type_Float)
        interp.runSession(s2)
        h = pull_output('hidden_states')
        kvk_buf = pull_output('kv_k_out'); kvv_buf = pull_output('kv_v_out')
        # codec_head logits from MNN hidden (torch linear)
        with torch.inference_mode():
            lg0 = ch(torch.from_numpy(h))
        lg0v = lg0[0,0]
        top1 = int(torch.argmax(lg0v))
        # sampling policy: temp 0.9, top_k 50 (official defaults)
        torch.manual_seed(42 + step)
        lg_s = lg0v / 0.9
        topk_v, topk_i = torch.topk(lg_s, 50)
        probs = torch.softmax(topk_v, dim=-1)
        sampled = topk_i[torch.multinomial(probs, 1).item()].item()
        log('step ' + str(step) + ' cp ' + str(cp_val) + ' top1 ' + str(top1) + ' sampled_code0 ' + str(sampled) + ' max_logit ' + str(round(float(lg0v.max()),3)))
        traj.append({'step': step, 'cp': cp_val, 'code0_top1': top1, 'code0_sampled': sampled})
        # 15 codepred codes from same hidden
        with torch.no_grad():
            pj = proj(torch.from_numpy(h)).clone()
        codes15 = []
        hid_t = pj
        for g in range(15):
            with torch.inference_mode():
                o = cpm(inputs_embeds=hid_t, use_cache=False)
                lgg = lm[g](o.last_hidden_state)
            codes15.append(int(torch.argmax(lgg[0,0])))
        traj[-1]['codes15'] = codes15
        # next emb: code0 emb + 15 code emb sum + tts_pad (use tts_pad_embed via torch)
        code0_emb = talker.get_input_embeddings()(torch.tensor([[sampled]]))
        codec_h = [code0_emb] + [cp_emb[g](torch.tensor([[codes15[g]]])) for g in range(15)]
        emb = torch.cat(codec_h, dim=1).sum(1, keepdim=True).detach().float().numpy().astype(np.float32)
        log('  codes15 ' + str(codes15[:4]) + '...')
    except Exception as ex:
        log('STEP ERR: ' + str(ex)[:400])
        traceback.print_exc(file=LOG)
        raise
with open(D + '/p5_codec_traj.json', 'w') as f:
    f.write(json.dumps({'steps': traj}, indent=2))
log('P5_CODEC_DONE ' + str(len(traj)) + ' steps')
print('P5_CODEC_DONE', len(traj), flush=True)