# Gate4 PC loop closure: PyTorch authoritative trajectory vs MNN per-step mirror
# Contract: GRAPHB_INPUT_ORDER is a runtime manifest, never index-guessed
import os, json, numpy as np, torch, MNN
os.environ['HF_HUB_OFFLINE'] = '1'
D = r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/device-smoke'
G = r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/m22/golden'
GRAPHB_INPUT_ORDER = ['cache_position', 'inputs_embeds', 'kv_k', 'kv_v', 'position_ids']
NUM_STEPS = 20
from qwen_tts import Qwen3TTSModel
gw = Qwen3TTSModel.from_pretrained(r'E:/AndroidStudioProjects/qwen3tts-mnn/models/qwen3-tts/Qwen3-TTS-12Hz-1.7B-VoiceDesign', dtype=torch.float32, device_map='cpu')
talker = gw.model.talker
cpm = talker.code_predictor.model
lm = talker.code_predictor.lm_head
cp_emb = talker.code_predictor.get_input_embeddings()
proj = talker.code_predictor.small_to_mtp_projection
print('LOADED', flush=True)
# GraphB MNN
interp = MNN.Interpreter(D + '/graphb_static_t35_fp16.mnn')
interp.setExternalFile(D + '/graphb_static_t35_fp16.mnn.weight')
s2 = interp.createSession()
interp.resizeSession(s2)
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
# codepred MNN 15 graphs (single-step contract: input (1,1,1024) -> logits (1,2048))
cp_interps = []
for g in range(15):
    ii = MNN.Interpreter(D + '/codepred_g' + str(g) + '_fp16.mnn')
    ss = ii.createSession()
    t0 = ii.getSessionInput(ss, 'inputs_embeds')
    ii.resizeTensor(t0, (1,1,1024))
    ii.resizeSession(ss)
    cp_interps.append((ii, ss))
def run_cp_mnn(cp_in_np, g):
    ii, ss = cp_interps[g]
    t = ii.getSessionInput(ss, 'inputs_embeds')
    ht = MNN.Tensor((1,1,1024), MNN.Halide_Type_Float, cp_in_np.ravel(), MNN.Tensor_DimensionType_Caffe)
    t.copyFromHostTensor(ht)
    ii.runSession(ss)
    ot = ii.getSessionOutput(ss, 'logits')
    osh = [d if d > 0 else 2048 for d in ot.getShape()]
    ho = MNN.Tensor(osh, MNN.Halide_Type_Float, MNN.Tensor_DimensionType_Caffe)
    ot.copyToHostTensor(ho)
    return np.asarray(ho.getNumpyData()).reshape(osh).astype(np.float32)
# start state from golden step0 (authoritative inputs)
gold = torch.load(G + '/g1_decode_golden.pt', map_location='cpu', weights_only=False)
st0 = gold['decode_steps'][0]
emb0 = st0['inputs_embeds'].float().numpy()          # (1,1,2048)
pos0 = st0['position_ids'].float().numpy()           # (3,1,1)
cp0 = int(st0['cache_position'].item())
print('start emb', emb0.shape, 'pos', pos0.shape, 'cp', cp0, flush=True)
from transformers.cache_utils import DynamicCache
kv = DynamicCache()
steps = []
cos_fn = lambda a,b: float(np.dot(a.ravel(),b.ravel())/(np.linalg.norm(a)*np.linalg.norm(b)+1e-9))
emb = emb0.copy()
pos = pos0.copy()
LOG = open('E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/device-smoke/loop_diag.log', 'w')
LOG.write('INIT emb ' + str(emb.shape) + '\n')
LOG.flush()
for step in range(NUM_STEPS):
    cp_val = cp0 + step
    cpn = np.array([cp_val], dtype=np.int32)
    pos = np.array([[[cp_val]],[[cp_val]],[[cp_val]]], dtype=np.float32)  # (3,1,1) mrope pos advances with cp
    LOG.write('step ' + str(step) + ' emb ' + str(emb.shape) + ' pos ' + str(pos.shape) + ' cp ' + str(cp_val) + '\n')
    LOG.flush()
    try:
        with torch.inference_mode():
            out = talker.model(inputs_embeds=torch.from_numpy(emb), position_ids=torch.from_numpy(pos), past_key_values=kv, cache_position=torch.tensor([cp_val]), use_cache=True)
    except Exception as ex:
        LOG.write('TORCH ERR at step ' + str(step) + ': ' + str(ex)[:300] + '\n')
        LOG.flush()
        raise
    hidden_pt = out.last_hidden_state.float().numpy()
    LOG.write('  torch out ok hidden ' + str(hidden_pt.shape) + '\n')
    LOG.flush()
    try:
        kk = torch.stack([kv.layers[i].keys for i in range(28)], dim=0).float().numpy().astype(np.float32)
        vv = torch.stack([kv.layers[i].values for i in range(28)], dim=0).float().numpy().astype(np.float32)
        LOG.write('  kv stack ok ' + str(kk.shape) + '\n')
    except Exception as ex:
        LOG.write('KV ERR: ' + str(ex)[:300] + ' layers=' + str(len(kv.layers)) + ' l0keys=' + str(kv.layers[0].keys.shape if kv.layers[0].keys is not None else 'None') + '\n')
        LOG.flush()
        raise
    feed('cache_position', cpn, MNN.Halide_Type_Int)
    feed('inputs_embeds', emb, MNN.Halide_Type_Float)
    feed('kv_k', kk, MNN.Halide_Type_Float)
    feed('kv_v', vv, MNN.Halide_Type_Float)
    feed('position_ids', pos, MNN.Halide_Type_Float)
    LOG.write('  feed ok\n')
    LOG.flush()
    interp.runSession(s2)
    LOG.write('  graphb run ok\n')
    LOG.flush()
    hidden_mnn = pull_output('hidden_states')
    LOG.write('  hidden_mnn ' + str(hidden_mnn.shape) + ' max ' + str(round(float(np.abs(hidden_mnn).max()),4)) + '\n')
    LOG.flush()
    cos_h = cos_fn(hidden_pt, hidden_mnn)
    finite_h = bool(np.isfinite(hidden_mnn).all())
    # codepred: authoritative torch vs MNN, exact 15/15
    with torch.no_grad():
        pj = proj(torch.from_numpy(hidden_pt)).clone()
    pj_np = pj.float().numpy()
    codes_pt = []; codes_mnn = []; top2_pt = []; top2_mnn = []
    hid_t = pj.clone()
    for g in range(15):
        try:
            with torch.inference_mode():
                o = cpm(inputs_embeds=hid_t, use_cache=False)
            lg = lm[g](o.last_hidden_state)
        lgv = lg[0,0]
        c_pt = int(torch.argmax(lgv))
        codes_pt.append(c_pt)
        top2_pt.append([int(x) for x in torch.argsort(lgv, descending=True)[:2].tolist()])
        lg_mnn = run_cp_mnn(pj_np, g)[0,0]
        except Exception as ex:
            LOG.write('CP ERR g' + str(g) + ' hid_t ' + str(hid_t.shape) + ': ' + str(ex)[:200] + '\n')
            LOG.flush()
            raise
        c_mnn = int(np.argmax(lg_mnn))
        codes_mnn.append(c_mnn)
        top2_mnn.append([int(x) for x in np.argsort(lg_mnn)[::-1][:2].tolist()])
        hid_t = cp_emb[g](torch.tensor([[c_pt]]))
    exact15 = codes_pt == codes_mnn
    print('step', step, 'cos_h', round(cos_h,6), 'finite', finite_h, 'exact15', exact15, flush=True)
    steps.append({
        'step': step, 'cache_position': cp_val, 'cos_hidden': round(cos_h,6), 'hidden_finite': finite_h,
        'codes_pt': codes_pt, 'codes_mnn': codes_mnn, 'exact15_15': exact15,
        'top2_pt': top2_pt, 'top2_mnn': top2_mnn,
        'kv_finite': bool(np.isfinite(kk).all() and np.isfinite(vv).all()),
        'kv_shape': list(kk.shape),
    })
    codec_h = [torch.from_numpy(hidden_pt)] + [cp_emb[g](torch.tensor([[codes_pt[g]]])) for g in range(15)]
    emb = torch.cat(codec_h, dim=1).sum(1, keepdim=True).float().numpy()
traj = {'steps': steps}
with open(D + '/loop_trajectory.json', 'w') as f:
    f.write(json.dumps(traj, indent=2))
print('LOOP_TRAJECTORY_SAVED', flush=True)
all_exact = all(s['exact15_15'] for s in steps)
all_fin = all(s['hidden_finite'] and s['kv_finite'] for s in steps)
print('G4_ALL_EXACT15', all_exact, 'G4_ALL_FINITE', all_fin, flush=True)
print('G4_PC_CLOSURE_PASS' if (all_exact and all_fin) else 'G4_PC_CLOSURE_FAIL', flush=True)