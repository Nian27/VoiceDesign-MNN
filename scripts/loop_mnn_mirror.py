# Gate4 MNN mirror replay: read torch trajectory npz, feed GraphB + codepred15, compare
import os, json, numpy as np, MNN
D = r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/device-smoke'
GRAPHB_INPUT_ORDER = ['cache_position', 'inputs_embeds', 'kv_k', 'kv_v', 'position_ids']
NUM_STEPS = 20
LOG = open(D + '/loop_mnn_diag.log', 'w')
def log(s):
    LOG.write(s + '\n'); LOG.flush()
cos_fn = lambda a,b: float(np.dot(a.ravel(),b.ravel())/(np.linalg.norm(a)*np.linalg.norm(b)+1e-9))
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
# codepred MNN 15
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
results = []
all_cos = True; all_exact = True; all_fin = True
for step in range(NUM_STEPS):
    z = np.load(D + '/loop_pt/step' + str(step) + '.npz')
    emb = z['emb']; pos = z['pos']; kvk = z['kvk']; kvv = z['kvv']; hidden_ref = z['hidden']; cp = z['cp']
    log('step ' + str(step) + ' emb ' + str(emb.shape) + ' kv ' + str(kvk.shape) + ' cp ' + str(int(cp[0])))
    feed('cache_position', cp.astype(np.int32), MNN.Halide_Type_Int)
    feed('inputs_embeds', emb, MNN.Halide_Type_Float)
    kk_pad = np.zeros((28,1,8,35,128), dtype=np.float32)
    kvv_pad = np.zeros((28,1,8,35,128), dtype=np.float32)
    n = kvk.shape[3]
    kk_pad[:,:,:,:n,:] = kvk
    kvv_pad[:,:,:,:n,:] = kvv
    feed('kv_k', kk_pad, MNN.Halide_Type_Float)
    feed('kv_v', kvv_pad, MNN.Halide_Type_Float)
    feed('position_ids', pos, MNN.Halide_Type_Float)
    interp.runSession(s2)
    hidden_mnn = pull_output('hidden_states')
    cos_h = cos_fn(hidden_ref, hidden_mnn)
    fin = bool(np.isfinite(hidden_mnn).all())
    all_cos = all_cos and cos_h >= 0.999
    all_fin = all_fin and fin
    log('  cos_h ' + str(round(cos_h,6)) + ' finite ' + str(fin) + ' max_ref ' + str(round(float(np.abs(hidden_ref).max()),3)) + ' max_mnn ' + str(round(float(np.abs(hidden_mnn).max()),3)))
    # codepred exact 15/15 (MNN vs torch trajectory codes in json)
    traj = json.load(open(D + '/loop_traj_torch.json'))
    codes_ref = traj['steps'][step]['codes']
    codes_mnn = []
    for g in range(15):
        lg = run_cp_mnn(z['hidden'], g)[0,0]
        codes_mnn.append(int(np.argmax(lg)))
    exact = codes_ref == codes_mnn
    all_exact = all_exact and exact
    log('  codes_ref ' + str(codes_ref))
    log('  codes_mnn ' + str(codes_mnn) + ' exact ' + str(exact))
    results.append({'step': step, 'cos_hidden': round(cos_h,6), 'hidden_finite': fin, 'codes_exact15': exact})
with open(D + '/loop_mirror_result.json', 'w') as f:
    f.write(json.dumps({'results': results, 'all_cos_pass': all_cos, 'all_exact15': all_exact, 'all_finite': all_fin}, indent=2))
print('MNN_MIRROR_DONE all_cos', all_cos, 'all_exact', all_exact, 'all_finite', all_fin, flush=True)