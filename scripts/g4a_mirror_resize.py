# G4-A: static_t35 with per-step resizeTensor to growing kv_len (MNN dynamic shapes)
import os, json, numpy as np, MNN
D = r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/device-smoke'
NUM_STEPS = 20
LOG = open(D + '/g4a_mirror_resize.log', 'w')
def log(s):
    LOG.write(s + '\n'); LOG.flush()
cos_fn = lambda a,b: float(np.dot(a.ravel(),b.ravel())/(np.linalg.norm(a)*np.linalg.norm(b)+1e-9))
interp = MNN.Interpreter(D + '/graphb_static_t35_fp16.mnn')
interp.setExternalFile(D + '/graphb_static_t35_fp16.mnn.weight')
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
traj = json.load(open(D + '/loop_traj_torch.json'))
results = []; all_ok = True
for step in range(NUM_STEPS):
    z = np.load(D + '/loop_pt/step' + str(step) + '.npz')
    emb = z['emb']; pos = z['pos']; kvk = z['kvk']; kvv = z['kvv']; hidden_ref = z['hidden']; cp = z['cp']
    n = kvk.shape[3]
    log('step ' + str(step) + ' kv_len ' + str(n) + ' cp ' + str(int(cp[0])))
    # resize graph inputs to actual kv_len
    try:
        interp.resizeTensor(interp.getSessionInput(s2, 'kv_k'), (28,1,8,n,128))
        interp.resizeTensor(interp.getSessionInput(s2, 'kv_v'), (28,1,8,n,128))
        interp.resizeSession(s2)
        log('  resize ok')
    except Exception as ex:
        log('  RESIZE ERR: ' + str(ex)[:200])
        all_ok = False
        break
    feed('cache_position', cp.astype(np.int32), MNN.Halide_Type_Int)
    feed('inputs_embeds', emb, MNN.Halide_Type_Float)
    feed('kv_k', kvk, MNN.Halide_Type_Float)
    feed('kv_v', kvv, MNN.Halide_Type_Float)
    feed('position_ids', pos, MNN.Halide_Type_Float)
    interp.runSession(s2)
    hidden_mnn = pull_output('hidden_states')
    cos_h = cos_fn(hidden_ref, hidden_mnn)
    fin = bool(np.isfinite(hidden_mnn).all())
    log('  cos ' + str(round(cos_h,6)) + ' finite ' + str(fin) + ' max_mnn ' + str(round(float(np.abs(hidden_mnn).max()),3)))
    results.append({'step': step, 'cos': round(cos_h,6), 'finite': fin})
    all_ok = all_ok and cos_h >= 0.999 and fin
log('G4A_RESIZE_ALL ' + str(all_ok))
with open(D + '/g4a_resize_result.json', 'w') as f:
    f.write(json.dumps({'results': results, 'all_pass': all_ok}, indent=2))
print('G4A_RESIZE_DONE', all_ok, flush=True)