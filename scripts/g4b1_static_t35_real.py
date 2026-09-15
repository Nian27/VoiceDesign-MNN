# G4-B1: static_t35 MNN + real G4-B0 boundary (KV62) - does T35 graph handle 62?
import numpy as np, MNN, json
D = r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/device-smoke'
B = r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/device-smoke/g4b_boundary'
LOG = open(D + '/g4b1.log', 'w')
def log(s):
    LOG.write(s + '\n'); LOG.flush()
cos_fn = lambda a,b: float(np.dot(a.ravel(),b.ravel())/(np.linalg.norm(a)*np.linalg.norm(b)+1e-9))
z = np.load(B + '/step_decode_inputs.npz')
kvk = np.load(B + '/kv_k_decode.npy')
kvv = np.load(B + '/kv_v_decode.npy')
emb = z['emb']; pos = z['pos']; cp = z['cp']
log('boundary emb ' + str(emb.shape) + ' pos ' + str(pos.shape) + ' cp ' + str(cp.tolist()) + ' kv ' + str(kvk.shape))
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
# try resize kv to 62
try:
    interp.resizeTensor(interp.getSessionInput(s2, 'kv_k'), (28,1,8,62,128))
    interp.resizeTensor(interp.getSessionInput(s2, 'kv_v'), (28,1,8,62,128))
    interp.resizeSession(s2)
    log('resize to 62 ok')
except Exception as e:
    log('resize ERR: ' + str(e)[:200])
feed('cache_position', cp.astype(np.int32), MNN.Halide_Type_Int)
feed('inputs_embeds', emb, MNN.Halide_Type_Float)
feed('kv_k', kvk, MNN.Halide_Type_Float)
feed('kv_v', kvv, MNN.Halide_Type_Float)
feed('position_ids', pos, MNN.Halide_Type_Float)
interp.runSession(s2)
h = pull_output('hidden_states')
log('hidden shape ' + str(h.shape) + ' max ' + str(round(float(np.abs(h).max()),4)) + ' finite ' + str(bool(np.isfinite(h).all())))
print('G4B1_DONE max', round(float(np.abs(h).max()),4), flush=True)