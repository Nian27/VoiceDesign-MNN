# G4-B1: T128 static + KV[0:62] real, [62:128] zero, cp=[62] - does causal mask block zero KV?
import numpy as np, MNN
D = r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/device-smoke'
B = r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/device-smoke/g4b_boundary'
LOG = open(D + '/g4b1_t128_pad.log', 'w')
def log(s):
    LOG.write(s + '\n'); LOG.flush()
cos_fn = lambda a,b: float(np.dot(a.ravel(),b.ravel())/(np.linalg.norm(a)*np.linalg.norm(b)+1e-9))
z = np.load(B + '/step_decode_inputs.npz')
kvk62 = np.load(B + '/kv_k_decode.npy')
kvv62 = np.load(B + '/kv_v_decode.npy')
hidden_ref = np.load(B + '/decode_step0_hidden.npy')
emb = z['emb']; pos = z['pos']; cp = z['cp']
# pad to 128
kvk = np.zeros((28,1,8,128,128), dtype=np.float32)
kvv = np.zeros((28,1,8,128,128), dtype=np.float32)
kvk[:,:,:,:62,:] = kvk62; kvv[:,:,:,:62,:] = kvv62
log('padded kv ' + str(kvk.shape))
interp = MNN.Interpreter(D + '/graphb_static_t128_fp16.mnn')
interp.setExternalFile(D + '/graphb_static_t128_fp16.mnn.weight')
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
feed('cache_position', cp.astype(np.int32), MNN.Halide_Type_Int)
feed('inputs_embeds', emb, MNN.Halide_Type_Float)
feed('kv_k', kvk, MNN.Halide_Type_Float)
feed('kv_v', kvv, MNN.Halide_Type_Float)
feed('position_ids', pos, MNN.Halide_Type_Float)
interp.runSession(s2)
h = pull_output('hidden_states')
cos_h = cos_fn(hidden_ref, h)
fin = bool(np.isfinite(h).all())
log('cos ' + str(round(cos_h,6)) + ' finite ' + str(fin) + ' max_ref ' + str(round(float(np.abs(hidden_ref).max()),4)) + ' max_mnn ' + str(round(float(np.abs(h).max()),4)))
print('G4B1_PAD cos', round(cos_h,6), flush=True)