import numpy as np, MNN
D = r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/device-smoke'
emb = np.fromfile(D + '/grapha_emb_f32.raw', dtype=np.float32).reshape(1,34,2048)
pos = np.fromfile(D + '/grapha_pos_f32.raw', dtype=np.float32).reshape(3,1,34)
interp = MNN.Interpreter(D + '/grapha_prefill_static_fp32.mnn')
interp.setExternalFile(D + '/grapha_prefill_static_fp32.mnn.weight')
s2 = interp.createSession()
interp.resizeTensor(interp.getSessionInput(s2,'inputs_embeds'), (1,34,2048))
interp.resizeTensor(interp.getSessionInput(s2,'position_ids'), (3,1,34))
interp.resizeSession(s2)
def feed(name, arr, dt):
    t = interp.getSessionInput(s2, name)
    ht = MNN.Tensor(tuple(arr.shape), dt, arr.ravel(), MNN.Tensor_DimensionType_Caffe)
    t.copyFromHostTensor(ht)
feed('inputs_embeds', emb, MNN.Halide_Type_Float)
feed('position_ids', pos, MNN.Halide_Type_Float)
import time; t0=time.time(); interp.runSession(s2); t1=time.time()
ot = interp.getSessionOutput(s2, 'hidden_states')
osh = [d if d > 0 else 2048 for d in ot.getShape()]
ho = MNN.Tensor(osh, MNN.Halide_Type_Float, MNN.Tensor_DimensionType_Caffe)
ot.copyToHostTensor(ho)
mh = np.asarray(ho.getNumpyData()).reshape(osh).astype(np.float64)
ref = np.load(D + '/grapha_ref_h.npy').ravel().astype(np.float64)
n = min(mh.size, ref.size); a = mh.ravel()[:n]; b = ref[:n]
c = float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-9))
print('MNN-fp32 vs ORT-ref: cosine', round(c,6), 'mae', round(float(np.mean(np.abs(a-b))),6), 'elapsed', round(t1-t0,2))