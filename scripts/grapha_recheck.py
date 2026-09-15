import numpy as np, MNN
D = r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/device-smoke'
emb = np.fromfile(D + '/grapha_emb_f32.raw', dtype=np.float32).reshape(1,34,2048)
pos = np.fromfile(D + '/grapha_pos_f32.raw', dtype=np.float32).reshape(3,1,34)
interp = MNN.Interpreter(D + '/grapha_prefill_static_fp16.mnn')
interp.setExternalFile(D + '/grapha_prefill_static_fp16.mnn.weight')
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
interp.runSession(s2)
ot = interp.getSessionOutput(s2, 'hidden_states')
osh = [d if d > 0 else 2048 for d in ot.getShape()]
ho = MNN.Tensor(osh, MNN.Halide_Type_Float, MNN.Tensor_DimensionType_Caffe)
ot.copyToHostTensor(ho)
mh = np.asarray(ho.getNumpyData()).reshape(osh).astype(np.float64)
ref = np.load(D + '/grapha_ref_h.npy').ravel().astype(np.float64)
n = min(mh.size, ref.size)
a = mh.ravel()[:n]; b = ref[:n]
c = float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-9))
mae = float(np.mean(np.abs(a-b))); mx = float(np.max(np.abs(a-b)))
print('MNN-fp16 vs ORT-ref (same raw): cosine', round(c,6), 'mae', round(mae,6), 'maxabs', round(mx,6))
print('mh maxabs', float(np.abs(mh).max()), 'ref maxabs', float(np.abs(ref).max()))