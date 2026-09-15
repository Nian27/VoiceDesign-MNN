import numpy as np, MNN, time, sys
which = sys.argv[1] if len(sys.argv) > 1 else 'fp32'
D = r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/device-smoke'
emb = np.fromfile(D + '/grapha_emb_f32.raw', dtype=np.float32).reshape(1, 34, 2048)
pos = np.fromfile(D + '/grapha_pos_f32.raw', dtype=np.float32).reshape(3, 1, 34)
interp = MNN.Interpreter(D + '/grapha_prefill_static_' + which + '.mnn')
s2 = interp.createSession()
interp.resizeTensor(interp.getSessionInput(s2,'inputs_embeds'), (1,34,2048))
interp.resizeTensor(interp.getSessionInput(s2,'position_ids'), (3,1,34))
interp.resizeSession(s2)
def feed(name, arr):
    t = interp.getSessionInput(s2, name)
    ht = MNN.Tensor(tuple(arr.shape), MNN.Halide_Type_Float, arr.ravel(), MNN.Tensor_DimensionType_Caffe)
    t.copyFromHostTensor(ht)
feed('inputs_embeds', emb); feed('position_ids', pos)
oh = interp.getSessionOutput(s2, 'hidden_states')
okk = interp.getSessionOutput(s2, 'kv_k')
ovv = interp.getSessionOutput(s2, 'kv_v')
import time; t0=time.time(); interp.runSession(s2); t1=time.time()
def pull(t):
    osh = [d if d > 0 else 1 for d in t.getShape()]
    ho = MNN.Tensor(osh, MNN.Halide_Type_Float, MNN.Tensor_DimensionType_Caffe)
    t.copyToHostTensor(ho)
    return np.asarray(ho.getNumpyData()).reshape(osh)
mh = pull(oh); mk = pull(okk); mv = pull(ovv)
print(which, 'elapsed', round(t1-t0,2), 'maxabs_h', float(np.abs(mh).max()), flush=True)
ref_h = np.load(D + '/grapha_ref_h.npy')
af = ref_h.astype(np.float64).ravel(); bf = mh.astype(np.float64).ravel()
n = min(len(af), len(bf)); af=af[:n]; bf=bf[:n]
c = float(np.dot(af,bf)/(np.linalg.norm(af)*np.linalg.norm(bf)+1e-9))
print('cosine vs ORT:', round(c,6), 'maxabs_diff', float(np.max(np.abs(af-bf))), flush=True)