# PC MNN check with the SAME raw files used on device (isolate ref/input vs device path).
import numpy as np, MNN, os
D = r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/device-smoke'
IN = D + '/vd_m5a_inputs'
def rd(n): return np.fromfile(IN + '/' + n, dtype=np.float32)
emb = rd('step0_emb.raw'); pos = rd('step0_pos.raw')
kvk = rd('step0_kvk.raw'); kvv = rd('step0_kvv.raw')
cp = np.fromfile(IN + '/step0_cp.raw', dtype=np.int32)
ref = rd('step0_hidden_ref.raw')
print('shapes emb', emb.shape, 'pos', pos.shape, 'kvk', kvk.shape, 'cp', cp.shape)
emb = emb.reshape(1,1,2048); pos = pos.reshape(3,1,1)
kvk = kvk.reshape(28,1,8,768,128); kvv = kvv.reshape(28,1,8,768,128)
p = D + '/graphb_deploy_hand_fp16.mnn'
interp = MNN.Interpreter(p); interp.setExternalFile(p + '.weight')
s = interp.createSession()
def feed(name, arr, dt):
    t = interp.getSessionInput(s, name)
    ht = MNN.Tensor(tuple(arr.shape), dt, arr.ravel(), MNN.Tensor_DimensionType_Caffe)
    t.copyFromHostTensor(ht)
feed('inputs_embeds', emb, MNN.Halide_Type_Float)
feed('position_ids', pos, MNN.Halide_Type_Float)
feed('kv_k', kvk, MNN.Halide_Type_Float)
feed('kv_v', kvv, MNN.Halide_Type_Float)
feed('cache_position', cp, MNN.Halide_Type_Int)
interp.runSession(s)
ot = interp.getSessionOutput(s, 'hidden_states')
ho = MNN.Tensor((1,1,2048), MNN.Halide_Type_Float, MNN.Tensor_DimensionType_Caffe)
ot.copyToHostTensor(ho)
h = np.asarray(ho.getNumpyData()).reshape(-1).astype(np.float64)
r = ref.astype(np.float64)
c = float(np.dot(h,r)/(np.linalg.norm(h)*np.linalg.norm(r)+1e-12))
print('PC MNN vs ref: cos=%.7f maxabs=%.4f refmax=%.4f' % (c, float(np.abs(h-r).max()), float(np.abs(r).max())))
print('PC_MNN_REF_CHECK_DONE')
