import numpy as np, torch, MNN, os
D = r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/device-smoke'
OUT = D + '/vd_m5b'
def cos(a,b):
    a=np.asarray(a,np.float64).ravel(); b=np.asarray(b,np.float64).ravel()
    return float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12))
# torch reference for frame_emb: rebuild from the saved onnx? use the exported onnx via onnxruntime
import onnxruntime as ort
so = ort.InferenceSession(OUT + '/frame_emb.onnx', providers=['CPUExecutionProvider'])
rng = np.random.RandomState(7)
codes = rng.randint(0, 2048, size=(1,16)).astype(np.int64)
ref = so.run(None, {'codes': codes})[0].astype(np.float32)
it = MNN.Interpreter(OUT + '/frame_emb_fp16.mnn'); s = it.createSession()
t = it.getSessionInput(s, 'codes')
ht = MNN.Tensor((1,16), MNN.Halide_Type_Int, codes.astype(np.int32).ravel(), MNN.Tensor_DimensionType_Caffe)
t.copyFromHostTensor(ht)
it.runSession(s)
ot = it.getSessionOutput(s, 'emb')
osh = [d if d > 0 else 1 for d in ot.getShape()]
ho = MNN.Tensor(osh, MNN.Halide_Type_Float, MNN.Tensor_DimensionType_Caffe)
ot.copyToHostTensor(ho)
got = np.asarray(ho.getNumpyData()).reshape(osh).astype(np.float32)
print('frame_emb MNN vs ONNX: cos=%.8f maxdiff=%.5f  refmax=%.4f gotmax=%.4f' % (
    cos(got, ref), float(np.abs(got-ref).max()), float(np.abs(ref).max()), float(np.abs(got).max())))
print('FRAME_EMB_VERIFY_DONE')
