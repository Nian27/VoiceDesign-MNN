import numpy as np, onnxruntime as ort, MNN, time, os
D = r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/device-smoke'
np.random.seed(17)
ie = np.random.randn(1,1,1024).astype(np.float32)
# ORT 参照（g0..g14）
ref_logits = []
for g in range(15):
    so = ort.InferenceSession(D + '/codepred_g' + str(g) + '.onnx', providers=['CPUExecutionProvider'])
    lg = so.run(None, {'inputs_embeds': ie})[0]
    ref_logits.append(lg)
    print('ORT g', g, 'maxabs', round(float(np.abs(lg).max()),4), flush=True)
print('==== MNN ====')
for g in range(15):
    interp = MNN.Interpreter(D + '/codepred_g' + str(g) + '_fp16.mnn')
    s2 = interp.createSession()
    t = interp.getSessionInput(s2, 'inputs_embeds')
    ht = MNN.Tensor((1,1,1024), MNN.Halide_Type_Float, ie.ravel(), MNN.Tensor_DimensionType_Caffe)
    t.copyFromHostTensor(ht)
    ot = interp.getSessionOutput(s2, 'logits')
    interp.runSession(s2)
    osh = [d if d > 0 else 2048 for d in ot.getShape()]
    ho = MNN.Tensor(osh, MNN.Halide_Type_Float, MNN.Tensor_DimensionType_Caffe)
    ot.copyToHostTensor(ho)
    mh = np.asarray(ho.getNumpyData()).reshape(osh).astype(np.float64)
    ref = ref_logits[g].astype(np.float64)
    n = min(mh.size, ref.size); a = mh.ravel()[:n]; b = ref.ravel()[:n]
    c = float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-9))
    print('g', g, 'cosine', round(c,6), 'maxabs', round(float(np.abs(mh).max()),4), flush=True)
print('CODEPRED_PC_DONE', flush=True)