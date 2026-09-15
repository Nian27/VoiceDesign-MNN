import numpy as np, onnxruntime as ort
D = r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/device-smoke'
np.random.seed(17)
ie = np.random.randn(1,1,1024).astype(np.float32)
for g in range(15):
    so = ort.InferenceSession(D + '/codepred_g' + str(g) + '.onnx', providers=['CPUExecutionProvider'])
    lg = so.run(None, {'inputs_embeds': ie})[0]
    np.save(D + '/codepred_ref_g' + str(g) + '.npy', lg)
    print('ref g', g, 'maxabs', round(float(np.abs(lg).max()),4), flush=True)
print('CODEPRED_REFS_SAVED', flush=True)