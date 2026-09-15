import os, torch, numpy as np, onnxruntime as ort, time
ART = r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/m2'
g = torch.load(os.path.join(ART, 'golden_io.pt'), map_location='cpu')
codes = g['codes'].numpy().astype(np.int64)
wav_ref = g['wav'].numpy()
print('golden codes', codes.shape, 'wav', wav_ref.shape, flush=True)
sess = ort.InferenceSession(os.path.join(ART, 'tokenizer_decoder.onnx'), providers=['CPUExecutionProvider'])
iname = sess.get_inputs()[0].name
oname = sess.get_outputs()[0].name
print('io', [(i.name, i.shape, i.type) for i in sess.get_inputs()], [(o.name, o.shape, o.type) for o in sess.get_outputs()], flush=True)
t0 = time.time()
out = sess.run([oname], {iname: codes})[0]
t1 = time.time()
wav_ort = out
print('ort wav', wav_ort.shape, 'elapsed', round(t1 - t0, 3), flush=True)
a = wav_ref.astype(np.float64).ravel()
b = wav_ort.astype(np.float64).ravel()
n = min(len(a), len(b))
a, b = a[:n], b[:n]
cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
mae = float(np.mean(np.abs(a - b)))
mx = float(np.max(np.abs(a - b)))
si = 10 * np.log10(np.sum(a**2) / (np.sum((a-b)**2) + 1e-12))
print('cosine', round(cos, 6), 'mae', round(mae, 8), 'maxabs', round(mx, 6), 'SI-SDR(dB)', round(float(si), 3), flush=True)
print('ORT_DONE', flush=True)