import numpy as np, onnxruntime as ort, MNN, time
D = r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/device-smoke'
np.random.seed(13)
LP, H, Dd, Cd = 34, 8, 128, 2048
emb = np.random.randn(1, LP, Cd).astype(np.float32) * 0.1
pos = np.zeros((3, 1, LP), dtype=np.float32)
for j in range(LP): pos[0,0,j]=j; pos[1,0,j]=j; pos[2,0,j]=j
sess = ort.InferenceSession(D + '/grapha_prefill_static_nonan.onnx', providers=['CPUExecutionProvider'])
t0=time.time()
ref_h, ref_k, ref_v = sess.run(None, {'inputs_embeds': emb, 'position_ids': pos})