import numpy as np, onnxruntime as ort
D = r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/device-smoke'
np.random.seed(13)
LP = 34
emb = (np.random.randn(1, LP, 2048) * 0.1).astype(np.float32)
pos = np.zeros((3, 1, LP), dtype=np.float32)
for j in range(LP): pos[0,0,j]=j; pos[1,0,j]=j; pos[2,0,j]=j
sess = ort.InferenceSession(D + '/grapha_prefill_static_nonan.onnx', providers=['CPUExecutionProvider'])
h, k, v = sess.run(None, {'inputs_embeds': emb, 'position_ids': pos})
np.save(D + '/grapha_ref_h.npy', h); np.save(D + '/grapha_ref_k.npy', k); np.save(D + '/grapha_ref_v.npy', v)
emb.tofile(D + '/grapha_emb_f32.raw'); pos.tofile(D + '/grapha_pos_f32.raw')
print('REF saved, finite', bool(np.isfinite(h).all()), 'maxabs', float(np.abs(h).max()))