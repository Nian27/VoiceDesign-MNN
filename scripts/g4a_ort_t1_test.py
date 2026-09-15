import numpy as np, onnxruntime as ort, traceback
D = r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/device-smoke'
M22 = r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/m22/onnx'
try:
    sess = ort.InferenceSession(M22 + '/graphb_dynamic_fp32_t1.onnx', providers=['CPUExecutionProvider'])
    z = np.load(D + '/loop_pt/step0.npz')
    out = sess.run(None, {'inputs_embeds': z['emb'], 'position_ids': z['pos'], 'kv_k': z['kvk'], 'kv_v': z['kvv'], 'cache_position': z['cp'].astype(np.int64)})
    h = out[0]; ref = z['hidden']
    cos = float(np.dot(h.ravel(), ref.ravel())/(np.linalg.norm(h)*np.linalg.norm(ref)+1e-9))
    print('step0 cos', round(cos,6), 'max_ort', float(np.abs(h).max()), 'max_ref', float(np.abs(ref).max()), flush=True)
except Exception as e:
    traceback.print_exc()
print('DONE', flush=True)