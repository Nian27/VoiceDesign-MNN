import numpy as np
D = r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/device-smoke'
dev = np.fromfile(D + '/gb_out_dev_f32.raw', dtype=np.float32)
ref = np.load(D + '/gb_ref_hidden_fp32.npy').ravel()
n = min(len(dev), len(ref)); a = dev[:n].astype(np.float64); b = ref[:n].astype(np.float64)
cos = float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-9))
mae = float(np.mean(np.abs(a-b))); mx = float(np.max(np.abs(a-b)))
rel_l2 = float(np.linalg.norm(a-b)/np.linalg.norm(a))
print('n', n)
print('DEVICE vs PC-REF: cosine', round(cos,6), 'mae', round(mae,6), 'maxabs', round(mx,6), 'rel_l2', round(rel_l2,6))
print('dev maxabs', float(np.abs(dev).max()), 'ref maxabs', float(np.abs(ref).max()))