import numpy as np
D = r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/device-smoke'
dev = np.fromfile(D + '/gb_mod2_out_dev_f32.raw', dtype=np.float32)
ref = np.load(D + '/gb_ref_hidden_fp32.npy').ravel()
n = min(len(dev), len(ref)); a = dev[:n].astype(np.float64); b = ref[:n].astype(np.float64)
c = float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-9))
mae = float(np.mean(np.abs(a-b))); mx = float(np.max(np.abs(a-b)))
rel = float(np.linalg.norm(a-b)/np.linalg.norm(a))
print('Module-DEV vs ORT-ref: cosine', round(c,6), 'mae', round(mae,6), 'maxabs', round(mx,6), 'rel_l2', round(rel,6))
# 也对比 PC-MNN 输出（应为同源）
pcm = np.fromfile(D + '/gb_out_pc_mnn_f32.raw', dtype=np.float32)
n2 = min(len(dev), len(pcm)); x = dev[:n2].astype(np.float64); y = pcm[:n2].astype(np.float64)
c2 = float(np.dot(x,y)/(np.linalg.norm(x)*np.linalg.norm(y)+1e-9))
print('Module-DEV vs PC-MNN: cosine', round(c2,6))