import numpy as np
D = r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/device-smoke'
pc = np.fromfile(D + '/gb_out_pc_mnn_f32.raw', dtype=np.float32)
dev = np.fromfile(D + '/gb_out2_dev_f32.raw', dtype=np.float32)
n = min(len(pc), len(dev)); a = pc[:n].astype(np.float64); b = dev[:n].astype(np.float64)
c = float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-9))
print('PC-MNN vs DEV-MNN: cosine', round(c,6), 'maxabs_diff', float(np.max(np.abs(a-b))))
print('pc maxabs', float(np.abs(pc).max()), 'dev maxabs', float(np.abs(dev).max()))
# 前 8 个值对比
print('pc head', pc[:8])
print('dev head', dev[:8])