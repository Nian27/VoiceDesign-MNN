import numpy as np
D = r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/device-smoke'
dev = np.fromfile(D + '/gb_out2_dev_f32.raw', dtype=np.float32)
ref = np.load(D + '/gb_ref_hidden_fp32.npy').ravel()
n = min(len(dev), len(ref)); a = dev[:n].astype(np.float64); b = ref[:n].astype(np.float64)
cos = float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-9))
mae = float(np.mean(np.abs(a-b))); mx = float(np.max(np.abs(a-b)))
print('DEV vs REF: cosine', round(cos,6), 'mae', round(mae,6), 'maxabs', round(mx,6))
print('dev maxabs', float(np.abs(dev).max()), 'ref maxabs', float(np.abs(ref).max()))
# 也对比 PC-MNN fp16 输出（应该与设备一致）——如果没有保存则跳过