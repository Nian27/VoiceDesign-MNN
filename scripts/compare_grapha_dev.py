import numpy as np
D = r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/device-smoke'
dev_h = np.fromfile(D + '/ga_dev_h_f32.raw', dtype=np.float32)
dev_k = np.fromfile(D + '/ga_dev_k_f32.raw', dtype=np.float32)
dev_v = np.fromfile(D + '/ga_dev_v_f32.raw', dtype=np.float32)
ref_h = np.load(D + '/grapha_ref_h.npy').ravel()
ref_k = np.load(D + '/grapha_ref_k.npy').ravel()
ref_v = np.load(D + '/grapha_ref_v.npy').ravel()
def cmp(dev, ref, tag):
    n = min(dev.size, ref.size); a = dev[:n].astype(np.float64); b = ref[:n].astype(np.float64)
    c = float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-9))
    m = float(np.max(np.abs(a-b))); rel = float(np.linalg.norm(a-b)/np.linalg.norm(a))
    print(tag, 'cosine', round(c,6), 'maxabs_diff', round(m,6), 'rel_l2', round(rel,6))
cmp(dev_h, ref_h, 'hidden:')
cmp(dev_k, ref_k, 'kv_k:  ')
cmp(dev_v, ref_v, 'kv_v:  ')