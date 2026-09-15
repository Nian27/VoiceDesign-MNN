import numpy as np
D = r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/device-smoke'
ref = np.load(D + '/gb_ref_hidden_fp32.npy').astype(np.float64).ravel()
print('ref', ref.shape, 'maxabs %.4f' % np.abs(ref).max())
def cmp(name):
    a = np.fromfile(D + '/' + name, dtype=np.float32).astype(np.float64)
    n = min(a.size, ref.size)
    c = float(np.dot(a[:n], ref[:n]) / (np.linalg.norm(a[:n]) * np.linalg.norm(ref[:n]) + 1e-12))
    print('%-28s cos=%.7f  maxabs=%.4f  %s' % (name, c, np.abs(a).max(), 'PASS' if c >= 0.999 else 'FAIL'))
cmp('dev_gb_out_f32.raw')
cmp('dev_gb_mod3_out_f32.raw')
