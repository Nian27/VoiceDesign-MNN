import numpy as np, os
D = r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/device-smoke'
ref = np.fromfile(D + '/vd_m5a_inputs/step0_hidden_ref.raw', dtype=np.float32).astype(np.float64)
for tag, f in [('CPU-golden', 'vd_m5b_out/step0_hidden_devmod.raw'),
               ('HTP', 'vd_m5b_out/step0_hidden_htp.raw'),
               ('HTP+ATTNOFF', 'vd_m5b_out/step0_hidden_attnoff.raw')]:
    p = D + '/' + f
    if not os.path.exists(p):
        print('%-12s MISSING' % tag); continue
    a = np.fromfile(p, dtype=np.float32).astype(np.float64)
    c = float(np.dot(a, ref) / (np.linalg.norm(a) * np.linalg.norm(ref) + 1e-12))
    print('%-12s cos=%.7f maxabs=%.4f %s' % (tag, c, np.abs(a).max(), 'PASS' if c >= 0.999 else 'FAIL'))
