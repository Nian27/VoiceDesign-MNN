import numpy as np, os
D = r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/device-smoke'
OUT = D + '/vd_p2'
def cos(a,b):
    a=np.asarray(a,np.float64).ravel(); b=np.asarray(b,np.float64).ravel()
    return float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12))
print('=== G1: 28L GraphB v6 ===')
print('%-14s %12s %12s %12s' % ('tensor','CPU_cos','HTP_cos','torch_maxabs'))
for nm in ['hidden_states','cur_k','cur_v']:
    ref = np.fromfile(OUT + '/ref28_' + nm + '.raw', dtype=np.float32)
    a = np.fromfile(OUT + '/g1cpu/' + nm + '.raw', dtype=np.float32)
    b = np.fromfile(OUT + '/g1htp/' + nm + '.raw', dtype=np.float32)
    ca = cos(a, ref); cb = cos(b, ref)
    print('%-14s %12.7f %12.7f %12.5f   %s' % (nm, ca, cb, float(np.abs(ref).max()),
          'PASS' if cb >= 0.999 else 'FAIL'))
    if nm != 'hidden_states':
        # per-layer breakdown (28 layers x 1 x 8 x 128)
        cbv = cos(a, b)
        print('    CPU vs HTP direct cos = %.7f' % cbv)
        per = []
        for i in range(28):
            s = i*8*128; e = s + 8*128
            per.append(cos(a[s:e], b[s:e]))
        bad = [i for i,c in enumerate(per) if c < 0.999]
        print('    per-layer HTP-vs-CPU: min=%.6f max=%.6f  bad_layers=%s' % (min(per), max(per), bad if bad else 'none'))
        # how many layers have zero cur_v on HTP
        z = [i for i in range(28) if not np.any(b[i*8*128:(i+1)*8*128])]
        print('    HTP all-zero layers:', z if z else 'none')
