import numpy as np, os
D = r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/device-smoke'
names = ['norm_out','q_raw','k_raw','v_raw','q_norm','k_norm','q_rope','k_rope',
         'attn_out','o_proj_out','res1','post_norm','mlp_out','layer0_out','kv_k_out','kv_v_out']
def cos(a,b):
    a=np.asarray(a,np.float64).ravel(); b=np.asarray(b,np.float64).ravel()
    return float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12))
def load(dirp, i, nm):
    p = '%s/o%d_%s.raw' % (dirp, i, nm)
    return np.fromfile(p, dtype=np.float32) if os.path.exists(p) else None
print('%-11s %10s %10s %10s %10s' % ('boundary','CPU_cos','HTP_cos','CPU_maxab','HTP_maxab'))
first = None
for i, nm in enumerate(names):
    ref = np.fromfile(D + '/vd_p1/ref_%s.raw' % nm, dtype=np.float32)
    a = load(D + '/vd_p2/c2', i, nm); b = load(D + '/vd_p2/h2', i, nm)
    if a is None or b is None: print('%-11s MISSING' % nm); continue
    ca = cos(a, ref); cb = cos(b, ref)
    flag = ''
    if cb < 0.999 and first is None:
        first = (nm, i, ca, cb); flag = '  <== FIRST DIVERGENCE (v2)'
    print('%-11s %10.7f %10.7f %10.4f %10.4f%s' % (nm, ca, cb, np.abs(a).max(), np.abs(b).max(), flag))
print()
print('v2 FIRST_DIVERGENCE =', first)
