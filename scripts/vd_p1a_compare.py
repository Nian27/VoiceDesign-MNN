import numpy as np, os
D = r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/device-smoke/vd_p1'
names = ['norm_out','q_raw','k_raw','v_raw','q_norm','k_norm','q_rope','k_rope',
         'attn_out','o_proj_out','res1','post_norm','mlp_out','layer0_out','kv_k_out','kv_v_out']
def cos(a,b):
    a=np.asarray(a,np.float64).ravel(); b=np.asarray(b,np.float64).ravel()
    return float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12))
def load(dirp, i, nm):
    p = '%s/o%d_%s.raw' % (dirp, i, nm)
    return np.fromfile(p, dtype=np.float32) if os.path.exists(p) else None
first = None
print('%-11s %12s %12s %12s %12s' % ('boundary','CPU_cos','HTP_cos','CPU_maxabs','HTP_maxabs'))
for i, nm in enumerate(names):
    ref = np.fromfile(D + '/ref_%s.raw' % nm, dtype=np.float32)
    a = load(D + '/dc', i, nm); b = load(D + '/dh', i, nm)
    if a is None or b is None:
        print('%-11s MISSING' % nm); continue
    ca = cos(a, ref); cb = cos(b, ref)
    flag = ''
    if cb < 0.999 and first is None:
        first = (nm, i, ca, cb); flag = '   <== FIRST DIVERGENCE'
    print('%-11s %12.7f %12.7f %12.4f %12.4f%s' % (nm, ca, cb, np.abs(a).max(), np.abs(b).max(), flag))
print()
print('FIRST_DIVERGENCE_TENSOR =', first[0] if first else 'NONE')
print('  index=%d  CPU_cos=%.7f  HTP_cos=%.7f' % (first[1], first[2], first[3]) if first else '  -')
# k_rope detail
kr_a = load(D+'/dc', 7, 'k_rope'); kr_b = load(D+'/dh', 7, 'k_rope'); kr_r = np.fromfile(D+'/ref_k_rope.raw', dtype=np.float32)
if kr_a is not None and kr_b is not None:
    print()
    print('k_rope CPU[:6] =', np.round(kr_a.ravel()[:6], 4))
    print('k_rope HTP[:6] =', np.round(kr_b.ravel()[:6], 4))
    print('k_rope REF[:6] =', np.round(kr_r.ravel()[:6], 4))
    n = kr_r.size
    print('k_rope first-half HTP vs REF cos = %.6f' % cos(kr_b.ravel()[:n//2], kr_r.ravel()[:n//2]))
    print('k_rope second-half HTP vs REF cos= %.6f' % cos(kr_b.ravel()[n//2:], kr_r.ravel()[n//2:]))
print('P1A_CMP_DONE')
