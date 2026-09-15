import numpy as np, os
D = r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/device-smoke'
def cos(a,b):
    a=np.asarray(a,np.float64).ravel(); b=np.asarray(b,np.float64).ravel()
    return float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12))
check = ['k_all','k_T','qk_logits','qk_masked','attn_probs','attn_out','res1','layer0_out']
idx = {'k_all':8,'k_T':9,'qk_logits':10,'qk_masked':11,'attn_probs':12,'attn_out':13,'res1':16,'layer0_out':18}
print('%-11s %11s %11s | %s' % ('boundary','CPU_cos','HTP_cos','verdict'))
first=None
for nm in check:
    i = idx[nm]
    ref = np.fromfile(D + '/vd_p1/ref_%s.raw' % nm, dtype=np.float32)
    pa = '%s/vd_p2/c4/o%d_%s.raw' % (D, i, nm)
    pb = '%s/vd_p2/h4/o%d_%s.raw' % (D, i, nm)
    if not os.path.exists(pa) or not os.path.exists(pb): print('%-11s MISSING'%nm); continue
    a=np.fromfile(pa,dtype=np.float32); b=np.fromfile(pb,dtype=np.float32)
    ca=cos(a,ref); cb=cos(b,ref)
    v=''
    if cb<0.999:
        if first is None: first=(nm,cb); v='  <== first bad'
    print('%-11s %11.7f %11.7f |%s' % (nm, ca, cb, v))
print()
print('A first_bad =', first)
# k_all slot analysis: is HTP's k_all old-slot zeroed (=> defect3 contamination)?
kvk_in = np.fromfile(D + '/vd_m5a_inputs/step0_kvk.raw', dtype=np.float32).reshape(28,1,8,768,128)
cp=62
ka_cpu = np.fromfile(D+'/vd_p2/c4/o8_k_all.raw',dtype=np.float32).reshape(1,16,768,128)
ka_htp = np.fromfile(D+'/vd_p2/h4/o8_k_all.raw',dtype=np.float32).reshape(1,16,768,128)
src = np.repeat(kvk_in[0], 2, axis=0).reshape(1,16,768,128)
for tag,x in [('CPU',ka_cpu),('HTP',ka_htp)]:
    old = x[:,:,:cp,:]; cur = x[:,:,cp,:]
    z = int((old.ravel()==0).sum())
    print('%s k_all: old-slot vs src cos=%.6f zero_frac=%.3f | cur maxabs=%.4f' % (tag, cos(old, src[:,:,:cp,:]), z/old.size, float(np.abs(cur).max())))
