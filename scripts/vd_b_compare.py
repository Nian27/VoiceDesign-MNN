import numpy as np, os
D = r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/device-smoke'
names = ['norm_out','q_rope','k_rope','k_all','k_T','qk_logits','qk_masked','attn_probs','attn_out','layer0_out',
         'kv_k_slot','kv_v_slot','kv_k_old0','kv_v_old0']
def cos(a,b):
    a=np.asarray(a,np.float64).ravel(); b=np.asarray(b,np.float64).ravel()
    n=np.isfinite(a)&np.isfinite(b)
    a=a[n]; b=b[n]
    return float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12))
print('%-11s %11s %11s | %s' % ('boundary','CPU_cos','HTP_cos','verdict'))
worst=1.0; firstbad=None
for i,nm in enumerate(names):
    ref=np.fromfile(D+'/vd_p1/ref_%s.raw'%nm,dtype=np.float32)
    pa='%s/vd_p2/c5/o%d_%s.raw'%(D,i,nm); pb='%s/vd_p2/h5/o%d_%s.raw'%(D,i,nm)
    if not os.path.exists(pa) or not os.path.exists(pb): print('%-11s MISSING'%nm); continue
    a=np.fromfile(pa,dtype=np.float32); b=np.fromfile(pb,dtype=np.float32)
    ca=cos(a,ref); cb=cos(b,ref)
    v='OK'
    if cb<0.999: v='<== FAIL'; firstbad=firstbad or (nm,cb)
    if cb<worst: worst=cb
    print('%-11s %11.7f %11.7f | %s'%(nm,ca,cb,v))
print()
print('B v5 HTP worst cos = %.7f  first_bad = %s' % (worst, firstbad))
print('CPU worst cos = %.7f' % min(cos(np.fromfile('%s/vd_p2/c5/o%d_%s.raw'%(D,i,nm),dtype=np.float32), np.fromfile(D+'/vd_p1/ref_%s.raw'%nm,dtype=np.float32)) for i,nm in enumerate(names)))
