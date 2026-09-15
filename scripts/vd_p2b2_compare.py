import numpy as np, os
D = r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/device-smoke'
names = ['norm_out','q_raw','k_raw','v_raw','q_norm','k_norm','q_rope','k_rope','qk_logits','qk_masked','attn_probs',
         'attn_out','o_proj_out','res1','post_norm','mlp_out','layer0_out','kv_k_out','kv_v_out']
def cos(a,b):
    a=np.asarray(a,np.float64).ravel(); b=np.asarray(b,np.float64).ravel()
    n=np.isfinite(a)&np.isfinite(b)
    if n.sum()==0: return float('nan')
    a=a[n]; b=b[n]
    return float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12))
def load(dirp,i,nm):
    p='%s/o%d_%s.raw'%(dirp,i,nm)
    return np.fromfile(p,dtype=np.float32) if os.path.exists(p) else None
print('%-11s %11s %11s %11s %11s' % ('boundary','CPU_cos','HTP_cos','CPU_maxab','HTP_maxab'))
first=None
for i,nm in enumerate(names):
    ref=np.fromfile(D+'/vd_p1/ref_%s.raw'%nm,dtype=np.float32) if os.path.exists(D+'/vd_p1/ref_%s.raw'%nm) else None
    a=load(D+'/vd_p2/c3',i,nm); b=load(D+'/vd_p2/h3',i,nm)
    if a is None or b is None: print('%-11s MISSING'%nm); continue
    ca=cos(a,ref) if ref is not None else float('nan')
    cb=cos(b,ref) if ref is not None else float('nan')
    fin=lambda x: float(np.abs(x[np.isfinite(x)]).max()) if np.isfinite(x).any() else float('nan')
    flag=''
    if ref is not None and cb<0.999 and first is None:
        first=(nm,i,ca,cb); flag='  <== FIRST DIVERGENCE'
    print('%-11s %11.7f %11.7f %11.4f %11.4f%s'%(nm,ca,cb,fin(a),fin(b),flag))
print()
print('P2B2 FIRST_DIVERGENCE =',first)
# ---- KV slot analysis (user-requested) ----
cp=62
kv_v_in=np.fromfile(D+'/vd_m5a_inputs/step0_kvv.raw',dtype=np.float32).reshape(28,1,8,768,128)
kv_k_in=np.fromfile(D+'/vd_m5a_inputs/step0_kvk.raw',dtype=np.float32).reshape(28,1,8,768,128)
for tag,vin,dirp in [('CPU',kv_v_in,D+'/vd_p2/c3'),('HTP',kv_v_in,D+'/vd_p2/h3')]:
    o=load(dirp,18,'kv_v_out')
    if o is None: continue
    o=o.reshape(28,1,8,768,128)
    old=o[:,:,:,:cp,:]; cur=o[:,:,:,cp:cp+1,:]; fut=o[:,:,:,cp+1:,:]
    iold=vin[:,:,:,:cp,:]; ifut=vin[:,:,:,cp+1:,:]
    z=lambda x: int((x.ravel()==0).sum())
    print('%s kv_v_out: old-slot cos=%.6f zero_frac=%.3f | cur maxabs=%.4f | future zero_frac=%.3f (in_zero=%.3f)'%(
        tag, cos(old,iold), z(old)/old.size, np.abs(cur).max(), z(fut)/fut.size, z(ifut)/ifut.size))
