import numpy as np, os
D = r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/device-smoke/vd_p2/g2'
def cos(a,b):
    a=np.asarray(a,np.float64).ravel(); b=np.asarray(b,np.float64).ravel()
    return float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12))
print('%-6s %12s %12s %12s' % ('step','hidden_cos','cur_k_cos','cur_v_cos'))
hs=[];ks=[];vs=[]
for n in range(20):
    r={}
    for t in ['hidden','curk','curv']:
        a=np.fromfile('%s/g2cpu/step%02d_%s.raw'%(D,n,t),dtype=np.float32)
        b=np.fromfile('%s/g2htp/step%02d_%s.raw'%(D,n,t),dtype=np.float32)
        r[t]=cos(a,b)
    hs.append(r['hidden']); ks.append(r['curk']); vs.append(r['curv'])
    print('%-6d %12.7f %12.7f %12.7f' % (n, r['hidden'], r['curk'], r['curv']))
print()
print('hidden  min=%.7f  drift(first->last)=%+.7f' % (min(hs), hs[-1]-hs[0]))
print('cur_k   min=%.7f  drift(first->last)=%+.7f' % (min(ks), ks[-1]-ks[0]))
print('cur_v   min=%.7f  drift(first->last)=%+.7f' % (min(vs), vs[-1]-vs[0]))
print('G2A_VERDICT hidden>=0.999 over 20 steps:', all(c>=0.999 for c in hs))
print('G2A_VERDICT cur_v >=0.999 over 20 steps:', all(c>=0.999 for c in vs))
