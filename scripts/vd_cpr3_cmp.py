import numpy as np, os
D = r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/device-smoke/vd_cpr'
def cos(a,b):
    a=np.asarray(a,np.float64).ravel(); b=np.asarray(b,np.float64).ravel()
    return float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12))
def rd(side, nm):
    p = '%s/%s/%s' % (D, side, nm)
    return np.fromfile(p, dtype=np.float32) if os.path.exists(p) else None
print('=== CP-R3A: prefill (S=2, 暴露 Bug E 的那一段) ===')
for nm in ['prefill_hidden','prefill_kvk','prefill_kvv']:
    a = rd('cpu', nm + '.raw'); b = rd('htp', nm + '.raw')
    if a is None or b is None: print('  %-16s MISSING' % nm); continue
    print('  %-16s cos=%.7f  maxabs cpu=%.4f htp=%.4f' % (nm, cos(a,b), np.abs(a).max(), np.abs(b).max()))
print()
print('=== CP-R3B: forced-prefix 14 stages ===')
print('%-6s %12s %12s %12s' % ('stage','hidden_cos','kv_k_cos','kv_v_cos'))
hs=[];ks=[];vs=[]
for g in range(14):
    a_h = rd('cpu','step%02d_hidden.raw'%g); b_h = rd('htp','step%02d_hidden.raw'%g)
    a_k = rd('cpu','step%02d_kvk.raw'%g);   b_k = rd('htp','step%02d_kvk.raw'%g)
    a_v = rd('cpu','step%02d_kvv.raw'%g);   b_v = rd('htp','step%02d_kvv.raw'%g)
    if a_h is None or b_h is None: print('stage %2d MISSING'%g); continue
    ch=cos(a_h,b_h); ck=cos(a_k,b_k); cv=cos(a_v,b_v)
    hs.append(ch); ks.append(ck); vs.append(cv)
    print('%-6d %12.7f %12.7f %12.7f' % (g, ch, ck, cv))
print()
print('hidden  min=%.7f  drift=%+.7f' % (min(hs), hs[-1]-hs[0]))
print('kv_k    min=%.7f  drift=%+.7f' % (min(ks), ks[-1]-ks[0]))
print('kv_v    min=%.7f  drift=%+.7f' % (min(vs), vs[-1]-vs[0]))
