import numpy as np, os
D = r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/device-smoke/vd_cpr'
lw = np.fromfile(D + '/lm_head_weight.f32', dtype=np.float32).reshape(15,2048,1024)
def cos(a,b):
    a=np.asarray(a,np.float64).ravel(); b=np.asarray(b,np.float64).ravel()
    return float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12))
def rd(side, nm): return np.fromfile('%s/%s/%s'%(D,side,nm), dtype=np.float32)
print('=== CP-R3A prefill (S=2, 因果掩码已修正) ===')
for nm in ['prefill_hidden','prefill_kvk','prefill_kvv']:
    a=rd('cpu',nm+'.raw'); b=rd('htp',nm+'.raw')
    print('  %-16s cos=%.7f  maxabs cpu=%.4f htp=%.4f' % (nm, cos(a,b), np.abs(a).max(), np.abs(b).max()))
hc=rd('cpu','prefill_hidden.raw'); hh=rd('htp','prefill_hidden.raw')
lh=hc@lw[0].T; lhh=hh@lw[0].T
print('  lm_head[0] logits cos=%.7f top1 %d/%d ovl50=%d' % (cos(lh,lhh), int(np.argmax(lh)), int(np.argmax(lhh)),
      len(set(np.argsort(-lh)[:50].tolist()) & set(np.argsort(-lhh)[:50].tolist()))))
print()
print('=== CP-R3B forced-prefix 14 stages ===')
print('%-6s %11s %11s %11s | %11s %6s %6s %5s' % ('stage','hidden_cos','kv_k_cos','kv_v_cos','logits_cos','top1c','top1h','ovl'))
hs=[];ks=[];vs=[];cs=[];ovs=[]
for g in range(14):
    a_h=rd('cpu','step%02d_hidden.raw'%g); b_h=rd('htp','step%02d_hidden.raw'%g)
    a_k=rd('cpu','step%02d_kvk.raw'%g);   b_k=rd('htp','step%02d_kvk.raw'%g)
    a_v=rd('cpu','step%02d_kvv.raw'%g);   b_v=rd('htp','step%02d_kvv.raw'%g)
    ch=cos(a_h,b_h); ck=cos(a_k,b_k); cv=cos(a_v,b_v)
    la=a_h@lw[g+1].T; lb=b_h@lw[g+1].T
    cl=cos(la,lb); ov=len(set(np.argsort(-la)[:50].tolist()) & set(np.argsort(-lb)[:50].tolist()))
    ta=int(np.argmax(la)); tb=int(np.argmax(lb))
    hs.append(ch); ks.append(ck); vs.append(cv); cs.append(cl); ovs.append(ov)
    print('%-6d %11.7f %11.7f %11.7f | %11.7f %6d %6d %5d' % (g,ch,ck,cv,cl,ta,tb,ov))
print()
print('hidden   min=%.7f  drift=%+.7f' % (min(hs), hs[-1]-hs[0]))
print('kv_k     min=%.7f' % min(ks))
print('kv_v     min=%.7f' % min(vs))
print('logits   min=%.7f' % min(cs))
print('top50    min=%d/50' % min(ovs))
print()
print('GATE  hidden>=0.999      :', all(c>=0.999 for c in hs))
print('GATE  kv>=0.999          :', all(c>=0.999 for c in ks) and all(c>=0.999 for c in vs))
print('GATE  logits_cos>=0.999  :', all(c>=0.999 for c in cs))
print('GATE  top50>=49/50       :', all(o>=49 for o in ovs))
print('CPR3_FINAL_DONE')
