import numpy as np, os
D = r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/device-smoke/vd_cpr'
lw = np.fromfile(D + '/lm_head_weight.f32', dtype=np.float32).reshape(15,2048,1024)
def cos(a,b):
    a=np.asarray(a,np.float64).ravel(); b=np.asarray(b,np.float64).ravel()
    return float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12))
def rd(side, nm):
    return np.fromfile('%s/%s/%s'%(D,side,nm), dtype=np.float32)
forced = np.fromfile(D + '/g3/forced_tokens.i32', dtype=np.int32).tolist()
print('=== CP-R3: host lm_head logits (fp32) on CPU/HTP hidden ===')
print('%-8s %-9s %11s %9s %6s %6s %6s' % ('seg','head','logits_cos','rel_l2','top1c','top1h','ovl50'))
def rel_l2(a,b):
    return float(np.linalg.norm(a-b)/(np.linalg.norm(b)+1e-30))
def cmp(hc, hh, head, tag):
    lc = hc @ lw[head].T; lh = hh @ lw[head].T
    ta = int(np.argmax(lc)); tb = int(np.argmax(lh))
    oa = set(np.argsort(-lc)[:50].tolist()); ob = set(np.argsort(-lh)[:50].tolist())
    c = cos(lc, lh)
    print('%-8s %-9s %11.7f %9.6f %6d %6d %6d' % (tag, 'lm_head[%d]'%head, c, rel_l2(lh,lc), ta, tb, len(oa&ob)))
    return c, ta, tb, len(oa&ob)
res = []
res.append(cmp(rd('cpu','prefill_hidden.raw'), rd('htp','prefill_hidden.raw'), 0, 'prefill'))
for g in range(14):
    res.append(cmp(rd('cpu','step%02d_hidden.raw'%g), rd('htp','step%02d_hidden.raw'%g), g+1, 'step%02d'%g))
cs = [r[0] for r in res]; ov = [r[4] for r in res]
print()
print('logits cos: min=%.7f max=%.7f' % (min(cs), max(cs)))
print('top50 overlap: min=%d max=%d   top1 全部一致: %s' % (min(ov), max(ov), all(r[1]==r[2] for r in res)))
print('gate  logits_cos>=0.999 :', all(c>=0.999 for c in cs))
print('gate  top50 >= 49/50    :', all(o>=49 for o in ov))
