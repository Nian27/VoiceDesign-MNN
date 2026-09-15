import numpy as np, os
D = r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/device-smoke/vd_p2'
G2 = D + '/g2'
def cos(a,b):
    a=np.asarray(a,np.float64).ravel(); b=np.asarray(b,np.float64).ravel()
    return float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12))
cpu = np.fromfile(D + '/cpu_step1_lg15.raw', dtype=np.float32).reshape(15,2048)
htp = np.fromfile(G2 + '/g2bhtp3/step001_lg15.raw', dtype=np.float32).reshape(15,2048)
print('=== step1 codepred logits: CPU vs HTP per AR stage ===')
print('%-6s %11s %10s %10s %10s' % ('stage','logits_cos','top1_cpu','top1_htp','top50_ovl'))
first=None
for g in range(15):
    a = cpu[g]; b = htp[g]
    ta = int(np.argmax(a)); tb = int(np.argmax(b))
    oa = set(np.argsort(-a)[:50].tolist()); ob = set(np.argsort(-b)[:50].tolist())
    ovl = len(oa & ob)
    c = cos(a,b)
    tag = ''
    if (ta != tb or ovl < 50) and first is None:
        first = (g, c, ta, tb, ovl); tag = '  <== FIRST STAGE DIVERGENCE'
    if g < 5 or tag:
        print('%-6d %11.7f %10d %10d %10d%s' % (g, c, ta, tb, ovl, tag))
print()
print('first_stage_divergence =', first)
# margin detail for stage 0
def interval(lg, tok, u, temp=0.9, topk=50):
    order = np.argsort(-(lg/temp), kind='stable')[:topk]
    vals = lg[order]/temp; mx = vals.max(); p = np.exp(vals-mx); p/=p.sum()
    acc = np.cumsum(p)
    j = int(np.where(order == tok)[0][0]) if tok in order else -1
    if j < 0: return None
    lo = float(acc[j]-p[j]); hi = float(acc[j])
    return lo, hi, float(min(u-lo, hi-u))
u1 = 0.767108981
print()
print('=== first bad sample detail (step 1, code_index 1 = stage 0) ===')
print('u = %.9f' % u1)
for tag, lg, tok in [('CPU', cpu[0], 885), ('HTP', htp[0], 2000)]:
    r = interval(lg, tok, u1)
    print('%s token=%4d  interval=[%.6f, %.6f)  dist_to_boundary=%.9f' % (tag, tok, r[0], r[1], r[2]))
print()
print('logits_cos(stage0) = %.7f' % cos(cpu[0], htp[0]))
d = np.abs(cpu[0]-htp[0]); print('logits maxdiff=%.5f  meanabs=%.6f' % (float(d.max()), float(d.mean())))
# what token would CPU have chosen at HTP's interval boundaries
print()
print('--- boundary shift check ---')
o_cpu = np.argsort(-(cpu[0]/0.9))[:50]; o_htp = np.argsort(-(htp[0]/0.9))[:50]
print('CPU top5 =', o_cpu[:5].tolist(), ' HTP top5 =', o_htp[:5].tolist())
print('CPU rank of 885 =', int(np.where(o_cpu==885)[0][0]), ' CPU rank of 2000 =', int(np.where(o_cpu==2000)[0][0]) if 2000 in o_cpu else 'not in top50')
print('HTP rank of 885 =', int(np.where(o_htp==885)[0][0]) if 885 in o_htp else 'not in top50', ' HTP rank of 2000 =', int(np.where(o_htp==2000)[0][0]))
print('MARGIN_DONE')
