import re, numpy as np, os
D = r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/device-smoke/vd_cpr'
def parse(p):
    for ln in open(p, encoding='utf-8', errors='ignore'):
        m = re.search(r'frame (\d+) code0 (\d+)((?: \d+){16}) \| u((?: [\d.]+){16}) \| cdf((?: [\d.]+){32})', ln.strip())
        if m:
            return dict(code0=int(m.group(2)),
                        codes=[int(x) for x in m.group(3).split()],
                        u=[float(x) for x in m.group(4).split()],
                        cdf=[float(x) for x in m.group(5).split()])
    return None
A = parse(D + '/r4cpu/r4_records.txt'); B = parse(D + '/r4htp/r4_records.txt')
print('CPU codes =', A['codes'])
print('HTP codes =', B['codes'])
bad = next((g for g in range(16) if A['codes'][g] != B['codes'][g]), None)
print('first_bad_code_index =', bad)
print()
print('=== margin at the divergence sample ===')
u = A['u'][bad]
lo_c, hi_c = A['cdf'][2*bad], A['cdf'][2*bad+1]
lo_h, hi_h = B['cdf'][2*bad], B['cdf'][2*bad+1]
print('code_index %d  u=%.9f  CPU_token=%d  HTP_token=%d' % (bad, u, A['codes'][bad], B['codes'][bad]))
print('  CPU interval [%.6f, %.6f)  width=%.6f  dist_to_edge=%.9f' % (lo_c, hi_c, hi_c-lo_c, min(u-lo_c, hi_c-u)))
print('  HTP interval [%.6f, %.6f)  width=%.6f  dist_to_edge=%.9f' % (lo_h, hi_h, hi_h-lo_h, min(u-lo_h, hi_h-u)))
print('  boundary shift: lower %+.9f   upper %+.9f' % (lo_h-lo_c, hi_h-hi_c))
print()
print('=== all 16 samples (CPU vs HTP) ===')
print('%-4s %4s %4s %12s %12s %10s %10s' % ('idx','cpu','htp','dist_cpu','dist_htp','width_cpu','width_htp'))
for g in range(16):
    uu = A['u'][g]
    lc,hc = A['cdf'][2*g], A['cdf'][2*g+1]
    lh,hh = B['cdf'][2*g], B['cdf'][2*g+1]
    dc = min(uu-lc, hc-uu); dh = min(uu-lh, hh-uu)
    tag = '' if A['codes'][g]==B['codes'][g] else '   <== DIFF'
    print('%-4d %4d %4d %12.8f %12.8f %10.6f %10.6f%s' % (g, A['codes'][g], B['codes'][g], dc, dh, hc-lc, hh-lh, tag))
print()
print('verdict: dist_to_edge at divergence = %.8f (cpu) / %.8f (htp)' % (min(u-lo_c,hi_c-u), min(u-lo_h,hi_h-u)))
