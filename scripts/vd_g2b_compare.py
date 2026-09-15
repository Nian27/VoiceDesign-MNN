import os, re
D = r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/device-smoke/vd_p2/g2'
def parse(p):
    steps = {}
    for ln in open(p, encoding='utf-8', errors='ignore'):
        m = re.match(r'step (\d+) cp (\d+) gb_ms ([\d.]+) code0 (\d+)((?: -?\d+){16}) \| u((?: [\d.]+){16}) \| cdf0 ([\d.]+) ([\d.]+)', ln.strip())
        if m:
            st = int(m.group(1))
            codes = [int(x) for x in m.group(5).split()]
            us = [float(x) for x in m.group(6).split()]
            steps[st] = dict(cp=int(m.group(2)), code0=int(m.group(4)), codes=codes, u=us,
                             cdf=(float(m.group(7)), float(m.group(8))))
    return steps
A = parse(D + '/g2bcpu/records.txt')
B = parse(D + '/g2htp/records.txt')
print('steps parsed: CPU=%d HTP=%d' % (len(A), len(B)))
total = 0; exact = 0; first_bad = None
for st in sorted(set(A) & set(B)):
    a, b = A[st], B[st]
    us_same = all(abs(x-y) < 1e-12 for x, y in zip(a['u'], b['u']))
    bad = next((g for g in range(16) if a['codes'][g] != b['codes'][g]), None)
    total += 16
    if bad is None:
        exact += 16
    else:
        if first_bad is None:
            first_bad = (st, bad, a['codes'][bad], b['codes'][bad], a['u'][bad], a['cdf'])
        exact += bad
    tag = 'EXACT' if bad is None else ('FIRST_BAD code%d cpu=%d htp=%d' % (bad, a['codes'][bad], b['codes'][bad]))
    print('step %2d cp %d  u_same=%s  cpu_code0=%4d htp_code0=%4d  %s' % (st, a['cp'], us_same, a['code0'], b['code0'], tag))
print()
print('CODES EXACT = %d/%d' % (exact, total))
print('FIRST_BAD =', first_bad)
