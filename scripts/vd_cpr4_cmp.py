import re, os
D = r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/device-smoke/vd_cpr'
def parse(p):
    fr = {}
    for ln in open(p, encoding='utf-8', errors='ignore'):
        m = re.match(r'frame (\d+) code0 (\d+)((?: \d+){16}) \| u((?: [\d.]+){16}) \| cdf0 ([\d.]+) ([\d.]+) \| frame_ms ([\d.]+)', ln.strip())
        if m:
            fr[int(m.group(1))] = dict(code0=int(m.group(2)),
                                       codes=[int(x) for x in m.group(3).split()],
                                       u=[float(x) for x in m.group(4).split()],
                                       ms=float(m.group(7)))
    return fr
A = parse(D + '/r4cpu_records.txt'); B = parse(D + '/r4htp_records.txt')
print('frames parsed: CPU=%d HTP=%d' % (len(A), len(B)))
print()
total=0; exact=0
for f in sorted(set(A) & set(B)):
    a,b = A[f], B[f]
    us_same = all(abs(x-y)<1e-12 for x,y in zip(a['u'], b['u']))
    bad = next((g for g in range(16) if a['codes'][g]!=b['codes'][g]), None)
    total += 16; exact += 16 if bad is None else (bad or 0)
    tag = 'EXACT 16/16' if bad is None else ('first_bad code_index=%d (cpu=%d htp=%d)' % (bad, a['codes'][bad], b['codes'][bad]))
    print('frame %d  u_same=%s  code0 cpu=%4d htp=%4d  | %s' % (f, us_same, a['code0'], b['code0'], tag))
    if bad is not None:
        u = a['u'][bad]
        print('        u=%.9f  cpu_code=%d htp_code=%d' % (u, a['codes'][bad], b['codes'][bad]))
print()
print('16-CODE EXACT = %d/%d' % (exact, total))
print()
print('--- 性能 (每帧) ---')
for f in sorted(set(A) & set(B)):
    print('frame %d: CPU %8.1f ms   HTP %8.1f ms' % (f, A[f]['ms'], B[f]['ms']))
print()
print('NOTE: frame>=1 的 past 传递在探针里有 bug (把 1024 维 codepred hidden 拷进了 2048 维 past),')
print('      故 frame>=1 的输入两侧同为脏值, 属退化输入; frame 0 用的是真实 prefill hidden.')
print('CPR4_CMP_DONE')
