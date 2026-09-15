import subprocess, sys
raw = open(r'E:/AndroidStudioProjects/qwen3tts-mnn/scripts/export/tokcmp.txt', encoding='utf-8', errors='ignore').read().splitlines()
got = {}; exp = {}; cur = None
for ln in raw:
    ln = ln.strip()
    if ln.startswith('CASE '):
        cur = int(ln.split()[1])
    elif ln.startswith('GOT_A '): got[(cur,'A')] = ln[6:].strip().rstrip(',')
    elif ln.startswith('EXP_A '): exp[(cur,'A')] = ln[6:].strip().rstrip(',')
    elif ln.startswith('GOT_B '): got[(cur,'B')] = ln[6:].strip().rstrip(',')
    elif ln.startswith('EXP_B '): exp[(cur,'B')] = ln[6:].strip().rstrip(',')
tot = 0; ok = 0
for k in sorted(set(got) | set(exp)):
    g = got.get(k, ''); e = exp.get(k, '')
    tot += 1
    same = (g == e)
    ok += 1 if same else 0
    gl = g.split(',') if g else []; el = e.split(',') if e else []
    print('case %d seg %s : %s  (got %d ids, exp %d ids)' % (k[0], k[1], 'MATCH' if same else 'DIFF', len(gl), len(el)))
    if not same:
        for i in range(max(len(gl), len(el))):
            a = gl[i] if i < len(gl) else '-'; b = el[i] if i < len(el) else '-'
            if a != b: print('    first diff at %d: got=%s exp=%s' % (i, a, b)); break
print()
print('EXACT MATCH = %d/%d' % (ok, tot))
print('TOKENIZER_GATE =', 'PASS' if ok == tot and tot > 0 else 'FAIL')
