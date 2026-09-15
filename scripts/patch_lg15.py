import io
p = r'E:/AndroidStudioProjects/qwen3tts-mnn/scripts/export/vd_g2b_cpu_margin.py'
s = io.open(p, encoding='utf-8').read()
old = '        for g in range(15):'
new = '        lg15acc = np.zeros((15,2048), np.float32)\n        for g in range(15):'
n1 = s.count(old); s = s.replace(old, new)
old2 = '            codes.append(cg); us.append(ug); los.append(lo); his.append(hi)'
new2 = '            lg15acc[g] = lgg\n            codes.append(cg); us.append(ug); los.append(lo); his.append(hi)'
n2 = s.count(old2); s = s.replace(old2, new2)
old3 = "        emb = so_fe.run(None, {'codes': np.array([codes], dtype=np.int64)})[0].astype(np.float32)"
new3 = "        lg15acc.tofile(OUT + '/cpu_step%d_lg15.raw' % n)\n" + old3
n3 = s.count(old3); s = s.replace(old3, new3)
io.open(p, 'w', encoding='utf-8').write(s)
print('patched acc=%d store=%d dump=%d' % (n1, n2, n3))