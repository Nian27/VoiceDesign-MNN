import io
p = r'E:/AndroidStudioProjects/qwen3tts-mnn/scripts/export/vd_g2b_cpu_margin.py'
s = io.open(p, encoding='utf-8').read()
n = s.count('cp.lm[')
s = s.replace('cp.lm[', 'cp.lm_head[')
io.open(p, 'w', encoding='utf-8').write(s)
print('patched cp.lm -> cp.lm_head:', n)