import io
p = r'E:/AndroidStudioProjects/qwen3tts-mnn/scripts/export/vd_cpr2_clean.py'
s = io.open(p, encoding='utf-8').read()
old = 'am2[0,0,0,0]=0; am2[0,0,1,0]=0; am2[0,0,0,1]=0; am2[0,0,1,1]=0'
new = 'am2[0,0,0,0]=0; am2[0,0,1,0]=0; am2[0,0,1,1]=0   # causal: q0 only sees slot0, q1 sees slots 0..1'
n = s.count(old)
s = s.replace(old, new)
io.open(p, 'w', encoding='utf-8').write(s)
print('causal mask fixed, replaced=%d' % n)