import io
p = r'E:/AndroidStudioProjects/qwen3tts-mnn/android/cp_r4_probe.cpp'
s = io.open(p, encoding='utf-8').read()
old = '        fprintf(fr," | cdf0 %.6f %.6f", los[0], his[0]);'
new = '        fprintf(fr," | cdf"); for(int g=0;g<16;g++) fprintf(fr," %.6f %.6f", los[g], his[g]);'
n = s.count(old)
s = s.replace(old, new)
io.open(p,'w',encoding='utf-8').write(s)
print('cdf-all patch:', n)