import io
p = r'E:/AndroidStudioProjects/qwen3tts-mnn/android/g2b_loop2.cpp'
s = io.open(p, encoding='utf-8').read()
old1 = '        fprintf(fr," | cdf0 %.6f %.6f",los[0],his[0]);'
new1 = '        fprintf(fr," | cdf"); for(int g=0;g<16;g++) fprintf(fr," %.6f %.6f",los[g],his[g]);'
n1 = s.count(old1)
s = s.replace(old1, new1)
old2 = '        fprintf(fr," | stageTop"); for(int g=0;g<15;g++) fprintf(fr," %d",(int)stageTop[g]);'
new2 = ('        fprintf(fr," | stageTop"); for(int g=0;g<15;g++) fprintf(fr," %d",(int)stageTop[g]);\n'
        '        { char lp[256]; snprintf(lp,sizeof(lp),"%s/step%03d_lg15.raw",outDir,n); wf(lp,lg15.data(),15*2048); }')
n2 = s.count(old2)
s = s.replace(old2, new2)
io.open(p, 'w', encoding='utf-8').write(s)
print('patch1 cdf-all  replaced=%d' % n1)
print('patch2 lg15 dump replaced=%d' % n2)