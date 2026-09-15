import io
p = r'E:/AndroidStudioProjects/qwen3tts-mnn/android/cp_r3_probe.cpp'
s = io.open(p, encoding='utf-8').read()
old = '''    for(int i=0;i<CAP;i++) am[i] = (i<=1)?0.f:-1e4f;
    memcpy(vAm2->writeMap<float>(), am.data(), CAP*4);'''
new = '''    { std::vector<float> am2((size_t)2*CAP, 0.f);
      for(int r=0;r<2;r++) for(int i=0;i<CAP;i++) am2[(size_t)r*CAP+i] = (i<=r)?0.f:-1e4f;
      memcpy(vAm2->writeMap<float>(), am2.data(), (size_t)2*CAP*4); }'''
n = s.count(old)
s = s.replace(old, new)
io.open(p,'w',encoding='utf-8').write(s)
print('prefill mask fixed:', n)