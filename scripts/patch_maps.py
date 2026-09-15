import io
p = r'E:/AndroidStudioProjects/ReaderVoiceMobile/app-android/src/main/cpp/voicedesign_jni.cpp'
t = io.open(p, encoding='utf-8').read()
old = 'const float* cpW = nullptr;'
new = 'const float* cpW = nullptr;\n    void* lmMap = nullptr; void* cpMap = nullptr;'
n = t.count(old)
t = t.replace(old, new, 1)
io.open(p,'w',encoding='utf-8').write(t)
print('maps added:', n)