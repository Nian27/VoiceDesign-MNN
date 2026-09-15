import io
m = r'E:/AndroidStudioProjects/ReaderVoiceMobile/app-android/src/main/AndroidManifest.xml'
s = io.open(m, encoding='utf-8').read()
if 'FOREGROUND_SERVICE' not in s:
    s = s.replace('<application', '    <uses-permission android:name="android.permission.FOREGROUND_SERVICE" />\n    <uses-permission android:name="android.permission.FOREGROUND_SERVICE_DATA_SYNC" />\n    <uses-permission android:name="android.permission.POST_NOTIFICATIONS" />\n\n    <application', 1)
svc = '        <service\n            android:name=".VoiceDesignService"\n            android:exported="false"\n            android:foregroundServiceType="dataSync" />\n'
if 'VoiceDesignService' not in s:
    s = s.replace('        <activity', svc + '        <activity', 1)
io.open(m,'w',encoding='utf-8').write(s)
print('manifest service=%s perm=%s' % ('VoiceDesignService' in s, 'FOREGROUND_SERVICE' in s))
p = r'E:/AndroidStudioProjects/ReaderVoiceMobile/app-android/src/main/cpp/voicedesign_jni.cpp'
t = io.open(p, encoding='utf-8').read()
t = t.replace('#include <memory>', '#include <memory>\n#include <sys/mman.h>\n#include <fcntl.h>\n#include <unistd.h>', 1)
t = t.replace('std::vector<float> lmW, cpW;          // (15,2048,1024), (15,2048,2048)', 'const float* lmW = nullptr;\n    const float* cpW = nullptr;')
t = t.replace('s->lmW = readVec(d+"lm_head_weight.f32", (size_t)15*2048*1024);', 's->lmW = (const float*)mmapFile(d+"lm_head_weight.f32", (size_t)15*2048*1024, &s->lmMap);')
t = t.replace('s->cpW = readVec(d+"codec_emb_weight.f32", (size_t)15*2048*2048);', 's->cpW = (const float*)mmapFile(d+"codec_emb_weight.f32", (size_t)15*2048*2048, &s->cpMap);')
t = t.replace('if (s->lmW.empty() || s->cpW.empty() ||', 'if (s->lmW == nullptr || s->cpW == nullptr ||')
t = t.replace('    std::vector<char> lhB=rf(argv[5]), emB=rf(argv[6]), pastB=rf(argv[7]), cosB=rf(argv[8]), sinB=rf(argv[9]);', '    std::vector<char> pastB=rf(argv[7]), cosB=rf(argv[8]), sinB=rf(argv[9]);')
t = t.replace('void* lmMap = nullptr; void* cpMap = nullptr;', 'void* lmMap = nullptr; void* cpMap = nullptr;')
helper = 'static void* mmapFile(const std::string& p, size_t bytes, void** out) {\n    int fd = open(p.c_str(), O_RDONLY);\n    if (fd < 0) return nullptr;\n    void* q = mmap(nullptr, bytes, PROT_READ, MAP_PRIVATE, fd, 0);\n    close(fd);\n    if (q == MAP_FAILED) return nullptr;\n    *out = q;\n    return q;\n}\n'
t = t.replace('static std::shared_ptr<Module> loadM(', helper + 'static std::shared_ptr<Module> loadM(', 1)
io.open(p,'w',encoding='utf-8').write(t)
print('jni mmap helper=%s lmWptr=%s' % ('mmapFile' in t, 'const float* lmW = nullptr' in t))