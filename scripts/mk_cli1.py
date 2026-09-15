import io, os
SRC = r'E:/AndroidStudioProjects/ReaderVoiceMobile/voicedesign-app/src/main/cpp/voicedesign_jni.cpp'
s = io.open(SRC, encoding='utf-8').read()
# 1) 去掉 JNI 头与宏
s = s.replace('#include <jni.h>\n#include <android/log.h>\n', '#include <cstdio>\n')
s = s.replace('#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, "VDS", __VA_ARGS__)\n', '#define LOGI(...) do { printf(__VA_ARGS__); printf("\\n"); fflush(stdout); } while (0)\n')
s = s.replace('#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, "VDS", __VA_ARGS__)\n', '#define LOGE(...) do { fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\\n"); } while (0)\n')
# 2) 把 nativeLoad 主体改成 main 里的 loadAll()
i0 = s.find('extern "C" JNIEXPORT jlong JNICALL')(0) if False else s.find('extern "C" JNIEXPORT jlong JNICALL')
i1 = s.find('extern "C" JNIEXPORT jstring JNICALL Java_com_voicedesign_app_VoiceDesignEngine_nativeRun')
i2 = s.rfind('}', i1)
head = s[:i0]
body_load = s[s.find('extern "C" JNIEXPORT jboolean JNICALL'):s.find('extern "C" JNIEXPORT jlong JNICALL')]
io.open('E:/AndroidStudioProjects/qwen3tts-mnn/tmp_head.cpp','w',encoding='utf-8').write(head)
print('head len', len(head), 'jni_start', i0, 'run_start', i1)
print('HEAD_TAIL:'); print(head[-400:])