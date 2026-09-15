import io
# 1) JNI: 接收 nativeLibraryDir 并设置 ADSP_LIBRARY_PATH
p = r'E:/AndroidStudioProjects/ReaderVoiceMobile/app-android/src/main/cpp/voicedesign_jni.cpp'
s = io.open(p, encoding='utf-8').read()
old = 'extern "C" JNIEXPORT jboolean JNICALL Java_com_readervoice_app_VoiceDesignEngine_nativeLoad(\n        JNIEnv* env, jobject, jlong p, jstring jdir, jstring jbackend, jint maxPos) {'
new = 'extern "C" JNIEXPORT jboolean JNICALL Java_com_readervoice_app_VoiceDesignEngine_nativeLoad(\n        JNIEnv* env, jobject, jlong p, jstring jdir, jstring jbackend, jint maxPos, jstring jnativedir) {'
n1 = s.count(old); s = s.replace(old, new)
old2 = '    env->ReleaseStringUTFChars(jdir, dir); env->ReleaseStringUTFChars(jbackend, be);'
new2 = ('    std::string nd;\n'
        '    if (jnativedir) { const char* ndc = env->GetStringUTFChars(jnativedir, nullptr); nd = ndc; env->ReleaseStringUTFChars(jnativedir, ndc); }\n'
        '    env->ReleaseStringUTFChars(jdir, dir); env->ReleaseStringUTFChars(jbackend, be);\n'
        '    // DSP skel 必须能被 adsp 找到；命令行为此显式设 ADSP_LIBRARY_PATH，App 里同样需要\n'
        '    if (!nd.empty()) { setenv("ADSP_LIBRARY_PATH", nd.c_str(), 1); LOGI("ADSP_LIBRARY_PATH=%s", nd.c_str()); }')
n2 = s.count(old2); s = s.replace(old2, new2)
io.open(p,'w',encoding='utf-8').write(s)
# 2) Kotlin: 传 nativeLibraryDir
k = r'E:/AndroidStudioProjects/ReaderVoiceMobile/app-android/src/main/java/com/readervoice/app/VoiceDesignEngine.kt'
t = io.open(k, encoding='utf-8').read()
t = t.replace('fun load(modelDir: String, backend: String, maxPos: Int = 384): Boolean {', 'fun load(modelDir: String, backend: String, nativeLibDir: String = "", maxPos: Int = 384): Boolean {')
t = t.replace('return nativeLoad(nativePtr, modelDir, backend, maxPos)', 'return nativeLoad(nativePtr, modelDir, backend, maxPos, nativeLibDir)')
t = t.replace('private external fun nativeLoad(ptr: Long, modelDir: String, backend: String, maxPos: Int): Boolean', 'private external fun nativeLoad(ptr: Long, modelDir: String, backend: String, maxPos: Int, nativeLibDir: String): Boolean')
io.open(k,'w',encoding='utf-8').write(t)
# 3) Activity: 传 applicationInfo.nativeLibraryDir
a = r'E:/AndroidStudioProjects/ReaderVoiceMobile/app-android/src/main/java/com/readervoice/app/VoiceDesignActivity.kt'
u = io.open(a, encoding='utf-8').read()
u = u.replace('if (!eng.load(dir, backend))', 'if (!eng.load(dir, backend, applicationInfo.nativeLibraryDir))')
io.open(a,'w',encoding='utf-8').write(u)
print('jni=%d/%d  kotlin/activity patched' % (n1,n2))