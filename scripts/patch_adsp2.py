import io
base = r'E:/AndroidStudioProjects/ReaderVoiceMobile/voicedesign-app/src/main/java/com/voicedesign/app'
# 1) Engine: 类加载前设置 ADSP (静态 init 里做, 保证早于 System.loadLibrary)
k = base + '/VoiceDesignEngine.kt'
s = io.open(k, encoding='utf-8').read()
old = 'class VoiceDesignEngine {'
new = ('class VoiceDesignEngine {\n'
       '\n'
       '    companion object {\n'
        /**\n'
         * 关键：ADSP_LIBRARY_PATH 必须包含厂商 rfsa 路径，且必须在任何 MNN 库被 dlopen 之前设置。\n'
         * 只设 App 自己的 lib 目录会把 /vendor/lib/rfsa/adsp 覆盖掉 -> DSP 加载不到基础库 -> 挂死。\n'
         * 参考 MNN source/backend/hexagon/README.md 与官方 MnnLlmChat 的 QnnModule.kt。\n'
         */\n'
       '        @Volatile private var adspReady = false\n'
       '        fun prepareAdsp(nativeLibDir: String) {\n'
       '            if (adspReady) return\n'
       '            val dirs = listOf(nativeLibDir, "/vendor/lib/rfsa/adsp", "/system/lib/rfsa/adsp")\n'
       '                .filter { it.isNotBlank() }.joinToString(";")\n'
       '            try { android.system.Os.setenv("ADSP_LIBRARY_PATH", dirs, true) } catch (_: Throwable) {}\n'
       '            adspReady = true\n'
       '        }\n'
       '    }')
n1 = s.count(old); s = s.replace(old, new, 1)
io.open(k,'w',encoding='utf-8').write(s)
print('engine adsp helper:', n1)
# 2) Service: 在 new VoiceDesignEngine() 之前调用 prepareAdsp
v = base + '/VoiceDesignService.kt'
t = io.open(v, encoding='utf-8').read()
o2 = '            val eng = VoiceDesignEngine()'
n2 = t.count(o2)
t = t.replace(o2, '            VoiceDesignEngine.prepareAdsp(applicationInfo.nativeLibraryDir)\n            append(this, "ADSP_LIBRARY_PATH=" + (System.getenv("ADSP_LIBRARY_PATH") ?: ""))\n' + o2, 1)
io.open(v,'w',encoding='utf-8').write(t)
print('service prepareAdsp:', n2)
# 3) JNI: setenv 也补厂商路径
j = r'E:/AndroidStudioProjects/ReaderVoiceMobile/voicedesign-app/src/main/cpp/voicedesign_jni.cpp'
c = io.open(j, encoding='utf-8').read()
oj = 'if (!nd.empty()) { setenv("ADSP_LIBRARY_PATH", nd.c_str(), 1); LOGI("ADSP_LIBRARY_PATH=%s", nd.c_str()); }'
nj = ('if (!nd.empty()) { std::string ap = nd + ";/vendor/lib/rfsa/adsp;/system/lib/rfsa/adsp"; setenv("ADSP_LIBRARY_PATH", ap.c_str(), 1); LOGI("ADSP_LIBRARY_PATH=%s", ap.c_str()); }')
n3 = c.count(oj); c = c.replace(oj, nj)
io.open(j,'w',encoding='utf-8').write(c)
print('jni adsp vendor paths:', n3)