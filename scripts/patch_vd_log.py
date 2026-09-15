import io
p = r'E:/AndroidStudioProjects/ReaderVoiceMobile/app-android/src/main/java/com/readervoice/app/VoiceDesignActivity.kt'
s = io.open(p, encoding='utf-8').read()
# say() 同时写文件
old = '    private fun say(s: String) = ui.post { log.append(s + "\\n") }'
new = ('    private var logFile: File? = null\n'
       '    private fun say(s: String) {\n'
       '        ui.post { log.append(s + "\\n") }\n'
       '        try { logFile?.appendText(s + "\\n") } catch (_: Throwable) {}\n'
       '    }')
n1 = s.count(old); s = s.replace(old, new)
# runChain 开头设置日志文件
old2 = '        val out = File(dir, "reference_" + backend + "_" + stamp + ".wav")'
new2 = ('        val out = File(dir, "reference_" + backend + "_" + stamp + ".wav")\n'
        '        logFile = File(dir, "vd_run.log"); try { logFile!!.writeText("") } catch (_: Throwable) {}')
n2 = s.count(old2); s = s.replace(old2, new2)
io.open(p,'w',encoding='utf-8').write(s)
print('file-log patch say=%d runChain=%d' % (n1,n2))