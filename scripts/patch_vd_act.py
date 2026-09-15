import io
p = r'E:/AndroidStudioProjects/ReaderVoiceMobile/app-android/src/main/java/com/readervoice/app/VoiceDesignActivity.kt'
s = io.open(p, encoding='utf-8').read()
anchor = '        setContentView(root)'
add = ('        setContentView(root)\n'
       '        // adb 无头触发：--ez autorun true [--es backend htp]\n'
       '        if (intent?.getBooleanExtra("autorun", false) == true) {\n'
       '            runChain(intent?.getStringExtra("backend") ?: "htp")\n'
       '        }')
n = s.count(anchor)
s = s.replace(anchor, add, 1)
io.open(p,'w',encoding='utf-8').write(s)
print('autorun hook added:', n)