import io
# 1) manifest: VoiceDesignActivity exported=true (demo/debug entry)
m = r'E:/AndroidStudioProjects/ReaderVoiceMobile/app-android/src/main/AndroidManifest.xml'
s = io.open(m, encoding='utf-8').read()
old = '<activity android:name=".VoiceDesignActivity" android:exported="false" />'
new = '<activity android:name=".VoiceDesignActivity" android:exported="true" />'
n1 = s.count(old); s = s.replace(old, new)
io.open(m,'w',encoding='utf-8').write(s)
# 2) Activity: 用 App 内部目录（外部目录 adb 建的 shell 属主, App 无权访问）
p = r'E:/AndroidStudioProjects/ReaderVoiceMobile/app-android/src/main/java/com/readervoice/app/VoiceDesignActivity.kt'
t = io.open(p, encoding='utf-8').read()
old2 = 'val d = File(getExternalFilesDir(null), "voicedesign")'
new2 = 'val d = File(filesDir, "voicedesign")'
n2 = t.count(old2); t = t.replace(old2, new2)
io.open(p,'w',encoding='utf-8').write(t)
print('manifest exported=%d  internalDir=%d' % (n1, n2))