import io
m = r'E:/AndroidStudioProjects/ReaderVoiceMobile/app-android/src/main/AndroidManifest.xml'
s = io.open(m, encoding='utf-8').read()
old = '<activity android:name=".VoiceDesignActivity" android:exported="true" />'
new = ('<activity\n'
       '            android:name=".VoiceDesignActivity"\n'
       '            android:exported="true"\n'
       '            android:label="VoiceDesign">\n'
       '            <intent-filter>\n'
       '                <action android:name="android.intent.action.MAIN" />\n'
       '                <category android:name="android.intent.category.LAUNCHER" />\n'
       '            </intent-filter>\n'
       '        </activity>')
n = s.count(old); s = s.replace(old, new)
io.open(m,'w',encoding='utf-8').write(s)
print('launcher entry added:', n)