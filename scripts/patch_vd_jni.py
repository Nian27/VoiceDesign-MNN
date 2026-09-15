import io
p = r'E:/AndroidStudioProjects/ReaderVoiceMobile/app-android/src/main/cpp/voicedesign_jni.cpp'
s = io.open(p, encoding='utf-8').read()
n1 = s.count('struct Session {'); s = s.replace('struct Session {', 'struct VdSession {')
n2 = s.count('new Session()');     s = s.replace('new Session()', 'new VdSession()')
n3 = s.count('reinterpret_cast<Session*>'); s = s.replace('reinterpret_cast<Session*>', 'reinterpret_cast<VdSession*>')
io.open(p,'w',encoding='utf-8').write(s)
print('renamed struct=%d new=%d cast=%d' % (n1,n2,n3))