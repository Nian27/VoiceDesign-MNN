import io
p = r'E:/AndroidStudioProjects/ReaderVoiceMobile/app-android/src/main/cpp/voicedesign_jni.cpp'
s = io.open(p, encoding='utf-8').read()
# normalize then global rename (file never references MNN::Session explicitly)
s = s.replace('VdSession', 'Session')
n = s.count('Session')
s = s.replace('Session', 'VdSession')
io.open(p,'w',encoding='utf-8').write(s)
print('global rename, occurrences=%d ; MNN::Session present=%s' % (n, 'MNN::Session' in s))