import io, os, shutil
SRC = r'E:/AndroidStudioProjects/ReaderVoiceMobile/app-android'
DST = r'E:/AndroidStudioProjects/ReaderVoiceMobile/voicedesign-app'
os.makedirs(os.path.join(DST, 'src/main/cpp'), exist_ok=True)
os.makedirs(os.path.join(DST, 'src/main/java/com/voicedesign/app'), exist_ok=True)
os.makedirs(os.path.join(DST, 'src/main/jniLibs/arm64-v8a'), exist_ok=True)
# 1) JNI: copy + rename package in exported symbols
j = io.open(os.path.join(SRC,'src/main/cpp/voicedesign_jni.cpp'), encoding='utf-8').read()
j = j.replace('com_readervoice_app_VoiceDesignEngine', 'com_voicedesign_app_VoiceDesignEngine')
j = j.replace('"VD"', '"VDS"')
io.open(os.path.join(DST,'src/main/cpp/voicedesign_jni.cpp'),'w',encoding='utf-8').write(j)
print('jni copied, symbols renamed:', j.count('com_voicedesign_app'))
# 2) Kotlin: copy + rename package
for fn in ['VoiceDesignEngine.kt','VoiceDesignActivity.kt','VoiceDesignService.kt']:
    s = io.open(os.path.join(SRC,'src/main/java/com/readervoice/app',fn), encoding='utf-8').read()
    s = s.replace('package com.readervoice.app', 'package com.voicedesign.app')
    io.open(os.path.join(DST,'src/main/java/com/voicedesign/app',fn),'w',encoding='utf-8').write(s)
    print('kt copied:', fn)
# 3) jniLibs: 只需 MNN + Express + htpops + skel + libc++
need = ['libMNN.so','libMNN_Express.so','libMNN_htpops.so','libMNN_htpops_skel.so','libc++_shared.so']
for f in need:
    src = os.path.join(SRC,'src/main/jniLibs/arm64-v8a',f)
    if os.path.exists(src):
        shutil.copy2(src, os.path.join(DST,'src/main/jniLibs/arm64-v8a',f)); print('so copied:', f)
    else: print('MISSING so:', f)
print('MODULE_SKELETON_DONE')