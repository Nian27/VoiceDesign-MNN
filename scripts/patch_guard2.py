import io
p = r'E:/AndroidStudioProjects/ReaderVoiceMobile/voicedesign-app/src/main/cpp/voicedesign_jni.cpp'
s = io.open(p, encoding='utf-8').read()
n1 = s.count('if (!o[i]) { LOGI("NULL_VARP'); 
s = s.replace('if (!o[i]) { LOGI("NULL_VARP at %s[%zu]", tag, i); return false; }', 'if (o[i].get() == nullptr) { LOGI("NULL_VARP at %s[%zu]", tag, i); return false; }')
old_g = 'if (!outOk(o, 3, "graphb")) return env->NewStringUTF("ERR BACKEND_STOP_EXECUTE: graphb");'
new_g = 'if (!outOk(o, 3, "graphb")) return std::vector<float>();'
n2 = s.count(old_g); s = s.replace(old_g, new_g)
old_p = '        if (cp % 5 == 0 || cp == L-1) LOGI("prefill %d/%d", cp+1, L);'
new_p = '        if (hidden.size() != 2048) return env->NewStringUTF("ERR BACKEND_STOP_EXECUTE: graphb_prefill");\n' + old_p
n3 = s.count(old_p); s = s.replace(old_p, new_p)
old_d = '        hidden = gbStep(femb.data(), cp);'
new_d = '        hidden = gbStep(femb.data(), cp);\n        if (hidden.size() != 2048) return env->NewStringUTF("ERR BACKEND_STOP_EXECUTE: graphb_decode");'
n4 = s.count(old_d); s = s.replace(old_d, new_d)
io.open(p,'w',encoding='utf-8').write(s)
print('varp_null=%d graphb_guard=%d prefill_chk=%d decode_chk=%d' % (n1,n2,n3,n4))