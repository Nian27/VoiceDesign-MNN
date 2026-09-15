import io
p = r'E:/AndroidStudioProjects/ReaderVoiceMobile/voicedesign-app/src/main/cpp/voicedesign_jni.cpp'
s = io.open(p, encoding='utf-8').read()
# 插入防御 helper
anchor = 'static std::shared_ptr<Module> loadM('
helper = ('// 后端可能 stop-execute: 此时 onForward 返回空 vector 或含空 VARP。\n'
          '// 必须检查后再解引用，否则 VARP::operator->() 直接 SIGSEGV(fault addr 0x0)。\n'
          'static bool outOk(const std::vector<VARP>& o, size_t need, const char* tag) {\n'
          '    if (o.size() < need) { LOGI("BACKEND_STOP_EXECUTE at %s (out size=%zu need=%zu)", tag, o.size(), need); return false; }\n'
          '    for (size_t i = 0; i < need; i++) { if (!o[i]) { LOGI("NULL_VARP at %s[%zu]", tag, i); return false; } }\n'
          '    return true;\n'
          '}\n')
n0 = s.count(anchor)
s = s.replace(anchor, helper + anchor, 1)
def guard(sig, need, tag):
    global s
    i = s.find(sig)
    if i < 0: return 0
    j = s.find('\n', i) + 1
    ins = '        if (!outOk(%s, %d, "%s")) return env->NewStringUTF("ERR BACKEND_STOP_EXECUTE: %s");\n' % ('o' if ' o =' in sig else ('oc' if 'oc =' in sig else ('oe' if 'oe =' in sig else ('op' if 'op =' in sig else ('os_' if 'os_ =' in sig else ('of' if 'of =' in sig else 'od'))))), need, tag, tag)
    s = s[:j] + ins + s[j:]
    return 1
cnt = 0
cnt += guard('std::vector<VARP> oc = s->mCH->onForward', 1, 'codec_head')
cnt += guard('std::vector<VARP> oe = s->mCE->onForward', 1, 'codec_emb')
cnt += guard('std::vector<VARP> op = s->mCPF->onForward', 3, 'codepred_prefill')
cnt += guard('std::vector<VARP> os_ = s->mCPS->onForward', 3, 'codepred_step')
cnt += guard('std::vector<VARP> of = s->mFE->onForward', 1, 'frame_emb')
cnt += guard('std::vector<VARP> o = s->mGB->onForward', 3, 'graphb')
cnt += guard('std::vector<VARP> od = s->mDEC->onForward', 1, 'decoder')
io.open(p,'w',encoding='utf-8').write(s)
print('helper=%d guards=%d' % (n0, cnt))
print('outOk sites:', s.count('outOk('))