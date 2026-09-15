import os, io, numpy as np, torch
os.environ['HF_HUB_OFFLINE']='1'
MDL = r'E:/AndroidStudioProjects/qwen3tts-mnn/models/qwen3-tts/Qwen3-TTS-12Hz-1.7B-VoiceDesign'
OUT = r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/device-smoke/app_models'
from qwen_tts import Qwen3TTSModel
gw = Qwen3TTSModel.from_pretrained(MDL, dtype=torch.float32, device_map='cpu')
ce = gw.model.talker.model.codec_embedding.weight
print('talker codec_embedding', tuple(ce.shape))
ce.detach().float().contiguous().numpy().astype(np.float32).tofile(OUT + '/talker_codec_emb.f32')
print('bytes', os.path.getsize(OUT + '/talker_codec_emb.f32'))

# --- 更新 JNI: 用 instruct/text 在设备端构造 prompt ---
p = r'E:/AndroidStudioProjects/ReaderVoiceMobile/voicedesign-app/src/main/cpp/voicedesign_jni.cpp'
s = io.open(p, encoding='utf-8').read()
if '#include "vd_text.h"' not in s:
    s = s.replace('#include <memory>', '#include <memory>\n#include "vd_text.h"', 1)
s = s.replace('const float* lmW = nullptr;', 'vdt::Tokenizer tok;\n    vdt::TextTables txtTab;\n    std::vector<float> codecEmb;\n    const float* lmW = nullptr;', 1)
old_sig = 'JNIEnv* env, jobject, jlong p, jstring jdir, jstring jout, jint maxFrames) {'
new_sig = 'JNIEnv* env, jobject, jlong p, jstring jdir, jstring jout, jint maxFrames, jstring jinstruct, jstring jtext, jint jlang) {'
n1 = s.count(old_sig); s = s.replace(old_sig, new_sig)
old_rd = '    FILE* pf=fopen((d+"prompt_emb.f32").c_str(), "rb");'
new_rd = ('    // 设备端构造 prompt: tokenize -> text_embedding+projection -> 拼装\n'
          '    std::string instruct, text;\n'
          '    if (jinstruct) { const char* c = env->GetStringUTFChars(jinstruct, nullptr); instruct = c; env->ReleaseStringUTFChars(jinstruct, c); }\n'
          '    if (jtext) { const char* c = env->GetStringUTFChars(jtext, nullptr); text = c; env->ReleaseStringUTFChars(jtext, c); }\n'
          '    std::vector<float> peBuf;\n'
          '    int L = 0;\n'
          '    if (!instruct.empty() || !text.empty()) {\n'
          '        auto ii = s->tok.encode("<|im_start|>user\\n" + instruct + "<|im_end|>\\n");\n'
          '        auto ai = s->tok.encode("<|im_start|>assistant\\n" + text + "<|im_end|>\\n<|im_start|>assistant\\n");\n'
          '        LOGI("tokenized instruct=%zu assistant=%zu", ii.size(), ai.size());\n'
          '        peBuf.assign((size_t)512 * 2048, 0.f);\n'
          '        L = vdt::buildPrompt(s->tok, s->txtTab, ii, ai, jlang, s->codecEmb.data(), (int)(s->codecEmb.size() / 2048), peBuf.data(), 512);\n'
          '        if (L <= 0) return env->NewStringUTF("ERR prompt build failed");\n'
          '        LOGI("prompt tokens=%d", L);\n'
          '    }\n'
          '    FILE* pf = peBuf.empty() ? fopen((d+"prompt_emb.f32").c_str(), "rb") : nullptr;'
n2 = s.count(old_rd); s = s.replace(old_rd, new_rd)
s = s.replace('    std::vector<float> pe(L*2048);\n    if (fread(pe.data(),4,pe.size(),pf)!=pe.size()) { fclose(pf); return env->NewStringUTF("ERR prompt_emb short"); }\n    fclose(pf);',
              '    std::vector<float> pe;\n    if (pf) { pe.resize((size_t)L*2048); if (fread(pe.data(),4,pe.size(),pf)!=pe.size()) { fclose(pf); return env->NewStringUTF("ERR prompt_emb short"); } fclose(pf); }\n    else { pe.swap(peBuf); }')
io.open(p,'w',encoding='utf-8').write(s)
print('jni patched: sig=%d read=%d' % (n1, n2))