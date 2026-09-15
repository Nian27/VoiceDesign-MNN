import io
p = r'E:/AndroidStudioProjects/qwen3tts-mnn/scripts/export/vd_cpr2_verify.py'
s = io.open(p, encoding='utf-8').read()
n = s.count("mk(D + '/codec_emb_fp16.mnn')")
s = s.replace("mk(D + '/codec_emb_fp16.mnn')", "mk(D + '/vd_m5b/codec_emb_fp16.mnn')")
io.open(p, 'w', encoding='utf-8').write(s)
print('patched', n)