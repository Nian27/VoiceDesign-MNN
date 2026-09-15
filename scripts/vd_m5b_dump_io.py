import MNN, os
D = r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/device-smoke'
for name in ['grapha_prefill_static_fp16.mnn', 'codec_head_fp16.mnn', 'tokenizer_decoder_static_t300.mnn']:
    p = D + '/' + name
    if not os.path.exists(p):
        print('--- %s MISSING ---' % name); continue
    print('--- %s (%.1f MB) ---' % (name, os.path.getsize(p)/1e6))
    it = MNN.Interpreter(p)
    w = p + '.weight'
    if os.path.exists(w):
        it.setExternalFile(w); print('   has external weight', os.path.getsize(w))
    s = it.createSession()
    ins = it.getSessionInputAll(s); outs = it.getSessionOutputAll(s)
    for k, v in (ins.items() if hasattr(ins,'items') else []):
        print('   IN  %-16s %s' % (k, tuple(v.getShape())))
    for k, v in (outs.items() if hasattr(outs,'items') else []):
        print('   OUT %-16s %s' % (k, tuple(v.getShape())))
print('DUMP_DONE')
