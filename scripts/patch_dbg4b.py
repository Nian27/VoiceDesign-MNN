import io
p = r'E:/AndroidStudioProjects/qwen3tts-mnn/scripts/export/vd_cpr_dbg4.py'
s = io.open(p, encoding='utf-8').read()
bad = "    print('cos(torch kr(post-rope), onnx kvk via mask) = %.7f' % cos(kvk_t.numpy(), ox['kvk']))\n"
n = s.count(bad)
s = s.replace(bad, '')
io.open(p, 'w', encoding='utf-8').write(s)
print('removed misplaced line:', n)