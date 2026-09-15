import io
p = r'E:/AndroidStudioProjects/qwen3tts-mnn/scripts/export/vd_cpr2_verify.py'
s = io.open(p, encoding='utf-8').read()
# 1) 去掉错误的 transpose 启发式
old = "if c.shape[2] != CAP: c = c.transpose(0,1); si = si.transpose(0,1)"
n1 = s.count(old); s = s.replace(old, '')
# 2) 强制确定性
anchor = 'from qwen_tts import Qwen3TTSModel'
n2 = s.count(anchor)
s = s.replace(anchor, 'torch.set_num_threads(1)\ntry:\n    torch.use_deterministic_algorithms(True, warn_only=True)\nexcept Exception:\n    pass\n' + anchor, 1)
io.open(p, 'w', encoding='utf-8').write(s)
print('transpose-hack removed=%d  determinism added=%d' % (n1, n2))