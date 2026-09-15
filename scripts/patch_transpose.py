import io
p = r'E:/AndroidStudioProjects/qwen3tts-mnn/scripts/export/vd_cpr1_export2.py'
s = io.open(p, encoding='utf-8').read()
old = 'ao = torch.matmul(probs, v_all).reshape(*hs,-1).contiguous()'
new = 'ao = torch.matmul(probs, v_all).transpose(1,2).reshape(*hs,-1).contiguous()'
n = s.count(old)
s = s.replace(old, new)
io.open(p, 'w', encoding='utf-8').write(s)
print('transpose fix applied to %d sites' % n)