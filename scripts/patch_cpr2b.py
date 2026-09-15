import io
p = r'E:/AndroidStudioProjects/qwen3tts-mnn/scripts/export/vd_cpr2_verify.py'
s = io.open(p, encoding='utf-8').read()
# 1) 删掉占位行
bad = 'hl, kvk, kvv = run1(pre_it, pre_s, \'past_hidden\', past,\n                    [\'hidden_last\'])  # placeholder to init\n'
s = s.replace(bad, '')
assert 'placeholder to init' not in s
# 2) prefill rope 切片
old1 = "('rope_cos', c[:2][None,None] if c.shape[0]>=2 else c[None,None,:2]),"
new1 = "('rope_cos', c[0,:2].reshape(1,1,2,HD)),"
n1 = s.count(old1); s = s.replace(old1, new1)
old2 = "('rope_sin', si[:2][None,None] if si.shape[0]>=2 else si[None,None,:2]),"
new2 = "('rope_sin', si[0,:2].reshape(1,1,2,HD)),"
n2 = s.count(old2); s = s.replace(old2, new2)
# 3) step rope 切片
old3 = 'rc = c[cp_pos].reshape(1,1,1,HD) if c.ndim==3 else c[None,None,cp_pos]'
new3 = 'rc = c[0,cp_pos].reshape(1,1,1,HD)'
n3 = s.count(old3); s = s.replace(old3, new3)
old4 = 'rs = si[cp_pos].reshape(1,1,1,HD) if si.ndim==3 else si[None,None,cp_pos]'
new4 = 'rs = si[0,cp_pos].reshape(1,1,1,HD)'
n4 = s.count(old4); s = s.replace(old4, new4)
io.open(p, 'w', encoding='utf-8').write(s)
print('placeholder removed; rope fixes prefill=%d/%d step=%d/%d' % (n1,n2,n3,n4))