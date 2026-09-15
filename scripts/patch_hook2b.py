import io
p = r'E:/AndroidStudioProjects/qwen3tts-mnn/scripts/export/vd_cpr_hook2.py'
s = io.open(p, encoding='utf-8').read()
s = s.replace("    hn_model = layer.input_layernorm(xp) if False else layer.input_layernorm(xp)", "    hn_model = xp")
s = s.replace("    print('attn input (pre input_layernorm?) shape', tuple(hn_model.shape))", "")
s = s.replace("    print('cos(attn_in, xp) = %.7f' % cos(hn_model.numpy(), xp.numpy()))", "")
s = s.replace("    print('cos(attn_in, input_layernorm(xp)) = %.7f' % cos(hn_model.numpy(), hn.numpy()))", "")
io.open(p, 'w', encoding='utf-8').write(s)
print('cleaned')