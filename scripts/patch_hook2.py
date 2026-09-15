import io
p = r'E:/AndroidStudioProjects/qwen3tts-mnn/scripts/export/vd_cpr_hook2.py'
s = io.open(p, encoding='utf-8').read()
bad = "hh2 = at.register_forward_hook(lambda m,i,o: cap.__setitem__('attn_in', i[0].detach().clone()))\n"
s = s.replace(bad, '')
s = s.replace("    hn_model = cap['attn_in']          # (1,2,1024)", "    hn_model = layer.input_layernorm(xp) if False else layer.input_layernorm(xp)")
s = s.replace('hh2.remove()', '')
io.open(p, 'w', encoding='utf-8').write(s)
print('hook2 removed; ok=', 'hh2' not in s)