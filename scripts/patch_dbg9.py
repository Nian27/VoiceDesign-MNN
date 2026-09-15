import io
p = r'E:/AndroidStudioProjects/qwen3tts-mnn/scripts/export/vd_cpr_dbg9.py'
s = io.open(p, encoding='utf-8').read()
old = "        print('cos(model attn_out, mine attn_out) = %.7f' % cos(model_ao.numpy(), mine_ao.numpy()))"
new = ("        print('cos(model attn_out, o_proj(mine attn_out)) = %.7f' % cos(model_ao.numpy(), at.o_proj(mine_ao).numpy()))\n"
       "        h_model = xp + model_ao\n"
       "        h_mine = xp + at.o_proj(mine_ao)\n"
       "        print('cos(h_pre_mlp model, mine) = %.7f' % cos(h_model.numpy(), h_mine.numpy()))\n"
       "        print('cos(v_all model, mine) = %.7f' % cos(kvv.repeat_interleave(2,dim=1).numpy(), kvv.repeat_interleave(2,dim=1).numpy()))")
n = s.count(old); s = s.replace(old, new)
io.open(p, 'w', encoding='utf-8').write(s)
print('patched', n)