import io
p = r'E:/AndroidStudioProjects/qwen3tts-mnn/scripts/export/vd_cpr_dbg4.py'
s = io.open(p, encoding='utf-8').read()
# 参考里补上正确的 V 缓冲
old = "    k_all = kvk_t.repeat_interleave(2, dim=1)"
new = ("    kvv_t = torch.zeros(1,NKV,CAP,HD)\n"
       "    kvv_t = kvv_t*(1.0-torch.from_numpy(sm0)-torch.from_numpy(sm1)) + v[:,:,0:1,:]*torch.from_numpy(sm0) + v[:,:,1:2,:]*torch.from_numpy(sm1)\n"
       "    print('cos(torch kvv, onnx kvv?)  (kvv not exported)')\n"
       "    k_all = kvk_t.repeat_interleave(2, dim=1)")
n1 = s.count(old); s = s.replace(old, new)
# ao 用正确的 V
old2 = "    ao = torch.matmul(pr, kvk_t.repeat_interleave(2,dim=1)).reshape(*hs,-1).contiguous()"
new2 = "    ao = torch.matmul(pr, kvv_t.repeat_interleave(2,dim=1)).reshape(*hs,-1).contiguous()"
n2 = s.count(old2); s = s.replace(old2, new2)
# q/k 比较改成 pre-rope（与导出对齐）
old3 = "    print('cos(torch qr, onnx q) = %.7f' % cos(qr2.numpy(), ox['q']))"
new3 = ("    print('cos(torch q(pre-rope), onnx q) = %.7f' % cos(q.numpy(), ox['q']))\n"
       "    print('cos(torch k(pre-rope), onnx k) = %.7f' % cos(k.numpy(), ox['k']))")
n3 = s.count(old3); s = s.replace(old3, new3)
old4 = "    print('cos(torch kr, onnx k) = %.7f' % cos(kr2.numpy(), ox['k']))"
s = s.replace(old4, "    print('cos(torch kr(post-rope), onnx kvk via mask) = %.7f' % cos(kvk_t.numpy(), ox['kvk']))")
io.open(p, 'w', encoding='utf-8').write(s)
print('ref fixed: kvv=%d ao=%d qpre=%d' % (n1,n2,n3))