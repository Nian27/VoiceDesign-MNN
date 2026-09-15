import numpy as np, os, MNN
D = r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/device-smoke'
OUT = D + '/vd_p2'
it = MNN.Interpreter(D + '/codec_head_fp16.mnn'); s = it.createSession()
def code0(h):
    t = it.getSessionInput(s, 'hidden_states')
    ht = MNN.Tensor((1,1,h.size), MNN.Halide_Type_Float, h.astype(np.float32).ravel(), MNN.Tensor_DimensionType_Caffe)
    t.copyFromHostTensor(ht)
    it.runSession(s)
    o = it.getSessionOutput(s, 'logits')
    sh = [d if d>0 else 1 for d in o.getShape()]
    ho = MNN.Tensor(sh, MNN.Halide_Type_Float, MNN.Tensor_DimensionType_Caffe)
    o.copyToHostTensor(ho)
    lg = np.asarray(ho.getNumpyData()).reshape(sh).ravel()
    masked = lg.copy()
    for i in range(2048, 3072):
        if i != 2150: masked[i] = -1e30
    return int(np.argmax(masked)), lg
def cos(a,b):
    a=np.asarray(a,np.float64).ravel(); b=np.asarray(b,np.float64).ravel()
    return float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12))
ref = np.fromfile(OUT + '/ref28_hidden_states.raw', dtype=np.float32)
cpu = np.fromfile(OUT + '/g1cpu/hidden_states.raw', dtype=np.float32)
htp = np.fromfile(OUT + '/g1htp/hidden_states.raw', dtype=np.float32)
c_ref, l_ref = code0(ref)
c_cpu, l_cpu = code0(cpu)
c_htp, l_htp = code0(htp)
print('code0  torch-ref=%d   device-CPU=%d   device-HTP=%d' % (c_ref, c_cpu, c_htp))
print('MATCH  CPU==ref -> %s   HTP==ref -> %s' % (c_cpu==c_ref, c_htp==c_ref))
print('logits cos  CPU vs ref = %.8f   HTP vs ref = %.8f' % (cos(l_cpu,l_ref), cos(l_htp,l_ref)))
top = np.argsort(-l_htp[l_htp > -1e29])[:5]
print('HTP top5 code0 =', top.tolist(), ' logits', np.round(l_htp[top],4).tolist())
topc = np.argsort(-l_cpu[l_cpu > -1e29])[:5]
print('CPU top5 code0 =', topc.tolist(), ' logits', np.round(l_cpu[topc],4).tolist())
