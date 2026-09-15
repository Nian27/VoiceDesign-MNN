# Verify MNN codepred_ar_fp16.mnn vs torch AR output.
import os, numpy as np, torch, MNN
D = r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/device-smoke'
def cos(a,b):
    a=np.asarray(a,np.float64).ravel(); b=np.asarray(b,np.float64).ravel()
    return float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12))
io = torch.load(D + '/codepred_ar/ar_torch_io.pt', weights_only=False)
ph = io['ph'].numpy().astype(np.float32); c0e = io['c0e'].numpy().astype(np.float32)
yt = io['y'].numpy().astype(np.float32)
print('ref', yt.shape)
p = D + '/codepred_ar_fp16.mnn'
interp = MNN.Interpreter(p)
if os.path.exists(p + '.weight'):
    interp.setExternalFile(p + '.weight'); print('external weight set')
s = interp.createSession()
ins = interp.getSessionInputAll(s)
print('inputs:', [(k, tuple(v.getShape())) for k, v in (ins.items() if hasattr(ins,'items') else [])])
outs = interp.getSessionOutputAll(s)
print('outputs:', [(k, tuple(v.getShape())) for k, v in (outs.items() if hasattr(outs,'items') else [])])
def feed(name, arr):
    t = interp.getSessionInput(s, name)
    ht = MNN.Tensor(tuple(arr.shape), MNN.Halide_Type_Float, arr.ravel(), MNN.Tensor_DimensionType_Caffe)
    t.copyFromHostTensor(ht)
feed('past_hidden', ph); feed('code0_emb', c0e)
interp.runSession(s)
ot = interp.getSessionOutput(s, 'logits15')
osh = [d if d > 0 else 1 for d in ot.getShape()]
ho = MNN.Tensor(osh, MNN.Halide_Type_Float, MNN.Tensor_DimensionType_Caffe)
ot.copyToHostTensor(ho)
ym = np.asarray(ho.getNumpyData()).reshape(osh).astype(np.float32)
print('mnn', ym.shape)
print('MNN vs torch AR: cos=%.8f  maxabsdiff=%.6f' % (cos(ym, yt), float(np.abs(ym - yt).max())))
# per-step cos
for g in range(15):
    print('  step%2d cos=%.7f' % (g, cos(ym[0,g], yt[0,g])))
# greedy codes from both
cm = [int(np.argmax(ym[0,g])) for g in range(15)]
ct = [int(np.argmax(yt[0,g])) for g in range(15)]
print('codes mnn', cm)
print('codes tor', ct)
print('codes exact =', cm == ct)
print('MNN_AR_VERIFY_DONE')
