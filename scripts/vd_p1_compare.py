import numpy as np, os
D = r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/device-smoke/vd_p1'
hexd = D + '/hex'
tl = np.load(D + '/torch_layers_step0.npy')      # (28,2048) torch reference (same input as HTP run)
tf = np.load(D + '/torch_final_step0.npy')       # (2048,)
def cos(a,b):
    a=np.asarray(a,np.float64).ravel(); b=np.asarray(b,np.float64).ravel()
    return float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12))
def rd16(p):
    return np.fromfile(p, dtype=np.float16).astype(np.float32)
print('=== layer boundary: HTP dump step 2+3i vs torch layer i ===')
bad = None
for i in range(28):
    s = 2 + 3*i
    p = '%s/hidden_%04d_out.bin' % (hexd, s)
    if not os.path.exists(p):
        print('layer %2d step %4d MISSING' % (i, s)); continue
    h = rd16(p)
    if h.size != 2048:
        print('layer %2d step %4d size %d unexpected' % (i, s, h.size)); continue
    c = cos(h, tl[i])
    mx = float(np.abs(h.astype(np.float64)-tl[i].astype(np.float64)).max())
    flag = ''
    if c < 0.999 and bad is None:
        bad = (i, s, c); flag = '   <== FIRST DIVERGENCE'
    print('layer %2d step %4d  cos=%.6f  maxdiff=%.5f  htpMax=%.3f torchMax=%.3f%s' % (
        i, s, c, mx, float(np.abs(h).max()), float(np.abs(tl[i]).max()), flag))
print()
print('FIRST_DIVERGENCE_LAYER =', bad)
# also the very first op (step0) should be layer0's input_layernorm input? compare with emb
emb = np.fromfile(r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/device-smoke/vd_m5a_inputs/step0_emb.raw', dtype=np.float32).ravel()
h0 = rd16(hexd + '/hidden_0000_out.bin')
print('step0 out vs emb        cos=%.6f' % cos(h0, emb))
h1 = rd16(hexd + '/hidden_0001_out.bin')
print('step1 out vs emb        cos=%.6f' % cos(h1, emb))
print('P1_CMP_DONE')
