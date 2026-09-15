import numpy as np, os
D = r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/device-smoke'
G2 = D + '/vd_p2/g2'
# concatenate 20 embs
embs = np.stack([np.fromfile('%s/traj20/emb_%03d.raw' % (D + '/vd_p2', i), dtype=np.float32) for i in range(20)])
embs.tofile(G2 + '/embs20.raw')
print('embs20', embs.shape)
# build 768-slot init KV from prefill (LP=62)
LP = 62; KV = 768
for tag in ['k','v']:
    src = np.fromfile(D + '/vd_m5b/prefill_kv_%s.raw' % tag, dtype=np.float32).reshape(28,1,8,LP,128)
    dst = np.zeros((28,1,8,KV,128), np.float32)
    dst[:,:,:,:LP,:] = src
    dst.tofile(G2 + '/init_kv_%s.raw' % tag)
    print('init_kv_%s' % tag, dst.shape, 'nonzero', int((dst!=0).sum()))
