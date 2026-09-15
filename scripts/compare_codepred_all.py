import numpy as np
D = r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/device-smoke'
def cmp(dev_path, ref_path, tag):
    a = np.fromfile(dev_path, dtype=np.float32).astype(np.float64)
    b = np.load(ref_path).ravel().astype(np.float64)
    n = min(a.size, b.size); a=a[:n]; b=b[:n]
    c = float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-9))
    m = float(np.max(np.abs(a-b))); rel = float(np.linalg.norm(a-b)/np.linalg.norm(a))
    ok = 'PASS' if c >= 0.999 else 'FAIL'
    print(tag, ok, 'cosine', round(c,6), 'maxdiff', round(m,5), 'rel_l2', round(rel,6))
allok = True
for g in range(15):
    dp = D + '/cp_g' + str(g) + '_dev_f32.raw'
    rp = D + '/codepred_ref_g' + str(g) + '.npy'
    a = np.fromfile(dp, dtype=np.float32)
    dmax = float(np.abs(a).max())
    print('g' + str(g), 'dev_maxabs', round(dmax,4))
    cmp(dp, rp, 'g' + str(g) + ':')
print('CODEPRED_DEVICE_ALL_DONE')