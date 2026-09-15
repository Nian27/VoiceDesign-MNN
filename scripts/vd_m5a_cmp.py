import numpy as np, zlib, sys
D = r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/device-smoke'
tag = sys.argv[1] if len(sys.argv) > 1 else 'step0'
a = np.fromfile(D + '/vd_m5a_out/%s_hidden_dev.raw' % tag, dtype=np.float32)
b = np.fromfile(D + '/vd_m5a_inputs/%s_hidden_ref.raw' % tag, dtype=np.float32)
print('%s: dev %s ref %s' % (tag, a.shape, b.shape))
print('  dev crc32 %08x | ref crc32 %08x' % (zlib.crc32(a.tobytes()) & 0xffffffff, zlib.crc32(b.tobytes()) & 0xffffffff))
aa = a.astype(np.float64); bb = b.astype(np.float64)
c = float(np.dot(aa, bb) / (np.linalg.norm(aa) * np.linalg.norm(bb) + 1e-12))
print('  cos=%.7f  maxabsdiff=%.6f  dev_maxabs=%.4f ref_maxabs=%.4f' % (c, float(np.abs(aa-bb).max()), float(np.abs(aa).max()), float(np.abs(bb).max())))
print('  %s' % ('PASS' if c >= 0.999 else 'FAIL'))
