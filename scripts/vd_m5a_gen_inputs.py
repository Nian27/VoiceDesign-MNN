# VD-M5A: generate frozen raw inputs (emb/pos/cp/kvk/kvv) + PC reference hidden
# for steps [0,1,5,10,19] from p1_traj, in the EXACT feed order of the new GraphB contract.
# Also writes a manifest with per-file CRC32 + sizes + PC reference hidden CRC32.
import numpy as np, os, zlib, json
D = r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/device-smoke'
T = D + '/p1_traj'
OUT = D + '/vd_m5a_inputs'
os.makedirs(OUT, exist_ok=True)
STEPS = [0, 1, 5, 10, 19]
MAX_KV = 768
manifest = {'steps': {}}
for s in STEPS:
    z = np.load(T + '/step' + str(s) + '.npz')
    emb = z['emb_in'].astype(np.float32)                 # (1,1,2048)
    pos = z['pos'].astype(np.float32)                    # (3,1,1)
    cp = np.array([int(z['cp'][0])], dtype=np.int32)     # (1,)
    kvk_cur = z['kvk'].astype(np.float32)                # (28,1,8,n,128)
    kvv_cur = z['kvv'].astype(np.float32)
    hidden_ref = z['hidden'].astype(np.float32)          # (1,1,2048)
    n = kvk_cur.shape[3]
    # running buffer: fill slots 0..n-2 from authority (model writes slot cp)
    kvk = np.zeros((28,1,8,MAX_KV,128), dtype=np.float32)
    kvv = np.zeros((28,1,8,MAX_KV,128), dtype=np.float32)
    kvk[:,:,:,:n-1,:] = kvk_cur[:,:,:,:n-1,:]
    kvv[:,:,:,:n-1,:] = kvv_cur[:,:,:,:n-1,:]
    files = {'emb': emb, 'pos': pos, 'cp': cp, 'kvk': kvk, 'kvv': kvv}
    rec = {'step': s, 'cp': int(cp[0]), 'kv_len_in': int(n)}
    for name, arr in files.items():
        p = OUT + '/step%d_%s.raw' % (s, name)
        arr.tofile(p)
        crc = zlib.crc32(arr.tobytes()) & 0xffffffff
        rec[name] = {'file': os.path.basename(p), 'bytes': os.path.getsize(p), 'crc32': '%08x' % crc, 'shape': list(arr.shape), 'dtype': str(arr.dtype)}
    # PC reference hidden
    hp = OUT + '/step%d_hidden_ref.raw' % s
    hidden_ref.tofile(hp)
    rec['hidden_ref'] = {'file': os.path.basename(hp), 'bytes': os.path.getsize(hp), 'crc32': '%08x' % (zlib.crc32(hidden_ref.tobytes()) & 0xffffffff), 'shape': list(hidden_ref.shape)}
    manifest['steps'][str(s)] = rec
with open(OUT + '/manifest.json', 'w') as f:
    json.dump(manifest, f, indent=2)
print('VD_M5A_INPUTS_GENERATED')
for s in STEPS:
    r = manifest['steps'][str(s)]
    print('step%d cp=%d kv_in=%d emb_crc=%s hidden_ref_crc=%s' % (s, r['cp'], r['kv_len_in'], r['emb']['crc32'], r['hidden_ref']['crc32']))
