# P3 ORT verify: StaticCacheDeploy vs authority trajectory (20 steps)
import os, json, numpy as np, onnxruntime as ort
D = r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/device-smoke'
T = r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/device-smoke/p1_traj'
MAX_KV = 768
NUM_STEPS = 20
LOG = open(D + '/p3_ort.log', 'w')
def log(s):
    LOG.write(s + '\n'); LOG.flush()
cos_fn = lambda a,b: float(np.dot(a.ravel(),b.ravel())/(np.linalg.norm(a)*np.linalg.norm(b)+1e-9))
sess = ort.InferenceSession(D + '/graphb_staticcache.onnx', providers=['CPUExecutionProvider'])
print('session ok', flush=True)
all_cos = True
# running KV buffer (runtime-maintained)
kvk_buf = np.zeros((28,1,8,MAX_KV,128), dtype=np.float32)
kvv_buf = np.zeros((28,1,8,MAX_KV,128), dtype=np.float32)
for step in range(NUM_STEPS):
    z = np.load(T + '/step' + str(step) + '.npz')
    emb_in = z['emb_in'].astype(np.float32)
    pos = z['pos'].astype(np.float32)
    cp_val = int(z['cp'][0])
    kvk_cur = z['kvk'].astype(np.float32)
    kvv_cur = z['kvv'].astype(np.float32)
    hidden_ref = z['hidden'].astype(np.float32)
    n = kvk_cur.shape[3]
    # maintain running buffer: slots 0..n-1 = authority KV (which includes current)
    kvk_buf[:,:,:,:n-1,:] = kvk_cur[:,:,:,:n-1,:]
    kvv_buf[:,:,:,:n-1,:] = kvv_cur[:,:,:,:n-1,:]
    try:
        out = sess.run(None, {'inputs_embeds': emb_in, 'position_ids': pos, 'cache_position': np.array([cp_val], dtype=np.int64), 'kv_k': kvk_buf, 'kv_v': kvv_buf})
        h = out[0].astype(np.float32)
        cos_h = cos_fn(hidden_ref, h)
        all_cos = all_cos and cos_h >= 0.999
        log('step ' + str(step) + ' cp ' + str(cp_val) + ' cos ' + str(round(cos_h,6)) + ' max_ref ' + str(round(float(np.abs(hidden_ref).max()),3)) + ' max_ort ' + str(round(float(np.abs(h).max()),3)))
    except Exception as ex:
        log('STEP ERR ' + str(step) + ': ' + str(ex)[:300])
        raise
log('P3_ORT_ALL_COS ' + str(all_cos))
print('P3_ORT_DONE', all_cos, flush=True)