# P2 ORT verify: deploy wrapper ONNX vs authority trajectory (20 steps, fixed MAX_KV)
import os, json, numpy as np, onnxruntime as ort
D = r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/device-smoke'
T = r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/device-smoke/p1_traj'
MAX_KV = 768
NUM_STEPS = 20
LOG = open(D + '/p2_ort.log', 'w')
def log(s):
    LOG.write(s + '\n'); LOG.flush()
cos_fn = lambda a,b: float(np.dot(a.ravel(),b.ravel())/(np.linalg.norm(a)*np.linalg.norm(b)+1e-9))
sess = ort.InferenceSession(D + '/graphb_deploy_m768.onnx', providers=['CPUExecutionProvider'])
print('session ok', flush=True)
all_cos = True
for step in range(NUM_STEPS):
    z = np.load(T + '/step' + str(step) + '.npz')
    emb_in = z['emb_in'].astype(np.float32)
    pos = z['pos'].astype(np.float32)
    cp_val = int(z['cp'][0])
    kvk_cur = z['kvk'].astype(np.float32)
    kvv_cur = z['kvv'].astype(np.float32)
    hidden_ref = z['hidden'].astype(np.float32)
    n = kvk_cur.shape[3]
    # fill KV_IN=767 buffers (base appends current -> 768)
    kvk = np.zeros((28,1,8,767,128), dtype=np.float32)
    kvv = np.zeros((28,1,8,767,128), dtype=np.float32)
    kvk[:,:,:,:n-1,:] = kvk_cur[:,:,:,:n-1,:]
    kvv[:,:,:,:n-1,:] = kvv_cur[:,:,:,:n-1,:]
    # mask 768: query at cp_val sees 0..cp_val
    am = np.zeros((1,1,1,768), dtype=np.float32)
    am[:,:,:,cp_val+1:] = float('-inf')
    am[:,:,:,:cp_val+1] = 0.0
    try:
        out = sess.run(None, {'inputs_embeds': emb_in, 'position_ids': pos, 'kv_k': kvk, 'kv_v': kvv, 'attention_mask': am})
        h = out[0].astype(np.float32)
        cos_h = cos_fn(hidden_ref, h)
        all_cos = all_cos and cos_h >= 0.999
        log('step ' + str(step) + ' cp ' + str(cp_val) + ' n ' + str(n) + ' cos ' + str(round(cos_h,6)) + ' max_ref ' + str(round(float(np.abs(hidden_ref).max()),3)) + ' max_ort ' + str(round(float(np.abs(h).max()),3)))
    except Exception as ex:
        log('STEP ERR ' + str(step) + ': ' + str(ex)[:300])
        raise
log('P2_ORT_ALL_COS ' + str(all_cos))
print('P2_ORT_DONE', all_cos, flush=True)