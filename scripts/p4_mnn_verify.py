# P4 MNN verify: hand deploy fp16 MNN vs authority 20 steps (past/present)
import os, json, numpy as np, MNN
D = r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/device-smoke'
T = r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/device-smoke/p1_traj'
MAX_KV = 768
NUM_STEPS = 20
LOG = open(D + '/p4_mnn.log', 'w')
def log(s):
    LOG.write(s + '\n'); LOG.flush()
cos_fn = lambda a,b: float(np.dot(a.ravel(),b.ravel())/(np.linalg.norm(a)*np.linalg.norm(b)+1e-9))
interp = MNN.Interpreter(D + '/graphb_deploy_hand_fp16.mnn')
interp.setExternalFile(D + '/graphb_deploy_hand_fp16.mnn.weight')
s2 = interp.createSession()
def feed(name, arr, dt):
    t = interp.getSessionInput(s2, name)
    ht = MNN.Tensor(tuple(arr.shape), dt, arr.ravel(), MNN.Tensor_DimensionType_Caffe)
    t.copyFromHostTensor(ht)
def pull_output(name):
    t = interp.getSessionOutput(s2, name)
    osh = [d if d > 0 else 1 for d in t.getShape()]
    ho = MNN.Tensor(osh, MNN.Halide_Type_Float, MNN.Tensor_DimensionType_Caffe)
    t.copyToHostTensor(ho)
    return np.asarray(ho.getNumpyData()).reshape(osh).astype(np.float32)
all_cos = True
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
    kvk_buf[:,:,:,:n-1,:] = kvk_cur[:,:,:,:n-1,:]
    kvv_buf[:,:,:,:n-1,:] = kvv_cur[:,:,:,:n-1,:]
    try:
        feed('inputs_embeds', emb_in, MNN.Halide_Type_Float)
        feed('position_ids', pos, MNN.Halide_Type_Float)
        feed('cache_position', np.array([cp_val], dtype=np.int32), MNN.Halide_Type_Int)
        feed('kv_k', kvk_buf, MNN.Halide_Type_Float)
        feed('kv_v', kvv_buf, MNN.Halide_Type_Float)
        interp.runSession(s2)
        h = pull_output('hidden_states')
        kvk_buf = pull_output('kv_k_out')
        kvv_buf = pull_output('kv_v_out')
        cos_h = cos_fn(hidden_ref, h)
        all_cos = all_cos and cos_h >= 0.999
        log('step ' + str(step) + ' cp ' + str(cp_val) + ' cos ' + str(round(cos_h,6)) + ' max_ref ' + str(round(float(np.abs(hidden_ref).max()),3)) + ' max_mnn ' + str(round(float(np.abs(h).max()),3)))
    except Exception as ex:
        log('STEP ERR ' + str(step) + ': ' + str(ex)[:300])
        raise
log('P4_MNN_ALL_COS ' + str(all_cos))
print('P4_MNN_DONE', all_cos, flush=True)