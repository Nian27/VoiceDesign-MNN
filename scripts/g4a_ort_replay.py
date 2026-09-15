# G4-A ORT sanity: fresh dynamic onnx vs torch frozen trajectory (20 steps)
import json, numpy as np, onnxruntime as ort
D = r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/device-smoke'
M22 = r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/m22/onnx'
NUM_STEPS = 20
LOG = open(D + '/g4a_ort_t1.log', 'w')
def log(s):
    LOG.write(s + '\n'); LOG.flush()
cos_fn = lambda a,b: float(np.dot(a.ravel(),b.ravel())/(np.linalg.norm(a)*np.linalg.norm(b)+1e-9))
sess = ort.InferenceSession(M22 + '/graphb_dynamic_fp32_t1.onnx', providers=['CPUExecutionProvider'])
print('session ok', flush=True)
results = []; all_ok = True
for step in range(NUM_STEPS):
    z = np.load(D + '/loop_pt/step' + str(step) + '.npz')
    emb = z['emb']; pos = z['pos']; kvk = z['kvk']; kvv = z['kvv']; hidden_ref = z['hidden']; cp = z['cp']
    n = kvk.shape[3]
    # dynamic kv_len: ORT needs exact shape
    out = sess.run(None, {'inputs_embeds': emb, 'position_ids': pos, 'kv_k': kvk, 'kv_v': kvv, 'cache_position': cp.astype(np.int64)})
    h = out[0]
    cos_h = cos_fn(hidden_ref, h)
    fin = bool(np.isfinite(h).all())
    log('step ' + str(step) + ' kv_len ' + str(n) + ' cos ' + str(round(cos_h,6)) + ' finite ' + str(fin) + ' max_ref ' + str(round(float(np.abs(hidden_ref).max()),3)) + ' max_ort ' + str(round(float(np.abs(h).max()),3)))
    results.append({'step': step, 'cos': round(cos_h,6), 'finite': fin})
    all_ok = all_ok and cos_h >= 0.9999 and fin
log('G4A_ORT_ALL ' + str(all_ok))
with open(D + '/g4a_ort_t1_result.json', 'w') as f:
    f.write(json.dumps({'results': results, 'all_pass': all_ok}, indent=2))
print('G4A_ORT_DONE', all_ok, flush=True)