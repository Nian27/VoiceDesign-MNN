# P1 compare: fixed-KV wrapper vs authority trajectory (per-step same input)
import os, json, numpy as np, torch, traceback
os.environ['HF_HUB_OFFLINE'] = '1'
D = r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/device-smoke'
T = r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/device-smoke/p1_traj'
MAX_KV = 768
NUM_STEPS = 20
LOG = open(D + '/p1_compare.log', 'w')
def log(s):
    LOG.write(s + '\n'); LOG.flush()
cos_fn = lambda a,b: float(np.dot(a.ravel(),b.ravel())/(np.linalg.norm(a)*np.linalg.norm(b)+1e-9))
from qwen_tts import Qwen3TTSModel
gw = Qwen3TTSModel.from_pretrained(r'E:/AndroidStudioProjects/qwen3tts-mnn/models/qwen3-tts/Qwen3-TTS-12Hz-1.7B-VoiceDesign', dtype=torch.float32, device_map='cpu')
base = gw.model.talker.model
log('LOADED')
from transformers.cache_utils import DynamicCache
# fixed buffer, slot-updated from authority trajectory
kvk_buf = torch.zeros(28,1,8,MAX_KV,128, dtype=torch.float32)
kvv_buf = torch.zeros(28,1,8,MAX_KV,128, dtype=torch.float32)
results = []; all_cos = True; all_codes = True
traj = json.load(open(T + '/traj.json'))
for step in range(NUM_STEPS):
    z = np.load(T + '/step' + str(step) + '.npz')
    emb_in = torch.from_numpy(z['emb_in']).float()
    pos = torch.from_numpy(z['pos']).float()
    cp_val = int(z['cp'][0])
    kvk_cur = torch.from_numpy(z['kvk']).float()
    kvv_cur = torch.from_numpy(z['kvv']).float()
    hidden_ref = torch.from_numpy(z['hidden']).float()
    codes_ref = z['codes'].tolist()
    n = kvk_cur.shape[3]  # authority current kv length (incl current token)
    try:
        # fixed: fill buffer slots 0..n-1 from authority kv (which already includes current)
        kvk_buf[:,:,:,:n,:] = kvk_cur
        kvv_buf[:,:,:,:n,:] = kvv_cur
        # fixed forward: pass kv up to n-1 (all current), cache_position=cp
        kv_fix = DynamicCache()
        for i in range(28):
            kv_fix.update(kvk_buf[i][:,:,:n-1].contiguous(), kvv_buf[i][:,:,:n-1].contiguous(), i)
        with torch.inference_mode():
            out_fix = base(inputs_embeds=emb_in, position_ids=pos, past_key_values=kv_fix, cache_position=torch.tensor([cp_val]), use_cache=True, output_hidden_states=True)
        h_fix = out_fix.last_hidden_state
        cos_h = cos_fn(hidden_ref.detach().numpy(), h_fix.detach().numpy())
        all_cos = all_cos and cos_h >= 0.9999
        # codes from fixed hidden (parallel 15 via authority proj? use fixed hidden for codepred)
        log('step ' + str(step) + ' cp ' + str(cp_val) + ' kvlen ' + str(n) + ' cos ' + str(round(cos_h,6)))
        results.append({'step': step, 'cp': cp_val, 'kv_len': n, 'cos': round(cos_h,6)})
    except Exception as ex:
        log('STEP ERR: ' + str(ex)[:400])
        traceback.print_exc(file=LOG)
        raise
log('P1_COMPARE_ALL_COS ' + str(all_cos))
with open(D + '/p1_compare_result.json', 'w') as f:
    f.write(json.dumps({'results': results, 'all_cos_pass': all_cos}, indent=2))
print('P1_COMPARE_DONE', all_cos, flush=True)