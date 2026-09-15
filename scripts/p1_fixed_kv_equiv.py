# VD-M4 P1: Fixed-KV GraphBDeployWrapper vs official DynamicCache - 20-step equiv
import os, json, numpy as np, torch, traceback
os.environ['HF_HUB_OFFLINE'] = '1'
D = r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/device-smoke'
MAX_KV = 768
NUM_STEPS = 20
LOG = open(D + '/p1_fixed_kv.log', 'w')
def log(s):
    LOG.write(s + '\n'); LOG.flush()
cos_fn = lambda a,b: float(np.dot(a.ravel(),b.ravel())/(np.linalg.norm(a)*np.linalg.norm(b)+1e-9))
from qwen_tts import Qwen3TTSModel
gw = Qwen3TTSModel.from_pretrained(r'E:/AndroidStudioProjects/qwen3tts-mnn/models/qwen3-tts/Qwen3-TTS-12Hz-1.7B-VoiceDesign', dtype=torch.float32, device_map='cpu')
talker = gw.model.talker
base = talker.model
log('LOADED')
from transformers.cache_utils import DynamicCache
kv_off = DynamicCache()
kvk_buf = torch.zeros(28,1,8,MAX_KV,128, dtype=torch.float32)
kvv_buf = torch.zeros(28,1,8,MAX_KV,128, dtype=torch.float32)
B = r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/device-smoke/g4b_boundary'
kvk62 = torch.from_numpy(np.load(B + '/kv_k_decode.npy')).float()
kvv62 = torch.from_numpy(np.load(B + '/kv_v_decode.npy')).float()
kvk_buf[:,:,:,:62,:] = kvk62
kvv_buf[:,:,:,:62,:] = kvv62
for i in range(28):
    kv_off.update(kvk62[i], kvv62[i], i)
z = np.load(B + '/step_decode_inputs.npz')
emb = torch.from_numpy(z['emb']).float()
pos = torch.from_numpy(z['pos']).float()
cp0 = int(z['cp'][0])
log('start kv62 cp ' + str(cp0))
def fixed_forward(emb_t, pos_t, cp_val):
    kv = DynamicCache()
    for i in range(28):
        kv.update(kvk_buf[i][:,:,:cp_val].contiguous(), kvv_buf[i][:,:,:cp_val].contiguous(), i)
    out = base(
        inputs_embeds=emb_t,
        position_ids=pos_t,
        past_key_values=kv,
        cache_position=torch.tensor([cp_val]),
        use_cache=True,
        output_hidden_states=True,
    )
    return out.last_hidden_state
results = []; all_cos = True
for step in range(NUM_STEPS):
    cp_val = cp0 + step
    try:
        with torch.inference_mode():
            out_off = base(inputs_embeds=emb, position_ids=pos, past_key_values=kv_off, cache_position=torch.tensor([cp_val]), use_cache=True, output_hidden_states=True)
        h_off = out_off.last_hidden_state
        h_fix = fixed_forward(emb, pos, cp_val)
        cos_h = cos_fn(h_off.detach().float().numpy(), h_fix.detach().float().numpy())
        all_cos = all_cos and cos_h >= 0.999
        log('step ' + str(step) + ' cp ' + str(cp_val) + ' cos ' + str(round(cos_h,6)))
        results.append({'step': step, 'cp': cp_val, 'cos': round(cos_h,6)})
        if step < NUM_STEPS - 1:
            new_k = torch.stack([kv_off.layers[i].keys[:,:,-1:,:].clone() for i in range(28)], 0)
            new_v = torch.stack([kv_off.layers[i].values[:,:,-1:,:].clone() for i in range(28)], 0)
            kvk_buf[:,:,:,cp_val,:] = new_k[:,:,0,:,:]
            kvv_buf[:,:,:,cp_val,:] = new_v[:,:,0,:,:]
    except Exception as ex:
        log('STEP ERR: ' + str(ex)[:400])
        traceback.print_exc(file=LOG)
        raise
log('P1_ALL_COS ' + str(all_cos))
with open(D + '/p1_fixed_kv_result.json', 'w') as f:
    f.write(json.dumps({'results': results, 'all_cos_pass': all_cos}, indent=2))
print('P1_FIXED_KV_DONE', all_cos, flush=True)