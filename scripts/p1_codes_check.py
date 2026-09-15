# P1 codes check: fixed-KV hidden -> codepred 15 codes vs authority codes (exact)
import os, json, numpy as np, torch
os.environ['HF_HUB_OFFLINE'] = '1'
D = r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/device-smoke'
T = r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/device-smoke/p1_traj'
MAX_KV = 768
NUM_STEPS = 20
LOG = open(D + '/p1_codes.log', 'w')
def log(s):
    LOG.write(s + '\n'); LOG.flush()
from qwen_tts import Qwen3TTSModel
gw = Qwen3TTSModel.from_pretrained(r'E:/AndroidStudioProjects/qwen3tts-mnn/models/qwen3-tts/Qwen3-TTS-12Hz-1.7B-VoiceDesign', dtype=torch.float32, device_map='cpu')
talker = gw.model.talker
base = talker.model
cpm = talker.code_predictor.model
lm = talker.code_predictor.lm_head
proj = talker.code_predictor.small_to_mtp_projection
log('LOADED')
from transformers.cache_utils import DynamicCache
kvk_buf = torch.zeros(28,1,8,MAX_KV,128, dtype=torch.float32)
kvv_buf = torch.zeros(28,1,8,MAX_KV,128, dtype=torch.float32)
all_exact = True
for step in range(NUM_STEPS):
    z = np.load(T + '/step' + str(step) + '.npz')
    emb_in = torch.from_numpy(z['emb_in']).float()
    pos = torch.from_numpy(z['pos']).float()
    cp_val = int(z['cp'][0])
    kvk_cur = torch.from_numpy(z['kvk']).float()
    kvv_cur = torch.from_numpy(z['kvv']).float()
    codes_ref = z['codes'].tolist()
    n = kvk_cur.shape[3]
    kvk_buf[:,:,:,:n,:] = kvk_cur
    kvv_buf[:,:,:,:n,:] = kvv_cur
    kv_fix = DynamicCache()
    for i in range(28):
        kv_fix.update(kvk_buf[i][:,:,:n-1].contiguous(), kvv_buf[i][:,:,:n-1].contiguous(), i)
    with torch.inference_mode():
        out_fix = base(inputs_embeds=emb_in, position_ids=pos, past_key_values=kv_fix, cache_position=torch.tensor([cp_val]), use_cache=True, output_hidden_states=True)
    h_fix = out_fix.last_hidden_state
    with torch.no_grad():
        pj = proj(h_fix).clone()
    codes_fix = []
    hid_t = pj
    for g in range(15):
        with torch.inference_mode():
            o = cpm(inputs_embeds=hid_t, use_cache=False)
            lg = lm[g](o.last_hidden_state)
        codes_fix.append(int(torch.argmax(lg[0,0])))
    exact = codes_ref == codes_fix
    all_exact = all_exact and exact
    log('step ' + str(step) + ' exact15 ' + str(exact) + ' ref ' + str(codes_ref[:5]) + ' fix ' + str(codes_fix[:5]))
log('P1_CODES_ALL_EXACT ' + str(all_exact))
print('P1_CODES_DONE', all_exact, flush=True)