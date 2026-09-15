# VD-M4: PyTorch StaticCache slot-update vs DynamicCache golden - 20-step equiv
import os, json, numpy as np, torch, traceback
os.environ['HF_HUB_OFFLINE'] = '1'
D = r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/device-smoke'
B = r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/device-smoke/g4b_boundary'
OUT = r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/device-smoke/staticcache_traj'
os.makedirs(OUT, exist_ok=True)
MAX_KV = 768
NUM_STEPS = 20
LOG = open(D + '/staticcache_equiv.log', 'w')
def log(s):
    LOG.write(s + '\n'); LOG.flush()
cos_fn = lambda a,b: float(np.dot(a.ravel(),b.ravel())/(np.linalg.norm(a)*np.linalg.norm(b)+1e-9))
from qwen_tts import Qwen3TTSModel
gw = Qwen3TTSModel.from_pretrained(r'E:/AndroidStudioProjects/qwen3tts-mnn/models/qwen3-tts/Qwen3-TTS-12Hz-1.7B-VoiceDesign', dtype=torch.float32, device_map='cpu')
talker = gw.model.talker
base = talker.model
cpm = talker.code_predictor.model
lm = talker.code_predictor.lm_head
cp_emb = talker.code_predictor.get_input_embeddings()
proj = talker.code_predictor.small_to_mtp_projection
log('LOADED')
from transformers.cache_utils import DynamicCache, StaticCache
# golden: DynamicCache growing
kv_dyn = DynamicCache()
# static: StaticCache max_cache_len=768
kv_stat = StaticCache(config=talker.model.config, max_batch_size=1, max_cache_len=MAX_KV, device='cpu', dtype=torch.float32)
log('StaticCache created: ' + str(type(kv_stat).__name__))
# prefill: real KV62 into both
kvk62 = torch.from_numpy(np.load(B + '/kv_k_decode.npy')).float()
kvv62 = torch.from_numpy(np.load(B + '/kv_v_decode.npy')).float()
for i in range(28):
    kv_dyn.update(kvk62[i], kvv62[i], i)
# StaticCache: write prefill slots 0..61 via update at cache_position=62-1? use update with full seq
# StaticCache.update(key, value, cache_position) writes at cache_position slots
pref_pos = torch.arange(62, dtype=torch.int64)
try:
    for i in range(28):
        kv_stat.update(kvk62[i][:, :, :62, :], kvv62[i][:, :, :62, :], pref_pos)
    log('StaticCache prefill ok')
except Exception as e:
    log('StaticCache prefill ERR: ' + str(e)[:300])
    traceback.print_exc(file=LOG)
z = np.load(B + '/step_decode_inputs.npz')
emb = torch.from_numpy(z['emb']).float()
pos = torch.from_numpy(z['pos']).float()
cp0 = int(z['cp'][0])
log('start cp ' + str(cp0))
results = []; all_cos = True; all_exact = True
for step in range(NUM_STEPS):
    cp_val = cp0 + step
    try:
        # A: dynamic golden
        with torch.inference_mode():
            out_dyn = base(inputs_embeds=emb, position_ids=pos, past_key_values=kv_dyn, cache_position=torch.tensor([cp_val]), use_cache=True, output_hidden_states=True)
        h_dyn = out_dyn.last_hidden_state
        # B: static slot-update
        with torch.inference_mode():
            out_stat = base(inputs_embeds=emb, position_ids=pos, past_key_values=kv_stat, cache_position=torch.tensor([cp_val]), use_cache=True, output_hidden_states=True)
        h_stat = out_stat.last_hidden_state
        cos_h = cos_fn(h_dyn.detach().float().numpy(), h_stat.detach().float().numpy())
        all_cos = all_cos and cos_h >= 0.999
        # codes from both
        with torch.no_grad():
            pj_d = proj(h_dyn).clone(); pj_s = proj(h_stat).clone()
        cd = []; cs = []
        for g in range(15):
            with torch.inference_mode():
                od = cpm(inputs_embeds=pj_d, use_cache=False); lg_d = lm[g](od.last_hidden_state)
                os_ = cpm(inputs_embeds=pj_s, use_cache=False); lg_s = lm[g](os_.last_hidden_state)
            cd.append(int(torch.argmax(lg_d[0,0]))); cs.append(int(torch.argmax(lg_s[0,0])))
        exact = cd == cs
        all_exact = all_exact and exact
        log('step ' + str(step) + ' cp ' + str(cp_val) + ' cos ' + str(round(cos_h,6)) + ' exact15 ' + str(exact))
        results.append({'step': step, 'cp': cp_val, 'cos': round(cos_h,6), 'exact15': exact})
        # next emb from dynamic (authoritative)
        codec_h = [h_dyn.detach()] + [cp_emb[g](torch.tensor([[cd[g]]])) for g in range(15)]
        emb = torch.cat(codec_h, dim=1).sum(1, keepdim=True)
    except Exception as ex:
        log('STEP ERR: ' + str(ex)[:400])
        traceback.print_exc(file=LOG)
        raise
log('STATICCACHE_ALL_COS ' + str(all_cos) + ' ALL_EXACT ' + str(all_exact))
with open(D + '/staticcache_result.json', 'w') as f:
    f.write(json.dumps({'results': results, 'all_cos': all_cos, 'all_exact': all_exact}, indent=2))
print('STATICCACHE_DONE', all_cos, all_exact, flush=True)