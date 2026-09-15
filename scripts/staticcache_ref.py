# P1: GraphBStaticCacheRef - custom StaticCache with slot-write update, used via base.forward
import os, json, numpy as np, torch, traceback
os.environ['HF_HUB_OFFLINE'] = '1'
D = r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/device-smoke'
B = r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/device-smoke/g4b_boundary'
MAX_KV = 768
NUM_STEPS = 20
LOG = open(D + '/staticcache_ref.log', 'w')
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
from transformers.cache_utils import DynamicCache
# custom StaticCache: slot-write at cache_position, seq_length = max written position + 1
class GraphBStaticCache(DynamicCache):
    def __init__(self, max_kv, n_layers=28, n_heads=8, head_dim=128, batch=1, dtype=torch.float32):
        super().__init__()
        self.max_kv = max_kv
        self._seq_len = 0
        self.layers = []
        for i in range(n_layers):
            self.layers.append(self._make_layer(max_kv, n_heads, head_dim, batch, dtype))
    def _make_layer(self, max_kv, n_heads, head_dim, batch, dtype):
        class _L:
            pass
        l = _L()
        l.keys = torch.zeros(batch, n_heads, max_kv, head_dim, dtype=dtype)
        l.values = torch.zeros(batch, n_heads, max_kv, head_dim, dtype=dtype)
        return l
    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        cp = int(cache_kwargs['cache_position'].item()) if cache_kwargs and 'cache_position' in cache_kwargs else self._seq_len
        # slot write at cp
        self.layers[layer_idx].keys[:, :, cp, :] = key_states[:, :, 0, :]
        self.layers[layer_idx].values[:, :, cp, :] = value_states[:, :, 0, :]
        self._seq_len = max(self._seq_len, cp + 1)
        return self.layers[layer_idx].keys[:, :, :self._seq_len], self.layers[layer_idx].values[:, :, :self._seq_len]
    def get_seq_length(self):
        return self._seq_len
    def get_mask_sizes(self, cache_position, layer_idx=None):
        return self._seq_len, 0
    def get_max_cache_shape(self):
        return self.max_kv
    def prefill(self, kv_k, kv_v, n):
        for i in range(28):
            self.layers[i].keys[:, :, :n, :] = kv_k[i][:, :, :n, :]
            self.layers[i].values[:, :, :n, :] = kv_v[i][:, :, :n, :]
        self._seq_len = n
# golden: DynamicCache
kv_dyn = DynamicCache()
# static ref
kv_stat = GraphBStaticCache(MAX_KV)
log('GraphBStaticCache created')
# prefill real KV62
kvk62 = torch.from_numpy(np.load(B + '/kv_k_decode.npy')).float()
kvv62 = torch.from_numpy(np.load(B + '/kv_v_decode.npy')).float()
for i in range(28):
    kv_dyn.update(kvk62[i], kvv62[i], i)
kv_stat.prefill(kvk62, kvv62, 62)
log('prefill both done, static seq=' + str(kv_stat.get_seq_length()))
z = np.load(B + '/step_decode_inputs.npz')
emb = torch.from_numpy(z['emb']).float()
pos = torch.from_numpy(z['pos']).float()
cp0 = int(z['cp'][0])
results = []; all_cos = True; all_exact = True
for step in range(NUM_STEPS):
    cp_val = cp0 + step
    try:
        with torch.inference_mode():
            out_dyn = base(inputs_embeds=emb, position_ids=pos, past_key_values=kv_dyn, cache_position=torch.tensor([cp_val]), use_cache=True, output_hidden_states=True)
        h_dyn = out_dyn.last_hidden_state
        with torch.inference_mode():
            out_stat = base(inputs_embeds=emb, position_ids=pos, past_key_values=kv_stat, cache_position=torch.tensor([cp_val]), use_cache=True, output_hidden_states=True)
        h_stat = out_stat.last_hidden_state
        cos_h = cos_fn(h_dyn.detach().float().numpy(), h_stat.detach().float().numpy())
        all_cos = all_cos and cos_h >= 0.9999
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
        log('step ' + str(step) + ' cp ' + str(cp_val) + ' cos ' + str(round(cos_h,6)) + ' exact15 ' + str(exact) + ' stat_seq ' + str(kv_stat.get_seq_length()))
        results.append({'step': step, 'cp': cp_val, 'cos': round(cos_h,6), 'exact15': exact})
        codec_h = [h_dyn.detach()] + [cp_emb[g](torch.tensor([[cd[g]]])) for g in range(15)]
        emb = torch.cat(codec_h, dim=1).sum(1, keepdim=True)
    except Exception as ex:
        log('STEP ERR: ' + str(ex)[:400])
        traceback.print_exc(file=LOG)
        raise
log('STATICREF_ALL_COS ' + str(all_cos) + ' ALL_EXACT ' + str(all_exact))
with open(D + '/staticcache_ref_result.json', 'w') as f:
    f.write(json.dumps({'results': results, 'all_cos': all_cos, 'all_exact': all_exact}, indent=2))
print('STATICREF_DONE', all_cos, all_exact, flush=True)