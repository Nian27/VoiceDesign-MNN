# P3: GraphBDeployRef - hand-written 28-layer loop, fixed KV buffer in-place, no DynamicCache
import os, json, numpy as np, torch, traceback
os.environ['HF_HUB_OFFLINE'] = '1'
D = r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/device-smoke'
B = r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/device-smoke/g4b_boundary'
MAX_KV = 768
NUM_STEPS = 20
LOG = open(D + '/deploy_ref.log', 'w')
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
from qwen_tts.core.models.modeling_qwen3_tts import apply_multimodal_rotary_pos_emb
class GraphBDeployRef(torch.nn.Module):
    def __init__(self, base, max_kv):
        super().__init__()
        self.base = base
        self.max_kv = max_kv
        self.k_cache = None
        self.v_cache = None
    def reset(self, kvk, kvv, prefill_len):
        self.k_cache = torch.zeros(28,1,8,self.max_kv,128, dtype=torch.float32)
        self.v_cache = torch.zeros(28,1,8,self.max_kv,128, dtype=torch.float32)
        self.k_cache[:,:,:,:prefill_len,:] = kvk[:,:,:,:prefill_len,:]
        self.v_cache[:,:,:,:prefill_len,:] = kvv[:,:,:,:prefill_len,:]
    def forward(self, inputs_embeds, position_ids, cache_position):
        cp = int(cache_position.item())
        hidden = inputs_embeds
        pos_emb = self.base.rotary_emb(hidden, position_ids)
        for i in range(28):
            layer = self.base.layers[i]
            residual = hidden
            h = layer.input_layernorm(hidden)
            attn = layer.self_attn
            head_dim = attn.head_dim
            hs = h.shape[:-1]
            hidden_shape = (*hs, -1, head_dim)
            q = attn.q_norm(attn.q_proj(h).view(hidden_shape)).transpose(1,2)
            k = attn.k_norm(attn.k_proj(h).view(hidden_shape)).transpose(1,2)
            v = attn.v_proj(h).view(hidden_shape).transpose(1,2)
            cos, sin = pos_emb
            q, k = apply_multimodal_rotary_pos_emb(q, k, cos, sin, attn.rope_scaling['mrope_section'], attn.rope_scaling['interleaved'])
            self.k_cache[i, :, :, cp, :] = k[:, :, 0, :]
            self.v_cache[i, :, :, cp, :] = v[:, :, 0, :]
            kv_groups = attn.num_key_value_groups
            k_all = self.k_cache[i, :, :, :cp+1, :].repeat_interleave(kv_groups, dim=1)
            v_all = self.v_cache[i, :, :, :cp+1, :].repeat_interleave(kv_groups, dim=1)
            attn_out = torch.nn.functional.scaled_dot_product_attention(q, k_all, v_all, dropout_p=0.0, is_causal=False)
            attn_out = attn_out.reshape(*hs, -1).contiguous()
            attn_out = attn.o_proj(attn_out)
            h = residual + attn_out
            residual2 = h
            h = layer.post_attention_layernorm(h)
            h = layer.mlp(h)
            hidden = residual2 + h
        hidden = self.base.norm(hidden)
        return hidden
deploy = GraphBDeployRef(base, MAX_KV).eval()
log('GraphBDeployRef created')
kv_dyn = DynamicCache()
kvk62 = torch.from_numpy(np.load(B + '/kv_k_decode.npy')).float()
kvv62 = torch.from_numpy(np.load(B + '/kv_v_decode.npy')).float()
for i in range(28):
    kv_dyn.update(kvk62[i], kvv62[i], i)
deploy.reset(kvk62, kvv62, 62)
log('reset done')
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
            h_deploy = deploy(emb, pos, torch.tensor([cp_val]))
        cos_h = cos_fn(h_dyn.detach().float().numpy(), h_deploy.detach().float().numpy())
        all_cos = all_cos and cos_h >= 0.9999
        with torch.no_grad():
            pj_d = proj(h_dyn).clone(); pj_s = proj(h_deploy).clone()
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
        codec_h = [h_dyn.detach()] + [cp_emb[g](torch.tensor([[cd[g]]])) for g in range(15)]
        emb = torch.cat(codec_h, dim=1).sum(1, keepdim=True)
    except Exception as ex:
        log('STEP ERR: ' + str(ex)[:400])
        traceback.print_exc(file=LOG)
        raise
log('DEPLOYREF_ALL_COS ' + str(all_cos) + ' ALL_EXACT ' + str(all_exact))
with open(D + '/deploy_ref_result.json', 'w') as f:
    f.write(json.dumps({'results': results, 'all_cos': all_cos, 'all_exact': all_exact}, indent=2))
print('DEPLOYREF_DONE', all_cos, all_exact, flush=True)