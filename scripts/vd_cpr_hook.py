import os, math, numpy as np, torch
os.environ['HF_HUB_OFFLINE'] = '1'; torch.set_num_threads(1)
D = r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/device-smoke'; CR = D + '/vd_cpr'
def cos(a,b):
    a=np.asarray(a,np.float64).ravel(); b=np.asarray(b,np.float64).ravel()
    return float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12))
def rh(x):
    x1 = x[..., : x.shape[-1]//2]; x2 = x[..., x.shape[-1]//2:]
    return torch.cat((-x2, x1), dim=-1)
from qwen_tts import Qwen3TTSModel
gw = Qwen3TTSModel.from_pretrained(r'E:/AndroidStudioProjects/qwen3tts-mnn/models/qwen3-tts/Qwen3-TTS-12Hz-1.7B-VoiceDesign', dtype=torch.float32, device_map='cpu')
cp = gw.model.talker.code_predictor; core = cp.model
at = core.layers[0].self_attn
cap = {}
def hook(name):
    def f(mod, inp, out):
        cap[name] = (inp[0].detach().clone() if len(inp) else None, out.detach().clone())
    return f
h1 = at.q_norm.register_forward_hook(hook('q_norm_out'))
h2 = at.k_norm.register_forward_hook(hook('k_norm_out'))
h3 = at.v_proj.register_forward_hook(hook('v_out'))
tew = np.fromfile(CR + '/talker_codec_emb_weight.f32', dtype=np.float32).reshape(3072,2048)
past = np.fromfile(D + '/vd_m5b/prefill_hidden_last.raw', dtype=np.float32).reshape(1,1,2048)
c0e = tew[1995].reshape(1,1,2048)
with torch.no_grad():
    xp = cp.small_to_mtp_projection(torch.cat([torch.from_numpy(past), torch.from_numpy(c0e)], dim=1))
    o = core(inputs_embeds=xp, use_cache=True)
    qn_model = cap['q_norm_out'][1].reshape(1,2,16,128).transpose(1,2)
    kn_model = cap['k_norm_out'][1].reshape(1,2,8,128).transpose(1,2)
    v_model  = cap['v_out'][1].reshape(1,2,8,128).transpose(1,2)
    # mine
    layer = core.layers[0]; hn = layer.input_layernorm(xp)
    shp = (*hn.shape[:-1], -1, 128)
    q_m = at.q_norm(at.q_proj(hn).view(shp)).transpose(1,2)
    k_m = at.k_norm(at.k_proj(hn).view(shp)).transpose(1,2)
    v_m = at.v_proj(hn).view(shp).transpose(1,2)
    print('cos(q_norm model, mine) = %.7f' % cos(qn_model.numpy(), q_m.numpy()))
    print('cos(k_norm model, mine) = %.7f' % cos(kn_model.numpy(), k_m.numpy()))
    print('cos(v      model, mine) = %.7f' % cos(v_model.numpy(), v_m.numpy()))
    # rope: model applies inside attention; compare post-rope K with the cache
    pk = o.past_key_values
    Kc = pk.key_cache[0] if hasattr(pk,'key_cache') else pk[0][0]
    pos = torch.arange(16).unsqueeze(0)
    c, si = core.rotary_emb(torch.zeros(1,1,1024), pos)
    c2 = c[:, :2].unsqueeze(1); s2 = si[:, :2].unsqueeze(1)
    k_mr = k_m*c2 + rh(k_m)*s2
    print('cos(cache K post-rope, my rope(k_norm)) = %.7f' % cos(Kc.numpy(), k_mr.numpy()))
    print('cache K shape', tuple(Kc.shape), ' my k_mr shape', tuple(k_mr.shape))
h1.remove(); h2.remove(); h3.remove()
print('HOOK_DONE')
