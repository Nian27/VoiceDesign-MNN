import os, math, numpy as np, torch
os.environ['HF_HUB_OFFLINE'] = '1'
torch.set_num_threads(1)
D = r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/device-smoke'; CR = D + '/vd_cpr'
def cos(a,b):
    a=np.asarray(a,np.float64).ravel(); b=np.asarray(b,np.float64).ravel()
    return float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12))
from qwen_tts import Qwen3TTSModel
import qwen_tts.core.models.modeling_qwen3_tts as MF
gw = Qwen3TTSModel.from_pretrained(r'E:/AndroidStudioProjects/qwen3tts-mnn/models/qwen3-tts/Qwen3-TTS-12Hz-1.7B-VoiceDesign', dtype=torch.float32, device_map='cpu')
cp = gw.model.talker.code_predictor; core = cp.model
print('attn impl =', getattr(core.config, '_attn_implementation', 'N/A'))
print('inner attn impl =', getattr(getattr(core,'config',None), '_attn_implementation_internal', 'N/A'))
tew = np.fromfile(CR + '/talker_codec_emb_weight.f32', dtype=np.float32).reshape(3072,2048)
past = np.fromfile(D + '/vd_m5b/prefill_hidden_last.raw', dtype=np.float32).reshape(1,1,2048)
c0e = tew[1995].reshape(1,1,2048)
with torch.no_grad():
    xp = cp.small_to_mtp_projection(torch.cat([torch.from_numpy(past), torch.from_numpy(c0e)], dim=1))
    core.config._attn_implementation = 'eager'
    o_e = core(inputs_embeds=xp, use_cache=True)
    he = o_e.last_hidden_state.numpy()
    core.config._attn_implementation = 'sdpa'
    o_s = core(inputs_embeds=xp, use_cache=True)
    hs = o_s.last_hidden_state.numpy()
    core.config._attn_implementation = 'eager'
    print('cos(eager, sdpa) = %.7f' % cos(he, hs))
    print('eager f4 pos1', he[0,1,:4])
    print('sdpa  f4 pos1', hs[0,1,:4])
    # hand-written reference (= my graph), using eager
    import numpy as np2
    pos = torch.arange(16).unsqueeze(0)
    c, si = core.rotary_emb(torch.zeros(1,1,1024), pos)
    c2 = c[:, :2]; s2 = si[:, :2]
    def rh(x):
        x1 = x[..., : x.shape[-1]//2]; x2 = x[..., x.shape[-1]//2:]
        return torch.cat((-x2, x1), dim=-1)
    h = xp; SCALE = 1.0/math.sqrt(128)
    am = torch.full((1,1,2,16), -1e4); am[0,0,0,0]=0; am[0,0,1,0]=0; am[0,0,1,1]=0
    sm0 = torch.zeros(1,1,16,1); sm0[0,0,0,0]=1
    sm1 = torch.zeros(1,1,16,1); sm1[0,0,1,0]=1
    for i in range(5):
        layer = core.layers[i]; residual = h
        hn = layer.input_layernorm(h); at = layer.self_attn
        sh = hn.shape[:-1]; shp = (*sh,-1,128)
        q = at.q_norm(at.q_proj(hn).view(shp)).transpose(1,2)
        k = at.k_norm(at.k_proj(hn).view(shp)).transpose(1,2)
        v = at.v_proj(hn).view(shp).transpose(1,2)
        qr = q*c2 + rh(q)*s2; kr = k*c2 + rh(k)*s2
        z = torch.zeros(1,8,16,128)
        kvk = z*(1-sm0-sm1) + kr[:,:,0:1,:]*sm0 + kr[:,:,1:2,:]*sm1
        kvv = z*(1-sm0-sm1) + v[:,:,0:1,:]*sm0 + v[:,:,1:2,:]*sm1
        sc = torch.matmul(qr, kvk.repeat_interleave(2,dim=1).transpose(-1,-2))*SCALE
        pr = torch.softmax(sc+am, dim=-1)
        ao = torch.matmul(pr, kvv.repeat_interleave(2,dim=1)).reshape(*sh,-1)
        h2 = residual + at.o_proj(ao); r2 = h2
        h = r2 + layer.mlp(layer.post_attention_layernorm(h2))
    hw = core.norm(h).numpy()
    print('cos(handwritten, eager) = %.7f' % cos(hw, he))
    print('cos(handwritten, sdpa)  = %.7f' % cos(hw, hs))
print('ATTN_IMPL_DONE')
