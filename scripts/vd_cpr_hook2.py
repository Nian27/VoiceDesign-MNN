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
hh = at.o_proj.register_forward_hook(lambda m,i,o: cap.__setitem__('ao_pre', i[0].detach().clone()))
tew = np.fromfile(CR + '/talker_codec_emb_weight.f32', dtype=np.float32).reshape(3072,2048)
past = np.fromfile(D + '/vd_m5b/prefill_hidden_last.raw', dtype=np.float32).reshape(1,1,2048)
c0e = tew[1995].reshape(1,1,2048)
with torch.no_grad():
    xp = cp.small_to_mtp_projection(torch.cat([torch.from_numpy(past), torch.from_numpy(c0e)], dim=1))
    o = core(inputs_embeds=xp, use_cache=True)
    ao_model = cap['ao_pre']           # (1,2,2048) pre-o_proj
    hn_model = xp


    layer = core.layers[0]
    hn = layer.input_layernorm(xp)

    shp = (*hn.shape[:-1], -1, 128)
    q = at.q_norm(at.q_proj(hn).view(shp)).transpose(1,2)
    k = at.k_norm(at.k_proj(hn).view(shp)).transpose(1,2)
    v = at.v_proj(hn).view(shp).transpose(1,2)
    pos = torch.arange(16).unsqueeze(0)
    c, si = core.rotary_emb(torch.zeros(1,1,1024), pos)
    c2 = c[:, :2].unsqueeze(1); s2 = si[:, :2].unsqueeze(1)
    qr = q*c2 + rh(q)*s2; kr = k*c2 + rh(k)*s2
    g = 2
    k_all = kr.repeat_interleave(g, dim=1)     # (1,16,2,128)
    v_all = v.repeat_interleave(g, dim=1)
    sc = torch.matmul(qr, k_all.transpose(-1,-2))/math.sqrt(128)
    am = torch.full((1,1,2,2), float('-inf')); am[0,0,0,0]=0; am[0,0,1,0]=0; am[0,0,1,1]=0
    pr = torch.softmax(sc+am, dim=-1)
    ao_mine = torch.matmul(pr, v_all).reshape(*hn.shape[:-1], -1)
    print('ao_model shape', tuple(ao_model.shape), ' ao_mine shape', tuple(ao_mine.shape))
    print('cos(ao_model, ao_mine) LENGTH2 = %.7f' % cos(ao_model.numpy(), ao_mine.numpy()))
    # now the 16-slot variant (what my graph does)
    z = torch.zeros(1,8,16,128)
    sm0 = torch.zeros(1,1,16,1); sm0[0,0,0,0]=1
    sm1 = torch.zeros(1,1,16,1); sm1[0,0,1,0]=1
    kvk = z*(1-sm0-sm1) + kr[:,:,0:1,:]*sm0 + kr[:,:,1:2,:]*sm1
    kvv = z*(1-sm0-sm1) + v[:,:,0:1,:]*sm0 + v[:,:,1:2,:]*sm1
    sc16 = torch.matmul(qr, kvk.repeat_interleave(g,dim=1).transpose(-1,-2))/math.sqrt(128)
    am16 = torch.full((1,1,2,16), -1e4); am16[0,0,0,0]=0; am16[0,0,1,0]=0; am16[0,0,1,1]=0
    pr16 = torch.softmax(sc16+am16, dim=-1)
    ao16 = torch.matmul(pr16, kvv.repeat_interleave(g,dim=1)).reshape(*hn.shape[:-1],-1)
    print('cos(ao_model, ao_mine) LENGTH16 = %.7f' % cos(ao_model.numpy(), ao16.numpy()))
hh.remove(); 
print('HOOK2_DONE')
