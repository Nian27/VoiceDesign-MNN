import os, math, numpy as np, torch
os.environ['HF_HUB_OFFLINE'] = '1'
torch.set_num_threads(1)
D = r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/device-smoke'; CR = D + '/vd_cpr'
def cos(a,b):
    a=np.asarray(a,np.float64).ravel(); b=np.asarray(b,np.float64).ravel()
    return float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12))
from qwen_tts import Qwen3TTSModel
gw = Qwen3TTSModel.from_pretrained(r'E:/AndroidStudioProjects/qwen3tts-mnn/models/qwen3-tts/Qwen3-TTS-12Hz-1.7B-VoiceDesign', dtype=torch.float32, device_map='cpu')
cp = gw.model.talker.code_predictor; core = cp.model
tew = np.fromfile(CR + '/talker_codec_emb_weight.f32', dtype=np.float32).reshape(3072,2048)
past = np.fromfile(D + '/vd_m5b/prefill_hidden_last.raw', dtype=np.float32).reshape(1,1,2048)
c0e = tew[1995].reshape(1,1,2048)
def rh(x):
    x1 = x[..., : x.shape[-1]//2]; x2 = x[..., x.shape[-1]//2:]
    return torch.cat((-x2, x1), dim=-1)
with torch.no_grad():
    xp = cp.small_to_mtp_projection(torch.cat([torch.from_numpy(past), torch.from_numpy(c0e)], dim=1))
    o = core(inputs_embeds=xp, use_cache=True, output_hidden_states=True)
    hs = [t.numpy() for t in o.hidden_states]
    pos = torch.arange(16).unsqueeze(0)
    c, si = core.rotary_emb(torch.zeros(1,1,1024), pos)
    c2 = c[:, :2].unsqueeze(1); s2 = si[:, :2].unsqueeze(1)
    am = torch.full((1,1,2,16), -1e4); am[0,0,0,0]=0; am[0,0,1,0]=0; am[0,0,1,1]=0
    sm0 = torch.zeros(1,1,16,1); sm0[0,0,0,0]=1
    sm1 = torch.zeros(1,1,16,1); sm1[0,0,1,0]=1
    h = xp
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
        sc = torch.matmul(qr, kvk.repeat_interleave(2,dim=1).transpose(-1,-2))/math.sqrt(128)
        pr = torch.softmax(sc+am, dim=-1)
        ao = torch.matmul(pr, kvv.repeat_interleave(2,dim=1)).reshape(*sh,-1)
        attn_out = at.o_proj(ao)
        h_pre_mlp = residual + attn_out
        post = layer.post_attention_layernorm(h_pre_mlp)
        mlp_out = layer.mlp(post)
        h = h_pre_mlp + mlp_out
        print('layer %d  post-attn cos=%.7f  post-mlp cos=%.7f' % (
            i, cos(h_pre_mlp.numpy(), hs[i+1]), cos(h.numpy(), hs[i+1])))
    print('final norm cos=%.7f' % cos(core.norm(h).numpy(), hs[5]))
print('DBG7_DONE')
