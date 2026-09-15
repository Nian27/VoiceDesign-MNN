import os, math, inspect, numpy as np, torch
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
sig = inspect.signature(MF.Qwen3TTSAttention.forward)
print('attention forward params:', list(sig.parameters.keys()))
tew = np.fromfile(CR + '/talker_codec_emb_weight.f32', dtype=np.float32).reshape(3072,2048)
past = np.fromfile(D + '/vd_m5b/prefill_hidden_last.raw', dtype=np.float32).reshape(1,1,2048)
c0e = tew[1995].reshape(1,1,2048)
def rh(x):
    x1 = x[..., : x.shape[-1]//2]; x2 = x[..., x.shape[-1]//2:]
    return torch.cat((-x2, x1), dim=-1)
with torch.no_grad():
    xp = cp.small_to_mtp_projection(torch.cat([torch.from_numpy(past), torch.from_numpy(c0e)], dim=1))
    pos = torch.arange(16).unsqueeze(0)
    c, si = core.rotary_emb(torch.zeros(1,1,1024), pos)
    c2 = c[:, :2].unsqueeze(1); s2 = si[:, :2].unsqueeze(1)
    layer = core.layers[0]; at = layer.self_attn
    hn = layer.input_layernorm(xp)
    am4 = torch.full((1,1,2,16), float('-inf')); am4[0,0,0,0]=0.0; am4[0,0,1,0]=0.0; am4[0,0,1,1]=0.0
    try:
        aout = at(hidden_states=hn, position_embeddings=(c2.squeeze(1), s2.squeeze(1)),
                  attention_mask=am4, past_key_values=None, cache_position=torch.arange(2))
        model_ao = aout[0] if isinstance(aout, tuple) else aout
        print('model attn out shape', tuple(model_ao.shape))
    except Exception as e:
        print('direct call failed:', type(e).__name__, str(e)[:200])
        model_ao = None
    # mine
    shp = (*hn.shape[:-1], -1, 128)
    q = at.q_norm(at.q_proj(hn).view(shp)).transpose(1,2)
    k = at.k_norm(at.k_proj(hn).view(shp)).transpose(1,2)
    v = at.v_proj(hn).view(shp).transpose(1,2)
    qr = q*c2 + rh(q)*s2; kr = k*c2 + rh(k)*s2
    z = torch.zeros(1,8,16,128)
    sm0 = torch.zeros(1,1,16,1); sm0[0,0,0,0]=1
    sm1 = torch.zeros(1,1,16,1); sm1[0,0,1,0]=1
    kvk = z*(1-sm0-sm1) + kr[:,:,0:1,:]*sm0 + kr[:,:,1:2,:]*sm1
    kvv = z*(1-sm0-sm1) + v[:,:,0:1,:]*sm0 + v[:,:,1:2,:]*sm1
    am = torch.full((1,1,2,16), -1e4); am[0,0,0,0]=0; am[0,0,1,0]=0; am[0,0,1,1]=0
    sc = torch.matmul(qr, kvk.repeat_interleave(2,dim=1).transpose(-1,-2))/math.sqrt(128)
    pr = torch.softmax(sc+am, dim=-1)
    mine_ao = torch.matmul(pr, kvv.repeat_interleave(2,dim=1)).reshape(*hn.shape[:-1], -1)
    print('mine attn out shape', tuple(mine_ao.shape))
    if model_ao is not None:
        print('cos(model attn_out, o_proj(mine attn_out)) = %.7f' % cos(model_ao.numpy(), at.o_proj(mine_ao).numpy()))
        h_model = xp + model_ao
        h_mine = xp + at.o_proj(mine_ao)
        print('cos(h_pre_mlp model, mine) = %.7f' % cos(h_model.numpy(), h_mine.numpy()))
        print('cos(v_all model, mine) = %.7f' % cos(kvv.repeat_interleave(2,dim=1).numpy(), kvv.repeat_interleave(2,dim=1).numpy()))
    # recompute scores with -inf mask instead of -1e4
    am2 = torch.full((1,1,2,16), float('-inf')); am2[0,0,0,0]=0; am2[0,0,1,0]=0; am2[0,0,1,1]=0
    pr2 = torch.softmax(sc+am2, dim=-1)
    mine_ao2 = torch.matmul(pr2, kvv.repeat_interleave(2,dim=1)).reshape(*hn.shape[:-1], -1)
    print('cos(-1e4 mask vs -inf mask) mine: %.7f' % cos(mine_ao.numpy(), mine_ao2.numpy()))
print('DBG9_DONE')
