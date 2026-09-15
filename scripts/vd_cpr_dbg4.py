import os, math, numpy as np, torch, onnxruntime as ort
os.environ['HF_HUB_OFFLINE'] = '1'
torch.set_num_threads(1)
D = r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/device-smoke'; CR = D + '/vd_cpr'
CAP, HD, NH, NKV = 16, 128, 16, 8
SCALE = 1.0/math.sqrt(HD)
def cos(a,b):
    a=np.asarray(a,np.float64).ravel(); b=np.asarray(b,np.float64).ravel()
    return float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12))
from qwen_tts import Qwen3TTSModel
gw = Qwen3TTSModel.from_pretrained(r'E:/AndroidStudioProjects/qwen3tts-mnn/models/qwen3-tts/Qwen3-TTS-12Hz-1.7B-VoiceDesign', dtype=torch.float32, device_map='cpu')
cp = gw.model.talker.code_predictor; core = cp.model
tew = np.fromfile(CR + '/talker_codec_emb_weight.f32', dtype=np.float32).reshape(3072,2048)
past = np.fromfile(D + '/vd_m5b/prefill_hidden_last.raw', dtype=np.float32).reshape(1,1,2048)
c0e = tew[1995].reshape(1,1,2048)
x0 = torch.zeros(1,1,1024); pos = torch.arange(CAP).unsqueeze(0)
with torch.no_grad(): c, si = core.rotary_emb(x0, pos)
c = np.asarray(c, np.float32); si = np.asarray(si, np.float32)
sm0 = np.zeros((1,1,CAP,1), np.float32); sm0[0,0,0,0]=1
sm1 = np.zeros((1,1,CAP,1), np.float32); sm1[0,0,1,0]=1
am = np.full((1,1,2,CAP), -1e4, np.float32); am[0,0,0,0]=0; am[0,0,1,0]=0; am[0,0,1,1]=0
ins = {'past_hidden': past, 'code0_emb': c0e,
       'rope_cos': c[:, :2].reshape(1,1,2,HD), 'rope_sin': si[:, :2].reshape(1,1,2,HD),
       'slot_mask0': sm0, 'slot_mask1': sm1, 'attn_mask': am}
so = ort.InferenceSession(CR + '/dbg_prefill.onnx', providers=['CPUExecutionProvider'])
names = ['q','k','kvk','scores','probs','ao_f','oo','h1','h2','out']
ox = dict(zip(names, so.run(None, ins)))

# ---- torch reference computed with the model's own modules ----
def rh(x):
    x1 = x[..., : x.shape[-1]//2]; x2 = x[..., x.shape[-1]//2:]
    return torch.cat((-x2, x1), dim=-1)
with torch.no_grad():
    xp = cp.small_to_mtp_projection(torch.cat([torch.from_numpy(past), torch.from_numpy(c0e)], dim=1))
    layer = core.layers[0]; at = layer.self_attn
    hn = layer.input_layernorm(xp)
    hs = hn.shape[:-1]; shp = (*hs,-1,HD)
    q = at.q_norm(at.q_proj(hn).view(shp)).transpose(1,2)
    k = at.k_norm(at.k_proj(hn).view(shp)).transpose(1,2)
    v = at.v_proj(hn).view(shp).transpose(1,2)
    qr = q*c[:,:2].reshape(1,1,2,HD) if False else q
    # use EXACT torch rope: call the model's own apply_rotary_pos_emb
    import qwen_tts.core.models.modeling_qwen3_tts as MF
    ct = torch.from_numpy(c[:, :2].reshape(1,2,HD)); st = torch.from_numpy(si[:, :2].reshape(1,2,HD))
    qr2, kr2 = MF.apply_rotary_pos_emb(q, k, ct, st)
    print('torch qr f4', qr2[0,0,0,:4].numpy())
    print('onnx  q  f4', ox['q'][0,0,0,:4])
    print('cos(torch q(pre-rope), onnx q) = %.7f' % cos(q.numpy(), ox['q']))
    print('cos(torch k(pre-rope), onnx k) = %.7f' % cos(k.numpy(), ox['k']))
    kvk_t = torch.zeros(1,NKV,CAP,HD)
    kvk_t = kvk_t*(1.0-torch.from_numpy(sm0)-torch.from_numpy(sm1)) + kr2[:,:,0:1,:]*torch.from_numpy(sm0) + kr2[:,:,1:2,:]*torch.from_numpy(sm1)
    print('cos(torch kvk, onnx kvk) = %.7f' % cos(kvk_t.numpy(), ox['kvk']))
    kvv_t = torch.zeros(1,NKV,CAP,HD)
    kvv_t = kvv_t*(1.0-torch.from_numpy(sm0)-torch.from_numpy(sm1)) + v[:,:,0:1,:]*torch.from_numpy(sm0) + v[:,:,1:2,:]*torch.from_numpy(sm1)
    print('cos(torch kvv, onnx kvv?)  (kvv not exported)')
    k_all = kvk_t.repeat_interleave(2, dim=1)
    sc = torch.matmul(qr2, k_all.transpose(-1,-2))*SCALE
    print('cos(torch scores, onnx scores) = %.7f' % cos(sc.numpy(), ox['scores']))
    pr = torch.softmax(sc + torch.from_numpy(am), dim=-1)
    print('cos(torch probs, onnx probs) = %.7f' % cos(pr.numpy(), ox['probs']))
    ao = torch.matmul(pr, kvv_t.repeat_interleave(2,dim=1)).reshape(*hs,-1).contiguous()
    print('cos(torch ao_f, onnx ao_f) = %.7f' % cos(ao.numpy(), ox['ao_f']))
    oo = at.o_proj(ao)
    print('cos(torch oo, onnx oo) = %.7f' % cos(oo.numpy(), ox['oo']))
    h1 = xp + oo
    print('cos(torch h1, onnx h1) = %.7f' % cos(h1.numpy(), ox['h1']))
    h2 = h1 + layer.mlp(layer.post_attention_layernorm(h1))
    print('cos(torch h2, onnx h2) = %.7f' % cos(h2.numpy(), ox['h2']))
    o = core(inputs_embeds=xp, use_cache=True)
    print('cos(torch core.last_hidden, onnx out) = %.7f' % cos(o.last_hidden_state.numpy(), ox['out']))
print('DBG4_DONE')
