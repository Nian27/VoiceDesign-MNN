import os, math, numpy as np, torch, MNN
os.environ['HF_HUB_OFFLINE'] = '1'
torch.set_num_threads(1)
D = r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/device-smoke'; CR = D + '/vd_cpr'
CAP, HD = 16, 128
def cos(a,b):
    a=np.asarray(a,np.float64).ravel(); b=np.asarray(b,np.float64).ravel()
    return float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12))
from qwen_tts import Qwen3TTSModel
gw = Qwen3TTSModel.from_pretrained(r'E:/AndroidStudioProjects/qwen3tts-mnn/models/qwen3-tts/Qwen3-TTS-12Hz-1.7B-VoiceDesign', dtype=torch.float32, device_map='cpu')
cp = gw.model.talker.code_predictor; core = cp.model
tew = np.fromfile(CR + '/talker_codec_emb_weight.f32', dtype=np.float32).reshape(3072,2048)
past = np.fromfile(D + '/vd_m5b/prefill_hidden_last.raw', dtype=np.float32).reshape(1,1,2048)
code0 = 1995
c0e = tew[code0].reshape(1,1,2048)
# torch
with torch.no_grad():
    xin = torch.cat([torch.from_numpy(past), torch.from_numpy(c0e)], dim=1)
    xp = cp.small_to_mtp_projection(xin)
    print('proj out shape', tuple(xp.shape), 'f4', xp[0,0,:4].numpy())
    o = core(inputs_embeds=xp, use_cache=True)
    hfull = o.last_hidden_state
    print('torch hidden (1,2,1024) f4 last-pos', hfull[0,1,:4].numpy())
    pk = o.past_key_values
    k0 = pk.key_cache[0] if hasattr(pk,'key_cache') else pk[0][0]
    print('torch layer0 K shape', tuple(k0.shape), 'slot0 f4', k0[0,0,0,:4].numpy(), 'slot1 f4', k0[0,0,1,:4].numpy())
# rope
x0 = torch.zeros(1,1,1024); pos = torch.arange(CAP).unsqueeze(0)
with torch.no_grad():
    c, si = core.rotary_emb(x0, pos)
c = np.asarray(c, np.float32); si = np.asarray(si, np.float32)
print('rope cos[0][:4]', c[0,0,:4], ' cos[1][:4]', c[0,1,:4])
# MNN prefill
it = MNN.Interpreter(CR + '/codepred_prefill_fp16.mnn'); s = it.createSession()
def feed(name, arr):
    t = it.getSessionInput(s, name); shp = [d if d>0 else 1 for d in t.getShape()]
    ht = MNN.Tensor(shp, MNN.Halide_Type_Float, np.ascontiguousarray(arr, np.float32).ravel(), MNN.Tensor_DimensionType_Caffe)
    t.copyFromHostTensor(ht)
def out(nm):
    o = it.getSessionOutput(s, nm); shp=[d if d>0 else 1 for d in o.getShape()]
    ho = MNN.Tensor(shp, MNN.Halide_Type_Float, MNN.Tensor_DimensionType_Caffe); o.copyToHostTensor(ho)
    return np.asarray(ho.getNumpyData()).reshape(shp).astype(np.float32)
sm0 = np.zeros((1,1,CAP,1), np.float32); sm0[0,0,0,0]=1
sm1 = np.zeros((1,1,CAP,1), np.float32); sm1[0,0,1,0]=1
am2 = np.full((1,1,2,CAP), -1e4, np.float32)
am2[0,0,0,0]=0; am2[0,0,1,0]=0; am2[0,0,0,1]=0; am2[0,0,1,1]=0
feed('past_hidden', past); feed('code0_emb', c0e)
feed('rope_cos', c[:, :2].reshape(1,1,2,HD)); feed('rope_sin', si[:, :2].reshape(1,1,2,HD))
feed('slot_mask0', sm0); feed('slot_mask1', sm1); feed('attn_mask', am2)
it.runSession(s)
hl = out('hidden_last'); kvk = out('kv_k_out'); kvv = out('kv_v_out')
print('mnn hidden_last shape', hl.shape, 'f4', hl.ravel()[:4])
print('cos(mnn hidden_last, torch hidden last) = %.7f' % cos(hl.ravel(), hfull[0,1].numpy()))
print('cos(mnn hidden_last, torch hidden pos0) = %.7f' % cos(hl.ravel(), hfull[0,0].numpy()))
print('mnn kvk shape', kvk.shape, 'slot0 f4', kvk[0,0,0,0,:4], 'slot1 f4', kvk[0,0,0,1,:4])
print('cos(mnn kvk slot0, torch K slot0) = %.7f' % cos(kvk[0,0,0,0], k0[0,0,0].numpy()))
print('cos(mnn kvk slot1, torch K slot1) = %.7f' % cos(kvk[0,0,0,1], k0[0,0,1].numpy()))
print('DBG1_DONE')
