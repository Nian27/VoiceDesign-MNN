import os, math, numpy as np, torch, onnxruntime as ort
os.environ['HF_HUB_OFFLINE'] = '1'; torch.set_num_threads(1)
D = r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/device-smoke'; CR = D + '/vd_cpr'
def cos(a,b):
    a=np.asarray(a,np.float64).ravel(); b=np.asarray(b,np.float64).ravel()
    return float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12))
from qwen_tts import Qwen3TTSModel
gw = Qwen3TTSModel.from_pretrained(r'E:/AndroidStudioProjects/qwen3tts-mnn/models/qwen3-tts/Qwen3-TTS-12Hz-1.7B-VoiceDesign', dtype=torch.float32, device_map='cpu')
cp = gw.model.talker.code_predictor; core = cp.model; talker = gw.model.talker
CAP, HD = 16, 128
tew = np.fromfile(CR + '/talker_codec_emb_weight.f32', dtype=np.float32).reshape(3072,2048)
past = np.fromfile(D + '/vd_m5b/prefill_hidden_last.raw', dtype=np.float32).reshape(1,1,2048)
lg0 = None
with torch.no_grad():
    lg0 = talker.codec_head(torch.from_numpy(past))[0,0].numpy().astype(np.float32)
code0 = int(np.argmax(np.where(np.arange(3072)>=2048, np.where(np.arange(3072)==2150, lg0, -1e30), lg0)))
print('code0 (argmax for determinism) =', code0)
c0e = tew[code0].reshape(1,1,2048)
x0 = torch.zeros(1,1,1024); pos = torch.arange(CAP).unsqueeze(0)
with torch.no_grad(): c, si = core.rotary_emb(x0, pos)
c = np.asarray(c, np.float32); si = np.asarray(si, np.float32)
sm0 = np.zeros((1,1,CAP,1), np.float32); sm0[0,0,0,0]=1
sm1 = np.zeros((1,1,CAP,1), np.float32); sm1[0,0,1,0]=1
am = np.full((1,1,2,CAP), -1e4, np.float32); am[0,0,0,0]=0; am[0,0,1,0]=0; am[0,0,1,1]=0
ins = {'past_hidden': past, 'code0_emb': c0e,
       'rope_cos': c[:,:2].reshape(1,1,2,HD), 'rope_sin': si[:,:2].reshape(1,1,2,HD),
       'slot_mask0': sm0, 'slot_mask1': sm1, 'attn_mask': am}
so = ort.InferenceSession(CR + '/codepred_prefill.onnx', providers=['CPUExecutionProvider'])
ox = so.run(None, ins)
with torch.no_grad():
    xp = cp.small_to_mtp_projection(torch.cat([torch.from_numpy(past), torch.from_numpy(c0e)], dim=1))
    o = core(inputs_embeds=xp, use_cache=True)
    th = o.last_hidden_state[0,1].numpy()
    pk = o.past_key_values
    Kc = pk.key_cache[0] if hasattr(pk,'key_cache') else pk[0][0]
print('cos(onnx hidden_last, torch last) = %.7f' % cos(ox[0].ravel(), th))
print('onnx f4', ox[0].ravel()[:4])
print('torch f4', th[:4])
print('cos(onnx kvk slot0, torch K slot0) = %.7f' % cos(ox[1][0,0,0,0], Kc[0,0,0].numpy()))
print('cos(onnx kvk slot1, torch K slot1) = %.7f' % cos(ox[1][0,0,0,1], Kc[0,0,1].numpy()))
print('QUICKFIX_DONE')
