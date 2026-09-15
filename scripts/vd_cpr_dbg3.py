import os, numpy as np, torch, MNN, onnxruntime as ort
os.environ['HF_HUB_OFFLINE'] = '1'
torch.set_num_threads(1)
D = r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/device-smoke'; CR = D + '/vd_cpr'
CAP, HD = 16, 128
def cos(a,b):
    a=np.asarray(a,np.float64).ravel(); b=np.asarray(b,np.float64).ravel()
    return float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12))
tew = np.fromfile(CR + '/talker_codec_emb_weight.f32', dtype=np.float32).reshape(3072,2048)
past = np.fromfile(D + '/vd_m5b/prefill_hidden_last.raw', dtype=np.float32).reshape(1,1,2048)
code0 = 1995
c0e = tew[code0].reshape(1,1,2048)
from qwen_tts import Qwen3TTSModel
gw = Qwen3TTSModel.from_pretrained(r'E:/AndroidStudioProjects/qwen3tts-mnn/models/qwen3-tts/Qwen3-TTS-12Hz-1.7B-VoiceDesign', dtype=torch.float32, device_map='cpu')
cp = gw.model.talker.code_predictor; core = cp.model
x0 = torch.zeros(1,1,1024); pos = torch.arange(CAP).unsqueeze(0)
with torch.no_grad():
    c, si = core.rotary_emb(x0, pos)
c = np.asarray(c, np.float32); si = np.asarray(si, np.float32)
sm0 = np.zeros((1,1,CAP,1), np.float32); sm0[0,0,0,0]=1
sm1 = np.zeros((1,1,CAP,1), np.float32); sm1[0,0,1,0]=1
am2 = np.full((1,1,2,CAP), -1e4, np.float32)
am2[0,0,0,0]=0; am2[0,0,1,0]=0; am2[0,0,1,1]=0
ins = {'past_hidden': past, 'code0_emb': c0e,
       'rope_cos': c[:, :2].reshape(1,1,2,HD), 'rope_sin': si[:, :2].reshape(1,1,2,HD),
       'slot_mask0': sm0, 'slot_mask1': sm1, 'attn_mask': am2}
so = ort.InferenceSession(CR + '/codepred_prefill.onnx', providers=['CPUExecutionProvider'])
ox = so.run(None, ins)
print('onnx hidden_last f4', ox[0].ravel()[:4])
# torch ref of the same forward
with torch.no_grad():
    xx = cp.small_to_mtp_projection(torch.cat([torch.from_numpy(past), torch.from_numpy(c0e)], dim=1))
    o = core(inputs_embeds=xx, use_cache=True)
    th = o.last_hidden_state[0,1].numpy()
print('torch hidden f4', th[:4])
print('cos(onnx, torch) = %.7f' % cos(ox[0].ravel(), th))
# MNN
it = MNN.Interpreter(CR + '/codepred_prefill_fp16.mnn'); s = it.createSession()
for nm, arr in ins.items():
    t = it.getSessionInput(s, nm); shp=[d if d>0 else 1 for d in t.getShape()]
    ht = MNN.Tensor(shp, MNN.Halide_Type_Float, np.ascontiguousarray(arr,np.float32).ravel(), MNN.Tensor_DimensionType_Caffe)
    t.copyFromHostTensor(ht)
it.runSession(s)
o2 = it.getSessionOutput(s, 'hidden_last'); shp=[d if d>0 else 1 for d in o2.getShape()]
ho = MNN.Tensor(shp, MNN.Halide_Type_Float, MNN.Tensor_DimensionType_Caffe); o2.copyToHostTensor(ho)
mn = np.asarray(ho.getNumpyData()).reshape(shp).astype(np.float32)
print('mnn   hidden f4', mn.ravel()[:4])
print('cos(mnn, torch)  = %.7f' % cos(mn.ravel(), th))
print('cos(mnn, onnx)   = %.7f' % cos(mn.ravel(), ox[0].ravel()))
# also with a causal-CORRECT mask vs onnx: recompute onnx with fixed mask
am2b = np.full((1,1,2,CAP), -1e4, np.float32)
am2b[0,0,0,0]=0; am2b[0,0,1,0]=0; am2b[0,0,1,1]=0
ins2 = dict(ins); ins2['attn_mask'] = am2b
ox2 = so.run(None, ins2)
print('cos(onnx_fixedmask, torch) = %.7f' % cos(ox2[0].ravel(), th))
print('DBG3_DONE')
