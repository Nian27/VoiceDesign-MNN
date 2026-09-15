# P5 verify: MNN codec_head vs torch, same hidden -> logits cos + top1
import os, json, numpy as np, torch, MNN
os.environ['HF_HUB_OFFLINE'] = '1'
D = r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/device-smoke'
LOG = open(D + '/p5_codec_mnn.log', 'w')
def log(s):
    LOG.write(s + '\n'); LOG.flush()
cos_fn = lambda a,b: float(np.dot(a.ravel(),b.ravel())/(np.linalg.norm(a)*np.linalg.norm(b)+1e-9))
from qwen_tts import Qwen3TTSModel
gw = Qwen3TTSModel.from_pretrained(r'E:/AndroidStudioProjects/qwen3tts-mnn/models/qwen3-tts/Qwen3-TTS-12Hz-1.7B-VoiceDesign', dtype=torch.float32, device_map='cpu')
ch = gw.model.talker.codec_head
log('LOADED')
interp = MNN.Interpreter(D + '/codec_head_fp16.mnn')
s2 = interp.createSession()
# test hidden from GraphB deploy (real step0 hidden)
B = r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/device-smoke/g4b_boundary'
z = np.load(B + '/step_decode_inputs.npz')
# use decode_step0_hidden as proxy hidden
h_ref = np.load(B + '/decode_step0_hidden.npy').astype(np.float32)
# torch logits
with torch.inference_mode():
    lg_t = ch(torch.from_numpy(h_ref))
lg_tv = lg_t[0,0].numpy()
# MNN logits
t_in = interp.getSessionInput(s2, 'hidden_states')
ht = MNN.Tensor((1,1,2048), MNN.Halide_Type_Float, h_ref.ravel(), MNN.Tensor_DimensionType_Caffe)
t_in.copyFromHostTensor(ht)
ot = interp.getSessionOutput(s2, 'logits')
interp.runSession(s2)
osh = [d if d > 0 else 1 for d in ot.getShape()]
ho = MNN.Tensor(osh, MNN.Halide_Type_Float, MNN.Tensor_DimensionType_Caffe)
ot.copyToHostTensor(ho)
lg_m = np.asarray(ho.getNumpyData()).reshape(osh).astype(np.float32)[0,0]
cos_l = cos_fn(lg_tv, lg_m)
t1 = int(np.argmax(lg_tv)); t2 = int(np.argmax(lg_m))
fin = bool(np.isfinite(lg_m).all())
log('logits cos ' + str(round(cos_l,6)) + ' top1 torch ' + str(t1) + ' mnn ' + str(t2) + ' finite ' + str(fin) + ' max_t ' + str(round(float(np.abs(lg_tv).max()),3)) + ' max_m ' + str(round(float(np.abs(lg_m).max()),3)))
print('P5_CODEC_MNN_DONE', round(cos_l,6), t1 == t2, flush=True)