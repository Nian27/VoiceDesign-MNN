import os, numpy as np, MNN
os.environ['HF_HUB_OFFLINE'] = '1'
D = r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/device-smoke'
LOG = open(D + '/p5_diag_shapes.log', 'w')
def log(s):
    LOG.write(s + '\n'); LOG.flush()
ch = MNN.Interpreter(D + '/codec_head_fp16.mnn')
chs = ch.createSession()
h = np.zeros((1,1,2048), dtype=np.float32)
t = ch.getSessionInput(chs, 'hidden_states')
ht = MNN.Tensor((1,1,2048), MNN.Halide_Type_Float, h.ravel(), MNN.Tensor_DimensionType_Caffe)
t.copyFromHostTensor(ht)
ch.runSession(chs)
ot = ch.getSessionOutput(chs, 'logits')
log('getShape: ' + str(ot.getShape()))
osh = [d if d > 0 else 3072 for d in ot.getShape()]
log('osh: ' + str(osh))
ho = MNN.Tensor(osh, MNN.Halide_Type_Float, MNN.Tensor_DimensionType_Caffe)
ot.copyToHostTensor(ho)
arr = np.asarray(ho.getNumpyData()).reshape(osh)
log('arr shape: ' + str(arr.shape) + ' flat size: ' + str(arr.ravel().size))
# sample with suppress
lg = arr.ravel().astype(np.float64) / 0.9
log('lg size: ' + str(lg.size) + ' suppress cond: ' + str(lg.size >= 3072))
if lg.size >= 3072:
    lg[2048:3072] = float('-inf')
    lg[2150] = lg[2150]
    log('suppress ok')
log('DONE')