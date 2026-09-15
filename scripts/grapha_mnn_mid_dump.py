import numpy as np, MNN
D = r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/device-smoke'
emb = np.fromfile(D + '/grapha_emb_f32.raw', dtype=np.float32).reshape(1,34,2048)
pos = np.fromfile(D + '/grapha_pos_f32.raw', dtype=np.float32).reshape(3,1,34)
interp = MNN.Interpreter(D + '/grapha_prefill_static_fp16.mnn')
interp.setExternalFile(D + '/grapha_prefill_static_fp16.mnn.weight')
s2 = interp.createSession()
interp.resizeTensor(interp.getSessionInput(s2,'inputs_embeds'), (1,34,2048))
interp.resizeTensor(interp.getSessionInput(s2,'position_ids'), (3,1,34))
interp.resizeSession(s2)
def feed(name, arr, dt):
    t = interp.getSessionInput(s2, name)
    ht = MNN.Tensor(tuple(arr.shape), dt, arr.ravel(), MNN.Tensor_DimensionType_Caffe)
    t.copyFromHostTensor(ht)
feed('inputs_embeds', emb, MNN.Halide_Type_Float)
feed('position_ids', pos, MNN.Halide_Type_Float)
interp.runSession(s2)
# 取最终输出确认（应损坏 1e18）
ot = interp.getSessionOutput(s2, 'hidden_states')
osh = [d if d > 0 else 2048 for d in ot.getShape()]
ho = MNN.Tensor(osh, MNN.Halide_Type_Float, MNN.Tensor_DimensionType_Caffe)
ot.copyToHostTensor(ho)
h = np.asarray(ho.getNumpyData()).reshape(osh)
print('MNN hidden maxabs', float(np.abs(h).max()))
# 尝试取中间张量：MNN 是否保留 mul_3 等名字？
for tname in ['mul_3', 'view_1', 'add_4', 'add_5', 'mul_8', 'linear']:
    try:
        t = interp.getSessionOutput(s2, tname)
        sh = [d if d > 0 else 1 for d in t.getShape()]
        ht = MNN.Tensor(sh, MNN.Halide_Type_Float, MNN.Tensor_DimensionType_Caffe)
        t.copyToHostTensor(ht)
        arr = np.asarray(ht.getNumpyData()).reshape(sh)
        print(tname, 'OK shape', sh, 'maxabs', round(float(np.abs(arr).max()),4))
    except Exception as e:
        print(tname, 'ERR', str(e)[:80])