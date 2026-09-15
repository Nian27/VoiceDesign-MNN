import numpy as np, MNN, MNN.expr as expr, time
D = r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/device-smoke'
emb = np.fromfile(D + '/gb_emb_f32.raw', dtype=np.float32).reshape(1,1,2048)
pos = np.fromfile(D + '/gb_pos_f32.raw', dtype=np.float32).reshape(3,1,1)
kvk = np.fromfile(D + '/gb_kvk_f32.raw', dtype=np.float32).reshape(28,1,8,35,128)
kvv = np.fromfile(D + '/gb_kvv_f32.raw', dtype=np.float32).reshape(28,1,8,35,128)
cp = np.fromfile(D + '/gb_cp_i32.raw', dtype=np.int32).reshape(1)
interp = MNN.Interpreter(D + '/graphb_static_t35_fp16.mnn')
interp.setExternalFile(D + '/graphb_static_t35_fp16.mnn.weight')
sess = interp.createSession()
ins = interp.getSessionInputAll(sess)
outs = interp.getSessionOutputAll(sess)
order = list(ins.keys())
print('MNN input order:', order)
mod = MNN.nn.load_module(order, list(outs.keys()), interp)
print('module loaded')
def mkvar(arr, dt):
    v = expr._Input(tuple(arr.shape), MNN.Tensor_DimensionType_Caffe, dt)
    expr.assign(v, expr.const(arr.ravel().tolist(), tuple(arr.shape), dt))
    return v
v_by_name = {'cache_position': mkvar(cp, MNN.Halide_Type_Int), 'inputs_embeds': mkvar(emb, MNN.Halide_Type_Float), 'kv_k': mkvar(kvk, MNN.Halide_Type_Float), 'kv_v': mkvar(kvv, MNN.Halide_Type_Float), 'position_ids': mkvar(pos, MNN.Halide_Type_Float)}
feed_vars = [v_by_name[n] for n in order]
t0 = time.time()
res = mod.onForward(feed_vars)
t1 = time.time()
out = res[0].read()
mh = np.asarray(out).reshape(-1).astype(np.float32)
print('Module elapsed:', round(t1-t0,2), 'elems', mh.size, 'maxabs', float(np.abs(mh).max()), flush=True)
ref = np.load(D + '/gb_ref_hidden_fp32.npy').ravel()
n = min(len(mh), len(ref)); a = mh[:n].astype(np.float64); b = ref[:n].astype(np.float64)
c = float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-9))
print('PC-Module(MNN-order) vs ORT-ref: cosine', round(c,6), flush=True)