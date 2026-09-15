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
# 用 Interpreter 的 createFromSession? 更直接：用 expr.load_module with Var list
def mkvar(arr, dt):
    v = expr._Input(tuple(arr.shape), MNN.Tensor_DimensionType_Caffe, dt)
    expr.assign(v, expr.const(arr.ravel().tolist(), tuple(arr.shape), dt))
    return v
v_by_name = {'cache_position': mkvar(cp, MNN.Halide_Type_Int), 'inputs_embeds': mkvar(emb, MNN.Halide_Type_Float), 'kv_k': mkvar(kvk, MNN.Halide_Type_Float), 'kv_v': mkvar(kvv, MNN.Halide_Type_Float), 'position_ids': mkvar(pos, MNN.Halide_Type_Float)}
feed_vars = [v_by_name[n] for n in order]
# 从 interpreter 创建 module：MNN.nn.load_module 需要 Var——用 expr._get_or_create? 改用官方路径: Module::load via pymnn 的 nn.load_module([in_vars], [out_var], False)? 第3参 bool=is_training?
# 直接试：load_module(inputs=[inVars], outputs=[outVar], False) 需要先有 output Var——改用 Interpreter session 方式跑（已验证 cos1.0），此处仅确认顺序打印
print('feed order:', order)
print('SKIP_MODULE_ONFORWARD_PC (device probe will confirm)')