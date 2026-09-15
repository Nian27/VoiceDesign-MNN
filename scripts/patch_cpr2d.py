import io
p = r'E:/AndroidStudioProjects/qwen3tts-mnn/scripts/export/vd_cpr2_clean.py'
s = io.open(p, encoding='utf-8').read()
old = '''    ty = t.getType()
    if ty.code == MNN.Halide_Type_Int.code if hasattr(ty,'code') else False:
        pass
    flat = np.ascontiguousarray(arr).ravel()
    is_int = (flat.dtype == np.int32)
    ht = MNN.Tensor(shp, MNN.Halide_Type_Int if is_int else MNN.Halide_Type_Float,
                    flat.astype(np.int32 if is_int else np.float32), MNN.Tensor_DimensionType_Caffe)
    t.copyFromHostTensor(ht)'''
new = '''    flat = np.ascontiguousarray(arr, np.float32).ravel()
    ht = MNN.Tensor(shp, MNN.Halide_Type_Float, flat, MNN.Tensor_DimensionType_Caffe)
    t.copyFromHostTensor(ht)'''
n = s.count(old)
s = s.replace(old, new)
io.open(p, 'w', encoding='utf-8').write(s)
print('feed simplified, replaced=%d' % n)