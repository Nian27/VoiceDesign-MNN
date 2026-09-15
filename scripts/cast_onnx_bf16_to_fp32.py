import onnx, numpy as np, os, sys
src = sys.argv[1]
dst = sys.argv[2]
m = onnx.load(src)
g = m.graph
n = 0
from onnx import numpy_helper
for i in g.initializer:
    arr = numpy_helper.to_array(i)
    if arr.dtype == np.float16 or i.data_type == 16:
        i.CopyFrom(numpy_helper.from_array(arr.astype(np.float32), name=i.name))
        n += 1
print('cast', n, 'initializers')
onnx.save_model(m, dst, save_as_external_data=True, all_tensors_to_one_file=True, location=os.path.basename(dst) + '.data', size_threshold=1024)
print('saved', dst)