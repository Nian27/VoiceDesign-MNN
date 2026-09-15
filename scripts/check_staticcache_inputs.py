import onnx
m = onnx.load(r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/device-smoke/graphb_staticcache.onnx', load_external_data=False)
print('inputs:')
for i in m.graph.input:
    tt = i.type.tensor_type
    dims = [d.dim_value if d.HasField('dim_value') else (d.dim_param if d.HasField('dim_param') else '?') for d in tt.shape.dim]
    print(' ', i.name, dims, 'elem', tt.elem_type)
print('outputs:')
for o in m.graph.output:
    tt = o.type.tensor_type
    dims = [d.dim_value if d.HasField('dim_value') else (d.dim_param if d.HasField('dim_param') else '?') for d in tt.shape.dim]
    print(' ', o.name, dims, 'elem', tt.elem_type)
import onnxruntime as ort
sess = ort.InferenceSession(r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/device-smoke/graphb_staticcache.onnx', providers=['CPUExecutionProvider'])
print('ORT session OK')
for i in sess.get_inputs(): print('ORT IN', i.name, i.shape, i.type)