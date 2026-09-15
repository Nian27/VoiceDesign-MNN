import onnx, onnxruntime as ort
m = onnx.load(r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/device-smoke/graphb_deploy_hand.onnx', load_external_data=False)
print('inputs:')
for i in m.graph.input:
    tt = i.type.tensor_type
    dims = [d.dim_value if d.HasField('dim_value') else (d.dim_param if d.HasField('dim_param') else '?') for d in tt.shape.dim]
    print(' ', i.name, dims)
print('outputs:')
for o in m.graph.output:
    tt = o.type.tensor_type
    dims = [d.dim_value if d.HasField('dim_value') else (d.dim_param if d.HasField('dim_param') else '?') for d in tt.shape.dim]
    print(' ', o.name, dims)
sess = ort.InferenceSession(r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/device-smoke/graphb_deploy_hand.onnx', providers=['CPUExecutionProvider'])
print('ORT session OK, cache_position present:', any(i.name=='cache_position' for i in sess.get_inputs()))