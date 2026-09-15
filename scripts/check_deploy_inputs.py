import onnx
m = onnx.load(r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/device-smoke/graphb_deploy_m768.onnx', load_external_data=False)
for i in m.graph.input:
    tt = i.type.tensor_type
    dims = [d.dim_value if d.HasField('dim_value') else (d.dim_param if d.HasField('dim_param') else '?') for d in tt.shape.dim]
    print(i.name, dims, 'elem', tt.elem_type)