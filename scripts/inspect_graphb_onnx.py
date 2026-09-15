import onnx
m = onnx.load('E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/m22/onnx/graphb_replay_step_fp32_nonan.onnx')
print('=== inputs ===')
for i in m.graph.input:
    tt = i.type.tensor_type
    dims = [d.dim_value if d.HasField('dim_value') else (d.dim_param if d.HasField('dim_param') else '?') for d in tt.shape.dim]
    print(i.name, dims, 'elem_type=', tt.elem_type)
print('=== outputs ===')
for o in m.graph.output:
    tt = o.type.tensor_type
    dims = [d.dim_value if d.HasField('dim_value') else (d.dim_param if d.HasField('dim_param') else '?') for d in tt.shape.dim]
    print(o.name, dims, 'elem_type=', tt.elem_type)