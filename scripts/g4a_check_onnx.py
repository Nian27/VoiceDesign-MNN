import onnx
from onnx import checker
p = r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/m22/onnx/graphb_dynamic_fp32_trad.onnx'
m = onnx.load(p, load_external_data=False)
print('ir', m.ir_version, 'opsets', [(o.domain, o.version) for o in m.opset_import])
try:
    checker.check_model(m)
    print('CHECKER OK')
except Exception as e:
    print('CHECKER FAIL:', str(e)[:300])
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
print('CHECK_DONE', flush=True)