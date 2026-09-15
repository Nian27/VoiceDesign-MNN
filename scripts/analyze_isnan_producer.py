import onnx, numpy as np
p = r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/m2/tokenizer_decoder.onnx'
m = onnx.load(p, load_external_data=False)
g = m.graph
# map output->node and initializer values
prod = {}
for n in g.node:
    for o in n.output: prod[o] = n
inits = {i.name: i for i in g.initializer}
def describe(name, depth=0):
    if name in inits:
        arr = onnx.numpy_helper.to_array(inits[name])
        print('  '*depth + f'{name}: INITIALIZER shape={arr.shape} dtype={arr.dtype} val={arr.ravel()[:8]}')
        return
    if name in prod:
        n = prod[name]
        print('  '*depth + f'{name}: from {n.op_type} inputs={list(n.input)[:4]}')
        if depth < 2 and n.op_type not in ('Constant',):
            for i in n.input[:4]:
                describe(i, depth+1)
    else:
        gv = [vi for vi in g.value_info if vi.name == name]
        print('  '*depth + f'{name}: GRAPH INPUT/value_info' + (f' shape={gv[0].type.tensor_type.shape}' if gv else ''))
print('=== val_137 provenance ===')
describe('val_137')
print('=== val_367 (first IsNaN input) provenance ===')
describe('val_367')
# also check the Where semantics: Where(IsNaN, val_137, val_367) => nan replaced by val_137
print('=== count ===')
isnan = [n for n in g.node if n.op_type == 'IsNaN']
print('IsNaN nodes:', len(isnan))
wheres = [n for n in g.node if n.op_type == 'Where']
print('Where nodes:', len(wheres))