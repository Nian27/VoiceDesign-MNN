import onnx, numpy as np
p = r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/m2/tokenizer_decoder.onnx'
m = onnx.load(p, load_external_data=False)
g = m.graph
print('total nodes:', len(g.node))
for i, n in enumerate(g.node):
    if n.op_type in ('IsNaN', 'Where', 'If', 'NonZero', 'Equal', 'Not', 'Cast', 'Greater', 'Less'):
        inp = list(n.input); out = list(n.output)
        print(i, n.op_type, 'in=', inp[:4], 'out=', out[:2])
# find consumers of IsNaN output
isnan_outs = [n.output[0] for n in g.node if n.op_type == 'IsNaN']
for i, n in enumerate(g.node):
    if any(x in n.input for x in isnan_outs):
        print('CONSUMER', i, n.op_type, 'in=', list(n.input)[:4], 'out=', list(n.output)[:2])