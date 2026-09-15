import onnx, numpy as np
p = r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/m22/onnx/graphb_replay_step.onnx'
m = onnx.load(p, load_external_data=False)
g = m.graph
print('total nodes:', len(g.node))
from collections import Counter
ops = Counter(n.op_type for n in g.node)
print('ops:', dict(ops.most_common(50)))
isnan = [n for n in g.node if n.op_type == 'IsNaN']
print('IsNaN:', len(isnan))