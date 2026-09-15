import onnx, numpy as np, os
ART = r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/m2'
p = os.path.join(ART, 'tokenizer_decoder.onnx')
m = onnx.load(p, load_external_data=False)
g = m.graph
nodes = list(g.node)
# find IsNaN nodes and consumer Where(IsNaN_out, zero_const, x)
removed_idx = set()
renames = {}
for i, n in enumerate(nodes):
    if n.op_type != 'IsNaN':
        continue
    isn_out = n.output[0]
    for j, w in enumerate(nodes):
        if w.op_type == 'Where' and isn_out in list(w.input) and j not in removed_idx:
            ins = list(w.input)
            idx = ins.index(isn_out)
            a_name = ins[1]; b_name = ins[2]
            a_init = [x for x in g.initializer if x.name == a_name]
            a_is_zero = False
            if a_init:
                arr = onnx.numpy_helper.to_array(a_init[0])
                a_is_zero = bool(np.all(arr == 0))
            # Where(IsNaN(x), 0, x) -> replace output with x (b_name); guard only when a is zero const
            if a_is_zero and idx == 0:
                renames[w.output[0]] = b_name
                removed_idx.add(j)
                removed_idx.add(i)
            break
print('IsNaN found:', sum(1 for n in nodes if n.op_type == 'IsNaN'))
print('removed node idx:', sorted(removed_idx))
new_nodes = []
for i, n in enumerate(nodes):
    if i in removed_idx:
        continue
    n2 = onnx.NodeProto()
    n2.CopyFrom(n)
    for k in range(len(n2.input)):
        if n2.input[k] in renames:
            n2.input[k] = renames[n2.input[k]]
    new_nodes.append(n2)
del g.node[:]
g.node.extend(new_nodes)
op = os.path.join(ART, 'tokenizer_decoder_nonan.onnx')
onnx.save_model(m, op, save_as_external_data=True, all_tensors_to_one_file=True, location='tokenizer_decoder_nonan.onnx.data', size_threshold=1024)
print('saved', op)
m2 = onnx.load(op, load_external_data=False)
print('nodes after:', len(m2.graph.node), 'IsNaN left:', sum(1 for n in m2.graph.node if n.op_type == 'IsNaN'))