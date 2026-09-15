import onnx, numpy as np, os, sys
src = sys.argv[1]
dst = sys.argv[2]
m = onnx.load(src, load_external_data=False)
g = m.graph
nodes = list(g.node)
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
            a_name = ins[1]
            b_name = ins[2]
            a_init = [x for x in g.initializer if x.name == a_name]
            a_is_zero = False
            if a_init:
                arr = onnx.numpy_helper.to_array(a_init[0])
                a_is_zero = bool(np.all(arr == 0))
            if a_is_zero and idx == 0:
                renames[w.output[0]] = b_name
                removed_idx.add(j)
                removed_idx.add(i)
            break
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
onnx.save_model(m, dst, save_as_external_data=True, all_tensors_to_one_file=True, location=os.path.basename(dst) + '.data', size_threshold=1024)
print('STRIPPED', len(removed_idx), 'remaining', len(new_nodes))