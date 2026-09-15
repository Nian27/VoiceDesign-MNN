import onnx, numpy as np, onnxruntime as ort
import onnx.utils
D = r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/device-smoke'
np.random.seed(13)
emb = (np.random.randn(1, 34, 2048) * 0.1).astype(np.float32)
pos = np.zeros((3, 1, 34), dtype=np.float32)
for j in range(34): pos[0,0,j]=j; pos[1,0,j]=j; pos[2,0,j]=j
m = onnx.load(D + '/grapha_prefill_static_nonan.onnx', load_external_data=False)
all_outs = set()
for n in m.graph.node: all_outs.update(n.output)
targets = ['mul_3', 'view_1', 'add_4', 'add_5']
for t in targets: print('target', t, 'present:', t in all_outs)
def probe(t):
    if t not in all_outs: return
    try:
        sub_path = D + '/probe_' + t + '.onnx'
        onnx.utils.extract_model(D + '/grapha_prefill_static_nonan.onnx', sub_path, ['inputs_embeds','position_ids'], [t])
        so = ort.InferenceSession(sub_path, providers=['CPUExecutionProvider'])
        out = so.run(None, {'inputs_embeds': emb, 'position_ids': pos})[0]
        print(t, 'ORT shape', out.shape, 'maxabs', round(float(np.abs(out).max()),4), 'finite', bool(np.isfinite(out).all()))
        sub_path = D + '/probe_' + t + '.onnx'
        onnx.save(sub, sub_path)
        so = ort.InferenceSession(sub_path, providers=['CPUExecutionProvider'])
        out = so.run(None, {'inputs_embeds': emb, 'position_ids': pos})[0]
        print(t, 'ORT shape', out.shape, 'maxabs', round(float(np.abs(out).max()),4), 'finite', bool(np.isfinite(out).all()))
    except Exception as e:
        print(t, 'ERR', str(e)[:150])
for t in targets: probe(t)