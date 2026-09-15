# VD-M5A-0: dump graphb_deploy_hand_fp16.mnn real input/output contract (name, order, shape, dtype)
import MNN, hashlib, os
D = r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/device-smoke'
mp = D + '/graphb_deploy_hand_fp16.mnn'
wp = D + '/graphb_deploy_hand_fp16.mnn.weight'
# SHA256
for p in (mp, wp):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1<<20), b''):
            h.update(chunk)
    print(os.path.basename(p), 'size', os.path.getsize(p), 'sha256', h.hexdigest())
interp = MNN.Interpreter(mp)
interp.setExternalFile(wp)
s = interp.createSession()
# session inputs in order
inputs = interp.getSessionInputAll(s)
print('n_inputs', len(inputs))
for i, t in enumerate(inputs.items() if hasattr(inputs, 'items') else inputs):
    name, ten = (t[0], t[1]) if hasattr(t, '__len__') and len(t)==2 else (None, t)
    print('IN[%d] name=%s shape=%s' % (i, name, str(ten.getShape())))
outputs = interp.getSessionOutputAll(s)
print('n_outputs', len(outputs))
for i, t in enumerate(outputs.items() if hasattr(outputs, 'items') else outputs):
    name, ten = (t[0], t[1]) if hasattr(t, '__len__') and len(t)==2 else (None, t)
    print('OUT[%d] name=%s shape=%s' % (i, name, str(ten.getShape())))
