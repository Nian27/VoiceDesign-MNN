import MNN
ART = r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/m22/onnx'
interp = MNN.Interpreter(ART + '/graphb_replay_fp32ex_nonan.mnn')
sess = interp.createSession()
inputs = interp.getSessionInputAll(sess)
outputs = interp.getSessionOutputAll(sess)
def desc(t):
    try: dt = t.getDataType()
    except Exception as e: dt = 'ERR:' + str(e)[:40]
    return {'shape': t.getShape(), 'dtype': str(dt), 'dimtype': str(t.getDimensionType())}
for name, t in inputs.items(): print('IN ', name, desc(t))
for name, t in outputs.items(): print('OUT', name, desc(t))