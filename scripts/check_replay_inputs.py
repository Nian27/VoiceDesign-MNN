import MNN
D = r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/m22/onnx'
interp = MNN.Interpreter(D + '/graphb_replay_step_nonan.mnn')
interp.setExternalFile(D + '/graphb_replay_step_nonan.mnn.weight')
s = interp.createSession()
ins = interp.getSessionInputAll(s)
for k, v in ins.items():
    print('in:', k, v.getShape())
outs = interp.getSessionOutputAll(s)
for k, v in outs.items():
    print('out:', k, v.getShape())