import MNN
ART = r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/m2'
interp = MNN.Interpreter(ART + '/tokenizer_decoder_nonan.mnn')
sess = interp.createSession()
it = interp.getSessionInput(sess)
print('input shape:', it.getShape())
print('input dtype code:', it.getDataType())
print('halide consts:', [n for n in dir(MNN) if n.startswith('Halide_Type')])
ot = interp.getSessionOutput(sess)
print('output shape:', ot.getShape())