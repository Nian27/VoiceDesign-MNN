import onnxruntime as ort
p = r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/m22/onnx/graphb_dynamic_fp32_trad.onnx'
try:
    sess = ort.InferenceSession(p, providers=['CPUExecutionProvider'])
    print('ORT LOAD OK')
    for i in sess.get_inputs(): print('IN', i.name, i.shape, i.type)
except Exception as e:
    print('ORT LOAD FAIL:', str(e)[:300])
print('TEST_DONE', flush=True)