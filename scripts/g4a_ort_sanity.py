import numpy as np, onnxruntime as ort
D = r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/m22/onnx'
sess = ort.InferenceSession(D + '/graphb_replay_step_fp32_nonan.onnx', providers=['CPUExecutionProvider'])
for i in sess.get_inputs():
    print('IN', i.name, i.shape, i.type)
for o in sess.get_outputs():
    print('OUT', o.name, o.shape, o.type)
try:
    import ml_dtypes
    print('ml_dtypes OK', ml_dtypes.bfloat16)
except Exception as e:
    print('ml_dtypes MISSING:', str(e)[:100])
print('ORT_DONE', flush=True)