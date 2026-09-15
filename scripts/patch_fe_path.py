import io
p = r'E:/AndroidStudioProjects/qwen3tts-mnn/scripts/export/vd_g2b_cpu_margin.py'
s = io.open(p, encoding='utf-8').read()
old = "so_fe = ort.InferenceSession(OUT + '/frame_emb.onnx', providers=['CPUExecutionProvider'])"
new = "so_fe = ort.InferenceSession(D + '/vd_m5b/frame_emb.onnx', providers=['CPUExecutionProvider'])"
n = s.count(old)
s = s.replace(old, new)
io.open(p, 'w', encoding='utf-8').write(s)
print('patched', n)
import os; print('exists:', os.path.exists(r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/device-smoke/vd_m5b/frame_emb.onnx'))