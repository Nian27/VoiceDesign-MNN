import onnx, numpy as np, onnxruntime as ort
import onnx.utils
D = r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/device-smoke'
m = onnx.load(D + '/grapha_prefill_static_nonan.onnx', load_external_data=False)
g = m.graph
# 找 decoder 层结构: layer0 的 RMSNorm/attention 等关键中间张量名
# 策略: 在每层输入 hidden_states 处插入 Identity 探针输出，跑 ORT 记录中间值
# 先列出前 60 个节点名+op 类型，理解层结构
for i, n in enumerate(g.node[:80]):
    print(i, n.op_type, '|', list(n.input)[:2], '->', list(n.output)[:1])