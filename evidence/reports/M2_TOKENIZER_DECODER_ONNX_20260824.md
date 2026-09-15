# M2 Tokenizer-12Hz Decoder ONNX 导出报告（2026-08-24）

## 结论

Qwen3-TTS-Tokenizer-12Hz 的 decoder（code->PCM 24kHz）已成功导出 ONNX——本工作区第一个 NPU 就绪制品：
codec 式解码路径（CausalConvNet + DecoderTransformer + RVQ + upsample），与 audio8 已验证的
MNN-QNN/HTP 路线（PCM cosine 0.999098 / RTF 0.32）同构。

## 合同（冻结）

```text
输入 codes : Int64 (B, Q=16, T_frames)   [每帧 16 个 codebook 层]
输出 waveform: Float32 (B, 1, T_frames * 1920)  [24kHz]
全量 decode == chunked_decode(chunk_size=300, left_context_size=25)  [逐帧相等已验证]
```

## 验证

- 输入 (1,16,300) → 输出 (1,1,576000)（300x1920），全部 finite，maxabs=0.95（clamp [-1,1]）
- chunked==full（allclose atol=1e-4）→ 流式/切块契约成立，Android 长句可切块
- PyTorch fp32 参考与 golden_io.pt 一致

## 产物

```text
artifacts/m2/tokenizer_decoder.onnx       2.7MB  (opset18, dynamo exporter)
artifacts/m2/tokenizer_decoder.onnx.data  456MB  (fp32 权重 external data)
artifacts/m2/golden_io.pt                 2.3MB  (codes + wav 黄金 IO)
```

## 环境改动

- venv 安装 onnxscript 0.7.1 + onnx_ir 1.0.0（dynamo exporter 需要）
- 旧版 torchscript exporter 报 invalid unordered_map key → 弃用，走默认 dynamo 导出

## Gate

| Gate | 结果 |
|---|---|
| decoder 前向（fp32, CUDA） | PASS（finite / maxabs 0.95 / shape 正确） |
| chunked==full | PASS（streaming 契约成立） |
| ONNX 导出 | PASS（opset18 + external data） |
| ONNX 数值对照 | PASS（ORT CPU：cosine 0.999996 / SI-SDR 50.6dB / maxabs 0.020 / mae 1.3e-4） |

## 下一步

1. ONNX Runtime 加载校验 + 与 PyTorch 输出 cosine 对照（M2 Gate 补全）
2. MNN 转换：MNNConvert（fp16）+ 与 ONNX cosine 对照（M3）
3. MNN-QNN offline → HTP V81 context → SM8850 真机（M4，照 audio8 管线）
4. talker(28L) + code_predictor(5L) ONNX 导出（M2 继续，LLM 部分走 Road B 分治）

## M2 Gate 补全（2026-08-24 第二轮）

- ORT CPUExecutionProvider 加载 tokenizer_decoder.onnx，输入 codes (1,16,300) → waveform (1,1,576000)，elapsed 5.54s（PC CPU，热态 RTF 未测）。
- 对照 golden_io.pt：cosine 0.999996 / SI-SDR 50.6dB / maxabs 0.020107 / mae 1.33e-4 —— 超过 audio8 参考 Gate（0.999098 / 27.43dB）。
- 动态形状正常（frames 动态，输出形状表达式可见 1920x upsample 链）。
