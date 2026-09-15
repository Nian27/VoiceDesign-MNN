# M2.2 GraphB talker decode-step ONNX 导出报告（2026-08-25）

## 结论

#### GRAPH_B_REPLAY_EXPORT_OK — talker 28L 单步 decode ONNX 导出成功

- graphb_replay_step.onnx（图 4.7MB）+ 权重 external data 2.8GB
- 输入契约（G1 golden 实证）：inputs_embeds (1,1,2048) bf16；position_ids (3,1,1) f32 mrope；
  kv_k/kv_v 堆叠单张量 (28,B,H,T,128)（28L, GQA 8 heads）；cache_position (1,)
- 输出：hidden_states (1,1,2048) + 层 hidden

## 导出修复链（F-Q-NPU-20260825-01，防重踩）

1. 输入 dtype 必须 bf16（模型权重 bf16，fp32 报 mismatch）
2. mrope 的 ComplexDouble 只能在 dynamo exporter（默认）下导出，legacy torchscript 拒绝
3. dynamo=True 时用 dynamic_shapes 而非 dynamic_axes
4. dynamic_shapes 键集合必须 = 全部 forward 参数名（漏 cache_position 报 mismatch）
5. kv_k/kv_v 是 28 张量 list，dynamo 无法描述 → 堆叠为单张量 (28,B,H,T,128)，forward 按 dim0 切片

## 关键发现: RoPE 已被 dynamo 拆成 Cos/Sin 算子

- 图 ops: Mul/Add/MatMul/Cast/Reshape/Slice/Concat/Transpose/Pow/ReduceMean/Sqrt/Gather/Softmax/Sigmoid + Cos/Sin/Range/LessOrEqual/ScatterND
- 没有 FusedRoPE(type-306) —— 正好绕开 Qwen3.5 spike 里 MNN converter 的已知拒绝点
- 28 个 IsNaN（28L softmax 防 NaN 守卫）已用 strip_isnan_generic.py 移除（56 节点，2723 剩余）

## 产物

```text
artifacts/m22/onnx/graphb_replay_step.onnx            原始（4.7MB + 2.8GB data）
artifacts/m22/onnx/graphb_replay_step_nonan.onnx      去 IsNaN（4.6MB）
scripts/export/export_graphb_replay.py                导出脚本（堆叠 KV 形态）
scripts/export/strip_isnan_generic.py                 通用 IsNaN 守卫移除工具
```

## 下一步

- M3-GraphB: MNNConvert（后台运行中 pwsh-81）→ pymnn 数值对照 vs golden
- 之后 GraphA(prefill) 导出 + tokenizer-decoder MNN 已 PASS（M3）