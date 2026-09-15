# GraphB 真机数值 Gate 报告（2026-08-25 RESOLVED）

## 结论

**GraphB(talker 28L decode-step) 在 SM8850 真机数值 PASS：cosine 1.0（vs ORT-ref，mae 9e-05 / maxabs 0.00114 / rel_l2 3.3e-05）。**

API isolation A/B 定论：同模型/同权重/同 raw/同设备/同 libMNN，仅替换 feed 接口
（Interpreter+CAFFE host tensor → Express Module）即从 cos 0.007 到 1.0。

## 根因（双因）

1. **输入顺序**：MNN 模型输入顺序 = [cache_position, inputs_embeds, kv_k, kv_v, position_ids]，
   与 ONNX 不同（cache_position 排第一）。按 ONNX 顺序喂入全部错位。
2. **feed 契约**：Interpreter + Tensor(t,CAFFE)+memcpy 对 5 输入不可靠；
   Express Module（_Input+writeMap+onForward）正确。
3. CRC32 校验全程通过：raw 数据 PC→push→设备读入 完全一致（排除文件支线）。

## 证据链

```text
raw[0] cache_position bytes=4    crc32=2b7fb8a9
raw[1] inputs_embeds   bytes=8192     crc32=541c327c
raw[2] kv_k            bytes=4014080  crc32=7b9d286f
raw[3] kv_v            bytes=4014080  crc32=cd8ad29a
raw[4] position_ids    bytes=12       crc32=060d8eea
Module-DEV vs ORT-ref : cosine 1.0 / mae 9e-05 / maxabs 0.00114
Module-DEV vs PC-MNN  : cosine 1.0
run_ms = 563 (CPU 4 线程真机基线), warm 9.97s (fp16 权重首载)
```

## 工具定案

- 正式工具：qwen3tts_module_probe（Express Module API + MNN 顺序 inputNames + CRC32 校验）
- 废弃：qwen3tts_probe（Interpreter + CAFFE feed）
- 输入 CRC32 校验成为 probe 标配（防文件支线）

## GraphB 五证（冻结）

| Gate | 结果 |
|---|---|
| conversion (PC-MNN vs ORT) | cos 1.0 |
| Android load / external weight | PASS |
| execute / deterministic | PASS |
| numeric (Module) | cos 1.0 |
| performance | run 563ms CPU |

## 下一步

- GraphA MNNConvert FIRST_BAD_OP 定位（fp16/fp32 转换均输出 1e18 损坏，独立问题 F-...-02）
- 之后：code_predictor 导出 + 真机闭环（prefill→decode→predictor→decoder→PCM）