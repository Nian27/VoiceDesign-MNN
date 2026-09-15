# M3-GraphB 数值 Gate 报告（2026-08-25）

## 结论

**PASS — GraphB (talker 28L decode-step) MNN vs ORT: cosine 1.0 / mae 9.03e-06 / maxabs 4.5e-05**

VoiceDesign 链路两大核心件全部打通 PyTorch→ONNX→MNN 且数值达标：
1. tokenizer-decoder (code→24kHz PCM): cos 0.999936 / SI-SDR 38.2dB
2. GraphB talker decode-step (28L full-attn + mrope + GQA): cos 1.0

## 制品

```text
artifacts/m22/onnx/graphb_replay_fp32ex_nonan.onnx   干净 fp32 源（strip 28 IsNaN 后）
artifacts/m22/onnx/graphb_replay_fp32ex_nonan.mnn    483KB 图
artifacts/m22/onnx/graphb_replay_fp32ex_nonan.mnn.weight  5.6GB fp32 权重
scripts/export/export_graphb_fp32ex.py               PyTorch fp32 直接导出（无 cast 混型）
scripts/export/verify_m33_graphb.py                  双引擎对照脚本
```

## 根因链（cosine 0.014 → 1.0 的修复过程）

1. bf16 导出 → MNN 'weight bits=16' 不落盘 → cast fp32 但内部仍混型 → ORT 拒载
2. 根治 = PyTorch dtype=float32 重新导出（export_graphb_fp32ex.py, dynamo exporter）
3. ORT 加载 PASS 后 MNN 仍 cosine 0.014 全错 → cache_position dtype 契约分裂：
   **ORT 要 int64、MNN 要 int32** —— 双份输入（同逻辑值不同 numpy dtype）后 cosine=1.0
4. 附带确认：MNN 会把 ONNX int64 输入静默转 int32；'Input type not match' 警告即此症状

## Gate

| Gate | 结果 |
|---|---|
| ORT(fp32ex 干净源) 前向 | PASS（finite, 0.4s） |
| MNN(fp32ex_nonan) 前向 | PASS（0.18s, PC CPU） |
| MNN vs ORT cosine | **PASS 1.0** |

## 下一步

- GraphA(prefill) 同法导出+Gate（补全 talker 两段）
- code_predictor(5L) 导出+Gate（15 码小 LM）
- 真机 SM8850 HTP 四证（audio8 WSL 路线）+ 最小 app 真测（用户指令）