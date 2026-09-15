# M3 MNN 数值 Gate 报告（2026-08-24）

## 结论

Tokenizer-12Hz decoder 已完成 全链 PyTorch → ONNX → MNN 3.6.1 验证：
- ONNX 转换成功（图手术去掉 IsNaN 后）
- MNN fp32 .mnn 228MB 成功生成
- pymnn 3.6.1 CPU 运行 vs PyTorch golden：**cosine 0.999936 / SI-SDR 38.18dB / maxabs 0.067**（audio8 黄金 Gate：0.999098 / 27.43dB，均超过）

## 踩坑记录（防再踩）

1. ONNX 图含 8 处 IsNaN+Where 防 NaN 守卫（softmax 链自动导出），MNN 转换器不支持 IsNaN →
   scripts/export/strip_isnan_where.py 图手术删除（Where(IsNaN(x),0,x) → x 直通），1196 节点剩余。
2. MNN 输入契约 = **INT32** 且动态 frames (1,16,-1)（与 audio8 报告一致），golden 是 int64 →
   喂入前必须 astype(int32) + resizeTensor + resizeSession，否则报 input type mismatch 且数值全错。
3. pymnn 3.6.1 是新 pybind 绑定：copyFromHostTensor/copyToHostTensor/getNumpyData 在 Tensor 上；
   resizeTensor + resizeSession 成对调用；输出宿主 tensor 用 (shape, Halide_Type_Float, DimType_Caffe)。

## 产物

```text
artifacts/m2/tokenizer_decoder_nonan.onnx(+.data)   图手术后的 ONNX（1196 节点）
artifacts/m2/tokenizer_decoder_nonan.mnn            MNN 模型 228MB fp32
scripts/export/strip_isnan_where.py                 图手术工具
scripts/export/verify_m3_mnn.py                     pymnn 数值验证脚本
```

## Gate

| Gate | 结果 |
|---|---|
| ONNX 转换（去 IsNaN 后） | PASS |
| MNN CPU 数值（cosine/SI-SDR） | PASS（0.999936 / 38.18dB） |
| MNN CPU 性能（PC 6.7s/576k 样本） | 记录（端侧以真机为准） |

## 下一步

- M3b：fp16 MNN 转换 + 数值对照（228→114MB）
- M4：MNN-QNN offline → HTP V81 context → SM8850 真机（QAIRT 2.49 工具已确认在 v2.49.0.260730/qairt）
- M2 剩余：talker(28L) + code_predictor(5L) ONNX 导出（Road B 分治）