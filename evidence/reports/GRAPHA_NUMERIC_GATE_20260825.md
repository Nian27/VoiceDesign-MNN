# GraphA prefill 数值 Gate 报告（2026-08-25 RESOLVED）

## 结论

**GraphA(talker prefill, L=34) MNN 转换正确：fp32 cos=1.0 / fp16 cos=0.999983（vs ORT 同 raw 参照）。**

## 纠偏记录（假失败教训）

- 早期报 '1e18 损坏' 是**对照脚本 bug**：np.random.seed(13) 重新生成 emb 的顺序与参照 npy 不一致 → 输入不同步。
- 用同一份 raw 文件（grapha_emb_f32.raw/grapha_pos_f32.raw）喂 MNN 后立即 cos≥0.99998。
- 强化纪律：任何对照必须同 raw 同源；生成参照与喂入共用一个文件。

## 制品

```text
artifacts/device-smoke/grapha_prefill_static_fp32.mnn(+.weight)  cos 1.0
artifacts/device-smoke/grapha_prefill_static_fp16.mnn(+.weight)  cos 0.999983
scripts/export/grapha_recheck.py / grapha_recheck_fp32.py
```

## 状态

GraphA 冻结 PASS。talker 两段（prefill + decode-step）PC 数值全绿。