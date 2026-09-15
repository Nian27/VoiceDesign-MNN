# HTP 支线 first-divergence 定位报告（2026-08-26）

## 结论（已定位到单算子）

    first_divergence_tensor = q_rope / k_rope     (apply_multimodal_rotary_pos_emb 的输出)
    first_divergence_op     = interleaved 多模态 RoPE 中的「步长=3 跨步切片读 + 跨步写」

## 方法与 Gate

### A0 受控诊断图 GraphB-1L-DIAG
只含 layer 0，把 16 个 boundary 全部暴露成**正式 graph outputs**（绕开已知损坏的 mapped-pointer dump）：
    norm_out, q_raw, k_raw, v_raw, q_norm, k_norm, q_rope, k_rope,
    attn_out, o_proj_out, res1, post_norm, mlp_out, layer0_out, kv_k_out, kv_v_out
产物：artifacts/device-smoke/vd_p1/graphb_1l_diag_fp16.mnn (100MB, 权重内嵌)

### A1 CPU Gate = PASS（诊断图未改变语义）
device CPU 16/16 boundary 与 torch 参考逐一吻合；且 kv_k_out maxabs=258.413422 与
先前 full-GraphB CPU 的 258.413422 逐位一致 -> 诊断图可信。

### A2 HTP 结果（同一张图，只换 backend）

| boundary | CPU_cos | HTP_cos | CPU_maxabs | HTP_maxabs |
|---|---|---|---|---|
| norm_out | 1.0000000 | 1.0000000 | 0.7240 | 0.7241 |
| q_raw | 1.0000000 | 1.0000000 | 2.6682 | 2.6680 |
| k_raw | 1.0000000 | 1.0000000 | 2.2433 | 2.2441 |
| v_raw | 1.0000000 | 1.0000000 | 0.5850 | 0.5850 |
| q_norm | 1.0000000 | 1.0000000 | 21.3237 | 21.3281 |
| k_norm | 1.0000000 | 1.0000000 | 254.8007 | 254.7500 |
| **q_rope** | 0.9999856 | **0.5777542** | 16.1512 | 13.1953 |
| **k_rope** | 0.9999990 | **0.4048883** | 254.8006 | 129.8750 |
| attn_out | 0.9999997 | 0.6858982 | 0.5850 | 0.5679 |
| o_proj_out | 0.9999997 | 0.8108326 | 1.7165 | 0.8613 |
| res1 | 0.9999997 | 0.8099947 | 1.7291 | 0.8647 |
| post_norm | 0.9999990 | 0.7533144 | 1.0851 | 2.4336 |
| mlp_out | 0.9999986 | 0.7451856 | 0.5887 | 1.5762 |
| layer0_out | 0.9999996 | 0.6936184 | 1.9037 | 1.8281 |
| kv_k_out | 1.0000000 | 0.0289565 | 258.4134 | 129.8750 |
| kv_v_out | 1.0000000 | 0.0010162 | 138.8099 | 0.5850 |

**上游 6 个 boundary（norm / QKV proj / q_norm / k_norm）在 HTP 上逐位正确（cos=1.0000000）**
**从 q_rope 起全部错误**

### 原始值铁证
    k_rope CPU[:6] = [-0.1624 -0.3863  0.4406 -1.4793  0.9855  0.1622]
    k_rope HTP[:6] = [-0.      0.      0.439   0.      0.      0.1671]
    k_rope REF[:6] = [-0.1624 -0.3865  0.4433 -1.5455  0.9836  0.1703]
  索引 2、5 非零（落在步长序列内），索引 0/1/3/4 被清零 -> 跨步写入只覆盖了部分 lane

## 机制（高置信假设）

    mrope_section = [24, 20, 20]   -> head_dim 128, modality_num = 3
    interleaved 分支:
        x_t = x[0].clone()
        for i, n in enumerate(mrope_section[1:], 1):
            beg = i; end = n * 3
            x_t[..., beg:end:3] = x[beg, ..., beg:end:3]
    即 ONNX 侧大概率落成 Slice(steps=3) + ScatterND/跨步赋值。
    HTP 对该模式 lowering/执行有误 -> 部分 lane 未写入（保留 clone 值或零）
    -> cos/sin 表错 -> RoPE 输出错 -> 下游全错（与实测一致）

## 排除项（本轮新增）
- 不是 QKV 投影 / LayerNorm / q_norm / k_norm（这些在 HTP 上逐位正确）
- 不是 htpops 缺失（修复后数值不变）
- 不是 attention 本体（上游已错，attention 只是受害者）
- **不是 5D 问题**：first bad op 是 cos/sin 的跨步索引构造，与 KV 的 5D 形状无关
  -> 因此不执行原计划的 C（KV 5D->4D），方向应放在 rope 算子

## 下一步（P2）
1. **最快确认实验**：把 interleaved cos/sin 改为**host 侧预计算**并作为 graph 输入传入
   （decode 单步只有 1 个位置，cos/sin 只是 1x128 向量）
   -> 若 HTP 随即变正确，则该算子被 100% 坐实
2. 或改写 mrope 为 HTP 友好形式（预生成索引张量 + Gather，避免跨步写）
3. 作为 MNN Hexagon 的 lowering bug 记录（跨步 slice/scatter）

## 产物
- artifacts/device-smoke/vd_p1/graphb_1l_diag_fp16.mnn
- artifacts/device-smoke/vd_p1/ref_*.raw (16 个 torch 参考)
- artifacts/device-smoke/vd_p1/dc/ (device CPU 16 输出) , dh/ (device HTP 16 输出)
- 探针源码 android/graphb_diag_probe.cpp
