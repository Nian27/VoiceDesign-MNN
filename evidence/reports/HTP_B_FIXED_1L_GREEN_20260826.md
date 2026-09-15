# HTP P2 收尾：B 修复成功，1L 诊断图 HTP 全绿（2026-08-26）

## 结论：HTP correctness 在 1L 范围已达到 PASS

    B v5 HTP worst cos = 0.9998744   first_bad = None     (14/14 boundary)

| boundary | CPU_cos | HTP_cos | 说明 |
|---|---|---|---|
| norm_out | 1.0000000 | 1.0000000 | |
| q_rope | 1.0000000 | 0.9999999 | 缺陷#1 修复后 |
| k_rope | 1.0000000 | 1.0000000 | 缺陷#1 修复后 |
| k_all | 1.0000000 | 1.0000000 | **缺陷#3 修复后** |
| k_T | 1.0000000 | 1.0000000 | transpose 一直是对的 |
| qk_logits | 1.0000000 | 0.9999996 | **缺陷#2 = 幻影（#3 下游）** |
| qk_masked | 1.0000000 | 1.0000000 | mask -1e4 |
| attn_probs | 1.0000000 | 0.9998744 | |
| attn_out | 1.0000000 | 0.9999610 | |
| layer0_out | 1.0000000 | 0.9999698 | |
| kv_k_slot / kv_v_slot | 1.0000000 | 1.0000000 | 当前槽写入正确 |
| kv_k_old0 / kv_v_old0 | 1.0000000 | 1.0000000 | **old slots 完整保留** |

## 三项修复总账

| # | 缺陷 | 修复方式 | 验证 |
|---|---|---|---|
| 1 | interleaved mRoPE stride-3 跨步读写 | **host 侧做三路 interleave**，图内只留连续 RoPE；每步多传 rope_cos/sin 各 128 维 | q/k_rope cos 0.58/0.40 -> 1.0 |
| 2 | （幻影）attention transpose/matmul | **无需修复** —— 由 #3 传导；#3 修好后自动消失 | qk_logits 0.157 -> 0.9999996 |
| 2b | mask -inf 被 fp16 钳位到 65504 | **host 侧预生成 mask**，取值 -1e4 | qk_masked neginf 归零 -> cos 1.0 |
| 3 | KV slot-write 不 preserve destination buffer | **scatter -> 广播 multiply-add**（无 clone-write、无 ScatterND） | old-slot zero_frac 1.000 -> 0.000, cos 1.0 |

关键：v5 的 ONNX 中 ScatterND 计数 = 0。

## 部署契约（当前）

    inputs : inputs_embeds, cache_position, kv_k, kv_v, rope_cos, rope_sin, attn_mask, slot_mask
    outputs: hidden_states, current_k, current_v   (正式版；诊断版另暴露中间量)

    host 侧每步负责: mRoPE interleave、causal mask、KV slot 拼装
    HTP 内部只做: QKV proj / 连续 RoPE / matmul / 广播 elementwise / softmax / MLP

## 性能记录（重要，B 的代价）

    1L 图 CPU FORWARD_MS=254   HTP FORWARD_MS=5860
原因：广播 multiply-add 按 (28,1,8,768,128) 全尺寸物化 delta（88MB x2），
      在 1L 诊断图里等于「为写 1 个槽而搬运整块 KV」。
28L 正式版必须先优化，否则 28 倍放大不可接受。

优化方案（28L 版采用）：
  逐层只对本层切片做加法:  kvk_i = kv_k[i] + k_rope_i * slot_mask_b     # (1,8,768,128) = 786K elems
  输出只给当前槽:          cur_k/cur_v 各 (28,1,8,128) = 114KB
  host 侧把这 114KB 写回自己的持久 KV，再把整块 KV 作为下一步输入
  => 图内不再物化 28 层 delta，也不再输出 88MB 的 kv_k_out/kv_v_out

## 产物
- scripts/export/vd_a_export_v4.py (A: 4 boundary + -1e4 mask)
- scripts/export/vd_b_export_v5.py (B: 广播 multiply-add 修复)
- android/graphb_diag_probe_v4.cpp / v5.cpp
- artifacts/device-smoke/vd_p2/graphb_1l_v5_fp16.mnn (100MB), slot_mask.raw
- artifacts/device-smoke/vd_p2/{c5,h5}/ (14 outputs each)
- 诊断链: vd_p1 (16 boundary) -> vd_p2 v2/v3/v4/v5 逐轮收敛
