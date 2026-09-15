# HTP P2 报告：mRoPE 修复确认 + 第二处分叉（2026-08-26）

## P2-A CPU 数学等价 Gate = PASS（bit-exact）
host 侧做三路 mRoPE interleave（stride=3 重排），图上只留连续 RoPE：
    merged_cos[128], merged_sin[128]  = host_merge(rotary_cos3, rotary_sin3, mrope_section=[24,20,20])
    q_rope = q * merged_cos + rotate_half(q) * merged_sin
实测：
    q_rope  host vs official  cos=1.00000000  maxdiff=0.000e+00
    k_rope  host vs official  cos=1.00000000  maxdiff=0.000e+00
    layer0_out(host rope) vs official ref  cos=1.00000000  maxdiff=0.000e+00

## P2-B 1L DIAG v2（新契约，position_ids 移出模型）
    inputs : inputs_embeds, cache_position, kv_k, kv_v, rope_cos, rope_sin
    outputs: 同 v1 的 16 个 boundary
    merged_cos/sin 每步仅 512B

### CPU Gate PASS（v2 与 v1 CPU 值逐位一致）
    q_rope 16.151224 / k_rope 254.800568 / kv_k_out 258.413422

### HTP 结果：第一处修复生效，第二处浮现
| boundary | CPU_cos | HTP_cos | 备注 |
|---|---|---|---|
| norm_out | 1.0000000 | 1.0000000 | |
| q_raw / k_raw / v_raw | 1.0000000 | 1.0000000 | |
| q_norm / k_norm | 1.0000000 | 1.0000000 | |
| **q_rope** | 1.0000000 | **0.9999999** | 修复前 0.5777 -> 已恢复 |
| **k_rope** | 1.0000000 | **1.0000000** | 修复前 0.4049 -> 已恢复 |
| **attn_out** | 1.0000000 | **0.4961523** | 新 first divergence |
| o_proj_out | 0.9999999 | 0.6988830 | |
| res1 | 0.9999999 | 0.6962757 | |
| post_norm | 0.9999996 | 0.6741439 | |
| mlp_out | 0.9999993 | 0.6471184 | |
| layer0_out | 0.9999999 | 0.5538941 | |
| kv_k_out | 1.0000000 | 0.0715174 | KV 回灌隐患 |
| kv_v_out | 1.0000000 | 0.0010162 | KV 回灌隐患 |

## 结论 1：mRoPE stride-3 lowering 错误已被证实 + 有效 workaround
把 interleave 搬出图后 q_rope/k_rope 立刻恢复到 cos≈1.0。
=> 「Hexagon 对 interleaved mRoPE 的 stride-3 跨步读写 lowering 有误；host-preinterleaved RoPE 是有效 workaround」
这个 workaround 可以直接作为正式实现（decode 单步只需 host 生成 128 维 cos/sin，每步 512B）。

## 结论 2：第二处分叉 attn_out（幅值对、数值错位）
    attn_out maxabs: CPU 0.5850 / HTP 0.5850   (相同)
    attn_out cos   : 0.4962
幅值一致而 cos 低 => 典型 layout/permutation 类错误，位置在 attention 内部或其后 reshape。
嫌疑（按优先级）：
  1. attention mask 构造：positions <= cache_position 的广播比较 + where(0,-inf)（fp16 下 -inf 处理）
     —— 与 DSP profile 中异常的 BINARY_ELEMENTWISE 对应
  2. attention 输出 reshape/transpose (1,16,1,128)->(1,1,2048) 的 layout 处理
验证手段与 mRoPE 同构：把 mask 改为 host 侧预生成（1x1x1x768，仅依赖 cache_position）作为 graph 输入。

## 结论 3：KV 输出在 HTP 上丢失输入数据（全链阻塞项）
    kv_v_out maxabs: CPU 138.81 / HTP 0.585   (HTP 只剩写入的那一片)
    kv_v_out cos   : 0.0010
自驱循环依赖 KV 回灌，此项不修全链无法运行。需区分：
  (a) 真实数据丢失（clone/copy 在 HTP 上未生效）
  (b) module 输出读回路径问题
验证：比对 HTP 上 kv_v_out 与输入 kv_v 的逐元素差异（其余层是否原样保留）。

## 产物
- scripts/export/vd_p2a_host_rope.py, vd_p2ab.py (P2-A + P2-B 导出)
- android/graphb_diag_probe_v2.cpp
- artifacts/device-smoke/vd_p2/graphb_1l_diag_v2_fp16.mnn, merged_cos.raw, merged_sin.raw
- artifacts/device-smoke/vd_p2/{c2,h2}/ (16 outputs each)
- artifacts/device-smoke/vd_p1/ref_*.raw (torch 参考，v2 复用同一份)
