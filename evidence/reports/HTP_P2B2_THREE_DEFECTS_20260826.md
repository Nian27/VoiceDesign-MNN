# HTP P2-B2 报告：三个独立 lowering 缺陷（2026-08-26）

## 契约演进（deployment contract）

    v1: inputs_embeds, position_ids, cache_position, kv_k, kv_v
    v2: inputs_embeds, cache_position, kv_k, kv_v, rope_cos, rope_sin          (host interleaved mRoPE)
    v3: inputs_embeds, cache_position, kv_k, kv_v, rope_cos, rope_sin, attn_mask  (+ host mask)
    每步 host 侧新增数据量: rope 2x128 fp32 = 1KB, mask 768 fp32 = 3KB  (可忽略)

## P2-A CPU Gate = PASS (bit-exact)
    q_rope  host vs official  cos=1.00000000  maxdiff=0
    k_rope  host vs official  cos=1.00000000  maxdiff=0
    layer0_out(host rope) vs official ref  cos=1.00000000  maxdiff=0

## P2-B2-A CPU Gate = PASS
    手工 attention + host mask  vs  官方 SDPA:  cos=1.00000000  maxdiff=5.96e-08

## P2-B2 结果（device CPU vs HTP，同一张 v3 图）

| boundary | CPU_cos | HTP_cos | CPU_maxabs | HTP_maxabs | 判定 |
|---|---|---|---|---|---|
| norm_out | 1.0000000 | 1.0000000 | 0.7240 | 0.7241 | OK |
| q_raw / k_raw / v_raw | 1.0000000 | 1.0000000 | 2.6682/2.2433/0.5850 | 2.6680/2.2441/0.5850 | OK |
| q_norm / k_norm | 1.0000000 | 1.0000000 | 21.3237/254.8007 | 21.3281/254.7500 | OK |
| q_rope | 1.0000000 | 0.9999999 | 16.1512 | 16.1562 | OK (缺陷1已修) |
| k_rope | 1.0000000 | 1.0000000 | 254.8006 | 254.7500 | OK (缺陷1已修) |
| **qk_logits** | 1.0000000 | **0.1567780** | 315.2457 | 315.2500 | **缺陷2** |
| **qk_masked** | 1.0000000 | **0.1567780** | 315.2457 | **65504.0000** | 缺陷2 + mask 钳位 |
| attn_probs | 1.0000000 | 0.7246542 | 1.0000 | 1.0000 | 缺陷2 下游 |
| attn_out | 1.0000000 | 0.7778584 | 0.5850 | 0.5850 | 缺陷2 下游 |
| o_proj_out | 1.0000000 | 0.8916572 | 1.7166 | 1.3223 | 下游 |
| res1/post_norm/mlp_out/layer0_out | ~1.0 | 0.90/0.73/0.51/0.70 | | | 下游 |
| **kv_k_out** | 1.0000000 | **0.0715174** | 258.4134 | 254.7500 | **缺陷3** |
| **kv_v_out** | 1.0000000 | **0.0010162** | 138.8099 | 0.5850 | **缺陷3** |

## 缺陷 #2：attention 内部 qk_logits 数值错位

    qk_logits maxabs: CPU 315.2457 / HTP 315.2500   (几乎相同)
    qk_logits cos   : 0.1567780
幅值相同而 cos 极低 => 值被置换 / 布局维度错。
嫌疑算子：torch.matmul(q_rope, k_all.transpose(-1,-2)) 中的 transpose + matmul lowering。
另：mask 的 -inf 在 HTP 上被钳为 fp16 极值 65504，neginf 计数归零（CPU 有 11280 个），
    即 mask 的 -inf 传播也有独立问题。
（注意：v2 用官方 SDPA 时首个分叉出现在 attn_out cos=0.496；v3 用手工 matmul 时上移到 qk_logits。
  两条路径都指向 attention 内部，但触发点不同 => 转置/matmul 是首要嫌疑。）

## 缺陷 #3：KV slot-write 不 preserve 目标 buffer（用户判据逐字命中）

    用户预判: "若 old slots = 0 / current slot 正确 / future slots = 0
               => HTP 的 slot-write / scatter update 没有 preserve destination buffer"
    实测:
      CPU kv_v_out: old-slot cos=1.000000  zero_frac=0.000 | cur maxabs=0.5850
      HTP kv_v_out: old-slot cos=0.000000  zero_frac=1.000 | cur maxabs=0.5850
即 old slots 全部被清零，只有当前写入槽有正确值。
对应图内操作: kvv_out[self.li,:,:,cache_position,:] = v[:,:,0,:].unsqueeze(2)
（clone + strided/scatter 写；ONNX 侧为 ScatterND 类）
=> 与 attention 数学完全独立的第三个 backend lowering 问题，且是全链阻塞项（KV 需回灌）。

## 结论：三个缺陷都是 lowering/layout 类，不是数学问题
    1. interleaved mRoPE stride-3 跨步读写      -> 已确诊，host interleave 修复有效
    2. attention 内部 transpose/matmul + mask(-inf) -> 已定位到算子层
    3. KV slot-write 不 preserve 目标 buffer     -> 已确诊（判据与预判一致）

## 下一步候选
- #2 细分：暴露 k_all 与 k_all.transpose(-1,-2) 两个张量作 output，
  判断是「transpose 本身错」还是「matmul 对转置输入错」
- #2b mask：把 mask 从 -inf 改为大负数（如 -1e4）避免 fp16 钳位，看是否恢复正常
- #3 workaround：KV 状态留 host，图只输出 current_k/current_v，host 更新固定 KV 再喂入
  （性能差但可先把 文字->音色 闭环打通；后续转向 MNN 原生 Attention/runtime-owned KV）

## 产物
- scripts/export/vd_p2a_host_rope.py, vd_p2b2_export.py
- android/graphb_diag_probe_v2.cpp, graphb_diag_probe_v3.cpp
- artifacts/device-smoke/vd_p2/graphb_1l_diag_v2_fp16.mnn / v3_fp16.mnn, merged_cos/sin.raw, attn_mask.raw
- artifacts/device-smoke/vd_p2/{c2,h2,c3,h3}/ (16/16/19/19 outputs)
