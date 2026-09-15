# codepred 重构 CP-R1 / CP-R2 报告（2026-08-26）

## 结论：CP-R1 PASS + CP-R2 PASS

CP-R2 实测（同一 RNG，逐 stage）：

    torch codes = [496 1114 1363 157 881 1579 145 1253 455 288 586 914 779 1219 901]
    mnn   codes = [496 1114 1363 157 881 1579 145 1253 455 288 586 914 779 1219 901]
    15-code exact = 15/15
    worst logits cos = 1.0000000
    top50 overlap = 50/50 (全部 15 stage)

## 本轮挖出的两个真 bug（都在旧 codepred_ar 图里）

### Bug D：AR 前缀用了贪心，不是采样值
旧导出脚本内：
    seq = torch.cat([seq, self.emb[g](torch.argmax(lg, dim=-1))], dim=1)
ONNX 里 ArgMax 14 个节点被烧死。后果：
- 与官方采样语义不一致（官方 subtalker 是 do_sample=True + top_k/top_p/temperature）
- stage 0 的微小差异会翻转 argmax -> 后续每级沿不同贪心路径 -> 差异被逐级放大
- 这解释了此前 stage1..4 的 top50 overlap 崩到 29/5/0/4

### Bug E：attention 输出缺少 transpose（本轮新发现）
HF 的实现：
    attn_output = torch.matmul(attn_weights, value_states)   # (B,H,S,D)
    attn_output = attn_output.transpose(1, 2).contiguous()   # (B,S,H,D)
    attn_output = attn_output.reshape(*input_shape, -1)
我旧代码漏了 transpose，直接 reshape。
- decode 时 S=1：(1,16,1,128).reshape(1,1,2048) 与先 transpose 等价 -> GraphB 未受影响（继续冻结）
- prefill S=2：布局错误 -> codepred prefill 全错（cos 0.09）
该 bug 只在 S>=2 时暴露，因此长期潜伏。

## 重要更正：此前对 codepred precision 的结论作废

G2-B 曾得出 "codepred stage0 logits cos 0.9956 / top1 不同 / top50 42/50 -> 材料级扰动"。
但那是**在带 Bug D 的旧图上**测的，且 stage>=1 被贪心分裂污染。
=> 该结论作废。codepred 的 HTP precision 必须在**新图**上重测（CP-R3）。

## CP-R1 产物（新架构）

    codepred_prefill_fp16.mnn   162 MB   输入 past_hidden/code0_emb/rope_cos/rope_sin/slot_mask0/slot_mask1/attn_mask
                                        输出 hidden_last(1,1,1024) + kv_k/kv_v(5,1,8,16,128)
    codepred_step_fp16.mnn      162 MB   输入 token_emb/kv_k/kv_v/rope_cos/rope_sin/slot_mask/attn_mask
                                        输出 hidden_last + kv_k_out/kv_v_out
    lm_head_weight.f32          126 MB   (15,2048,1024)  host 侧应用
    codec_emb_weight.f32        252 MB   (15,2048,2048)  host 侧查表
    talker_codec_emb_weight.f32  25 MB   (3072,2048)     host 侧 code0 查表

设计要点：
- KV capacity = 16（prefill 2 + 14 steps），**不是 768**
- ArgMax = 0，ScatterND = 0（KV 写入用 one-hot mask 广播加，沿用 GraphB v6 已验证模式）
- RoPE / mask 均由 host 提供（沿用 GraphB 的经验）
- lm_head 与 embedding 放在 host（CPU）执行：15 次 1024x2048 matmul 约 31.5M MAC，代价可忽略，
  换来避免 30 个微小图的每次调用开销

## 未完成

    CP-R3  HTP forced-prefix 逐 stage precision   （未开始）
    CP-R4  HTP shared-RNG self-driven             （未开始）
    G3     文字 -> VoiceDesign -> EOS -> decoder -> reference.wav（未开始）

## 产物路径
- scripts/export/vd_cpr1_export2.py           (CP-R1 导出)
- scripts/export/vd_cpr2_clean.py             (CP-R2 验证)
- artifacts/device-smoke/vd_cpr/             (两张 mnn + 三个 f32 权重表)
