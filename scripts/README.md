# scripts/ 索引

本目录是**全部处理脚本**（276 个），按流水线阶段分组。命名前缀：
- `vd_` = VoiceDesign 主线
- `cp_` = code_predictor（codepred）相关
- 无前缀 = 早期阶段（图导出 / 数值对齐 / 诊断）

---

## 一、图导出（从 HF 权重到 MNN）

| 脚本 | 产物 |
|---|---|
| `export_graphb_decode_step.py` → `vd_b_export_v5.py` → `vd_b28_export_v6.py` | **graphb28_v6_fp16.mnn**（28L 单步，KV 768 slot）|
| `vd_cpr1_export2.py` | **codepred_prefill / codepred_step**（host 采样 + KV cache）|
| `export_decoder_static_t300.py` / `export_m2_tokenizer*.py` | **tokenizer_decoder_static_t300.mnn** |
| `vd_m5b_export_codecemb.py` | codec_emb_fp16.mnn |
| `vd_m5b_export_grapha_t62.py` | （早期 GraphA 方案，已被"用 GraphB 自身做 prefill"取代）|
| `vd_p1_export_perlayer.py` / `vd_p1a_export_1l_diag.py` | 逐层 / 单层诊断图（排查 lowering bug 用）|

## 二、host 表导出

| 脚本 | 产物 |
|---|---|
| `vd_text_prep.py` | **tokenizer.bin** + **text_embedding.fp16** + **text_proj_fc1/fc2.fp16** |
| `vd_codec_emb.py` | **talker_codec_emb.f32** |
| `vd_app_prep.py` | talker_rope_*.f32 / codepred_rope_*.f32 / prompt_emb.f32 |

## 三、验证（本项目的 Gate 都靠这些）

| 脚本 | 验证什么 | 结果 |
|---|---|---|
| `vd_tok_cases.py` + `vd_tok_cmp.py` | 设备分词 vs HuggingFace | **10/10 逐 id 相同** |
| `vd_recon_prompt.py` | PC 复刻 prompt vs 官方捕获 | **exact_equal = True** |
| `vd_tok_probe.py` | 探查 talker 结构（两张 embedding 表）| — |
| `vd_proj_test.py` / `vd_proj_test2.py` / `vd_proj_bias.py` | **发现 MLP bias 缺失**（真凶 1）| cos 0.83 → **1.0** |
| `vd_cpr2_clean.py` / `vd_cpr2_verify.py` | codepred PC MNN vs Torch | **15/15 codes exact** |
| `vd_cpr3_prep.py` / `vd_cpr3_final.py` / `vd_cpr3_logits.py` | codepred HTP prefill+14 级 | hidden ≥0.99992 |
| `vd_cpr4_cmp.py` / `vd_cpr4_margin.py` | codepred 自驱轨迹 + 分叉 margin | boundary flip |
| `vd_wav_check.py` | WAV 内容（时长/峰值/RMS/静音段）| — |
| `vd_m5b_inspect_wav.py` | WAV 头解析 | — |

## 四、诊断（踩坑时写的，保留作参考）

| 脚本 | 查什么 |
|---|---|
| `vd_cpr_dbg1..9` / `vd_cpr_dbgx.py` | codepred 逐级手动复算，定位 Bug D/E |
| `vd_cp_dbg1..3.py` | codepred 中间量 |
| `vd_cpr_hook*.py` | hook 官方 forward 抓真实输入 |
| `vd_cpr_attn*.py` | attention 输出形状/转置（**Bug E**）|
| `vd_g2b_*.py` | G2-B 首分歧与 margin |
| `vd_attn_iso.py` / `vd_attn_src.py` | attention 隔离测试 |
| `analyze_isnan*.py` | 图里 `IsNaN` 节点来源 |
| `vd_determ_probe.py` | 确定性检查（同输入两次跑是否逐位一致）|

## 五、部署与运行时辅助

| 脚本 | 作用 |
|---|---|
| `patch_vd_*.py` | 开发期对 JNI/Kotlin 打补丁的小工具（留档）|
| `vd_jni_text.py` | 把"设备端构造 prompt"接进 JNI 的那次改动 |
| `compare_*.py` / `check_*.py` | 各阶段的产物比对 |

---

## 没有收录的

- MNN 源码补丁：见 `../patches/README.md`（本项目**不需要**补丁即可在 OpenCL 后端跑通）
- 模型权重：见 `../weights/` 与 `../release-manifest.json`
- 原始 HF 权重：请自行从 Qwen 官方获取
