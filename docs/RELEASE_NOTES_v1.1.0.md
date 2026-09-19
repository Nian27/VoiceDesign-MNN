# VoiceDesign-MNN v1.1.0 版本发布说明

**发布日期**: 2026年9月18日

📦 **产物**: `VoiceDesign-MNN-v1.1.0-arm64.apk`（12,378,500 B，正式签名）｜ [完整清单](https://github.com/Nian27/VoiceDesign-MNN/blob/main/docs/RELEASE_v1.1.0.md)

## 📌 版本概述

v1.1.0 把两条线合成**一个可运行的例子**，并修掉朗读链上「长文本只读开头一句」的问题：

*   **合体**：VoiceDesign（Qwen3-TTS 1.7B，纯文字造音色）+ CosyVoice3（0.5B，朗读合成）跑在同一个 App 里，共用同一套 enrollment 与音色档案。
*   **长文本修复**：真正的病根是**采样器从来没有重复惩罚** —— `cosyvoice_ras` 的管线写死为 `range → topK → topP → select`，不含 `penalty` 步，而 `repetition_penalty` 只在 `penalty` 步生效。同一句 92 字（`inputTokens=231`）此前 5 次 NPU + 2 次 CPU 尝试**全部被质量门拒绝**、耗时约 187～275 秒，最后交付的还是被截断的音频；修复后**首次尝试 11.94 秒即通过**。
*   **顺带推翻两个旧结论**：「长句不能用 NPU」（判据字段 `continuousHexagon` 经查只是配置回显）与「CPU 退化率最高」（那次 266 连击来自不带守卫的 `mixed` 采样器）。

---

## 🚀 性能提升（全部为 SM8850 真机实测）

| 项目 | 修复前 | 修复后 | 提升 |
|---|---|---|---|
| 长句 NPU **prefill**（63 字 / `inputTokens=196`） | 走 OpenCL：10.8 – 22.9 s | **0.62 – 0.78 s** | **14× – 35×** |
| 同一句 92 字**整条朗读链**（`inputTokens=231`） | 5 次 NPU（合计 100.4 s）+ CPU 兜底 87.1 s 起，**全部被拒** | **首次尝试 11.94 s 通过** | **≥15.7×**（保守口径：187.5 s ÷ 11.94 s；把第二次 CPU 兜底算进去约 **23×**） |
| 单次可生成语音时长 | `maxTokens=500` ≈ **20 秒** | `maxTokens=952` ≈ **38 秒** | **1.9×** |
| 长句首次通过率（63 字，NPU） | 0 / 7 次尝试通过 | **4 / 4 首次即通过** | — |
| 音色档案**有效保留率**（**正确性**，不是速度） | 设计 7.28 s，被 `min(tokens, frames/2, 125)` 截断到 **125 token / 250 帧**（保留 **69%**），而 `promptPrefix` 里仍是完整 26 字 → **文本提示与音频提示不对齐** | 设计 4.08 s，**102 token / 204 帧，100% 保留**，文本与音频对齐 | 少丢 2.28 s，并消除一处输入不一致 |
| 音色创建**端到端**（设计 + 注册） | **~7 分钟 / 条**（README 实测基线：prefill 71 s + decode 273 s + decoder 57 s + load/tokenize ≈ 401 s，RTF ~47） | **97.3 s ~ 153.5 s** | **2.6× ~ 4.1×** |
| ↳ `prefill` | 71 s / 60–62 token | 19.4 ~ 32.1 s | **2.2× ~ 3.7×** |
| ↳ `decode` | 273 s / 109 帧（~2.5 s/帧） | 61.5 ~ 106.2 s / 51 帧（~1.2 s/帧） | **2.6× ~ 4.4×**（含帧数变少） |
| ↳ `decoder` | **57 s**（静态 300 帧） | **6.0 ~ 6.6 s**（静态 96 帧） | **8.6× ~ 9.5×** |
| 注册参考音频设计耗时 | 26 字 → 设计 7.68 s，其中只留 5.0 s | 15 字 → 设计 **4.08 s**，**全部保留** | 设计 **1.88×**，丢弃率 35% → **0%** |
| 质量门误杀（63 字那批 7 次尝试） | 拒 7 / 7，其中 4 次是正常输出（误杀率 **57%**） | 0 次误杀 | — |

朗读 LLM 段分项实测（全部首次尝试即接受）：

| 用例 | prompt → 输出 token | 吞吐 | LLM wall |
|---|---:|---:|---:|
| 短文本 | 150 → 155 | **67.2 tok/s** | **3.05 s** |
| 短文本 | 150 → 151 | **58.7 tok/s** | **3.32 s** |
| 长文本（92 字） | 231 → 452 | **47.6 tok/s** | **11.94 s** |

> **口径声明（2026-09-18 更正两处）**
>
> **一、"7 分钟"确有其事，我先前说"设计环节没被优化"是错的。** 基线出自 [VoiceDesign-MNN README §7 实测性能](https://github.com/Nian27/VoiceDesign-MNN#7-实测性能honor-magic8-pro)（同机 Honor Magic8 Pro、同后端 OpenCL）：
> `load 1.6–6.1 s + tokenize ~10 s + prefill 71 s + decode 273 s + decoder 57 s ≈ 401 s ≈ 7 分钟 / 条`。
> 现在同机同后端实测 **97.3 ~ 153.5 s**，即 **2.6× ~ 4.1×**。其中 `decoder` 从 57 s 掉到 6 s（**8.6× ~ 9.5×**）是最大的一块 ——
> 对应解码图从 `tokenizer_decoder_static_t300.mnn` 换成 **`_t96.mnn`**（静态帧 300 → 96）并把权重拆成 `.mnn` + `.mnn.weight`。
> 注意这几个分项是在**不同输出长度**下测的（旧 109 帧 / 新 51 帧），所以 `decode` 那一行的倍数里混了「帧数变少」，不能全算成优化。
>
> **二、另外更正一条旧结论**：`CosyVoiceEnrollmentCore.cpp` 的注释曾记录「prompt 125 token → 连续 Hexagon 解码不启用、4.09 tok/s」，
> 因此一度把 125 当成坏点。**本次实测（同一句 10 字、只换音色）三个档案都正常**：
>
> | 音色 | prompt token | `inputTokens` | tok/s | decodeMs |
> |---|---:|---:|---:|---:|
> | 内置参考 | 87 | 121 | **46.8** | 1.26 s |
> | 新默认 15 字 | 102 | 136 | **44.9** | 1.74 s |
> | 旧默认 26 字 | 125 | 169 | **44.7** | 2.80 s |
>
> 真正的变量是**总 `inputTokens`**（旧坏例 183 → 4.09 tok/s，本次 121~169 全部正常）。**边界在 169 与 183 之间，尚未定位。**
>
> **三、26 → 15 字这一改本身的收益是正确性**（旧默认设计 7.28 s 却被截断到 125 token、且文本与音频提示不对齐），
> 旧默认的 26 字会设计出 7.28 s 音频，却被注册环节按 `min(tokens, frames/2, 125)` 截断到 125 token，
> 而 `promptPrefix` 仍然保存完整 26 字 —— 文本提示与音频提示不对齐，这是实打实的输入缺陷。
> 上面那行 1.60× 只是「参考文本变短、自回归步数 91 → 51」带来的副作用，**不要当成「修复提速 1.6 倍」引用**。
>
> 另外更正一条旧结论：`CosyVoiceEnrollmentCore.cpp` 的注释曾记录「prompt 125 token → 连续 Hexagon 解码不启用、
> 4.09 tok/s」，因此一度把 125 当成坏点。**本次实测（同一句 10 字、只换音色）三个档案都正常**：
>
> | 音色 | prompt token | `inputTokens` | tok/s | decodeMs |
> |---|---:|---:|---:|---:|
> | 内置参考 | 87 | 121 | **46.8** | 1.26 s |
> | 新默认 15 字 | 102 | 136 | **44.9** | 1.74 s |
> | 旧默认 26 字 | 125 | 169 | **44.7** | 2.80 s |
>
> 也就是说「125 token 就慢」在当前代码下不成立；真正的变量是**总 `inputTokens`**（旧坏例是 183 → 4.09 tok/s，
> 本次 121~169 全部正常）。**这个边界在 169 与 183 之间，尚未定位**，已列入已知问题。

---

## 🔧 修了什么，怎么修的

### 1. 采样器没有重复惩罚（最关键）

*   **现象**：92 字文本生成固定停在 500 token、没有 EOS，只读出开头一句。
*   **定位**：读构建机上的 MNN fork 源码 `transformers/llm/engine/src/sampler.cpp` —— `cosyvoice_ras` 的管线写死为 `stepCosyVoiceRange → stepTopK → stepTopP → stepSelect`，**没有 `penalty` 步**；MNN 默认的 `mixed_samplers` 也不含它。**这套 LLM 从来没有过重复惩罚。**
*   **修复**：runtime config 默认改为 `sampler_type=mixed` + `mixed_samplers=["penalty","topK","topP","temperature"]` + `repetition_penalty=1.15`。
*   **验证**（同一句 92 字 / hexagon）：

| 采样器 | 尝试 | 生成 token | EOS | unique | 最长连续重复 | 结果 |
|---|---:|---:|---|---:|---|---|
| `cosyvoice_ras`（旧） | 6 | 500 / 952 | ✗ 全部 | 57 – 219 | 33 / 314 / 345 / 610 / 858 | 全部被拒 |
| `mixed` + `penalty` 1.15 | **1** | **590** | **✓** | — | — | **接受，成品 23.56 s** |

### 2. `maxTokens` 写死 500（≈ 只够 20 秒语音）

*   **定位**：这条链的 LM 生成的是「音色前缀文本 + 正文」**整段**，音色档案里那 25 个字的参考文本同样占 token。实测约 **6 token/字**：39 字 → 184 token；63 字 → 353 – 443；92 字 → 必然撞满 500。
*   **修复**：`maxTokens = (前缀字数 + 正文字数) × 8 + 128`，钳在 `[500, 1024]`（92 字实测取 952）。

### 3. 质量门判据误杀正常输出

*   **定位**：旧判据 `longestRun >= 8 即拒` 在长输出上误杀 —— 63 字那批 7 次尝试里有 4 次是**正常终止**（370 – 387 token、有 EOS、unique 230 – 246），最长连续段只有 8 – 20 个 token（约 0.3 – 0.8 秒），在语音 token 里对应长元音 / 静音 / 拖音；真正跑飞的几次**都没有 EOS**，最长段 105 – 858。
*   **修复**：新判据只保留三条能证明退化的条件 ——

```text
1. 没有 EOS                 生成只可能在 EOS 或 maxTokens 处停下，没 EOS 就是撞满上限
2. unique <= 4 且长度 >= 20   灾难性坍缩
3. 单段连续重复 >= 64         "正常"最大 20、"跑飞"最小 105，64 落在两者之间
```

*   并新增**尾部空转裁剪**：没有 EOS、重复段延伸到序列末尾、重复段 ≥ 32 且保留段 ≥ 32 时切掉重复段（写 `llm-tail-repair.txt` 留证，不静默处理）。

### 4. 「长句不能用 NPU」这个结论是错的

*   **定位**：报告里的 `continuousHexagon` 算的是 `(配置名含 hexagon) && (hexagon-stage-layers.txt 非空)`，`npuFirstTokenValid` 直接等于 `hybridNpuPrefill`（= 配置含 hexagon 且 stage 文件为空）—— **两者都是配置回显，不是运行期测量**；而 stage 文件恰好由 App 自己在调用前写入，所以它必然为真。加上判据本身就是第 3 条那个误杀阈值。
*   **修复**：取消「≤ 24 字走 NPU、否则走 OpenCL」的分流，一律用硬件策略给出的后端。实测同一句 63 字 hexagon **4/4 首次即通过**。

### 5. 最后一次尝试不过质量门，退化音频被悄悄交付

*   **定位**：原来只有第一次尝试过门，最后那次 CPU 兜底跑完直接交付 —— 而 CPU 恰是退化率最高的那条路。
*   **修复**：尝试计划改为「路由后端 5 次 → CPU 兜底 2 次」，**每一次都过同一道门**，全不过就报错。

### 6. 注册参考文本 26 字 → 15 字

*   **定位**：注册时音频会被 `MAX_REALTIME_PROMPT_TOKENS = 125`（= 5.0 秒 / 250 帧）截断，而 `promptPrefix` 里存的仍是**完整文本** —— 文本提示与音频提示不对齐（26 字设计出 7.68 s，截到 5.0 s，前 5 秒只覆盖前约 17 个字）。
*   **修复**：默认参考文本改为 15 字（对齐内置基准音色 `builtin-mnn-reference-v1` 的 15 字 / 87 token / 3.48 秒）。实测新默认设计出 **4.08 秒 / 102 token，不再被截断**。`design.json` 现在记录 `promptTokenCount` / `promptFrameCount` / `promptTruncatedByRealtimeLimit`。

---

## 📦 产物与权重

本 Release 提供合体 APK。权重分两部分（全部为 GitHub 附件 / 已有 Release 复用，单个附件均 < 2 GB）：

*   **设计权重 · 合体版增量包（本版新增，已上传）**：`vd-integrated-delta.zip` 把权重对齐到**合体版 App** 那一套（24 个文件 + `SHA256SUMS.txt`）。
    因为单次 2 GB 上传在这个网络下会中途断，已按 400 MB 切成 5 片（这是**字节切片**，不是 5 个独立 zip）：

    ```bash
    # 下载 part0..part4 后拼回来
    cat vd-integrated-delta.zip.part0 vd-integrated-delta.zip.part1 vd-integrated-delta.zip.part2 \
        vd-integrated-delta.zip.part3 vd-integrated-delta.zip.part4 > vd-integrated-delta.zip
    # 期望：1,991,126,872 字节，sha256 1b7f2053ecad63f1168ee053050f9b69f35784a608e556a4e07a9c1e124c2af3
    sha256sum vd-integrated-delta.zip
    unzip vd-integrated-delta.zip -d voicedesign/
    ```

    分片校验值见附件 `vd-integrated-delta-PARTS-SHA256SUMS.txt`：

    ```text
    316b75846d8bea7fb283535f0ac182a17a0ee05d50402869862ed0aa78bd0036 *vd-integrated-delta.zip.part0
    2dd22ee0e3b2885c2db9616074e0938fda7abc8193482a2b295f5f6789c2005a *vd-integrated-delta.zip.part1
    699f1e5c23dd0ebef37962d87f5d2907d28ffa24f085f84d0e3150c46c7c00c0 *vd-integrated-delta.zip.part2
    4a1cae13736858f6c3ed56bccab58b51af13673898196bec02ab6ddab9dd2b18 *vd-integrated-delta.zip.part3
    1d8beaaca7f857949c11bc7a0d8f0ace8aa97a03073cb63c3a3d9519596baf3e *vd-integrated-delta.zip.part4
    ```

*   **设计权重 · 独立样例那一套**：`vd-weights-1..5` 见 [v1.0.0](https://github.com/Nian27/VoiceDesign-MNN/releases/tag/v1.0.0)（GraphB 常量池 2.82 GB 因超 2 GB 上限拆成 part0 / part1）。合体版只需要其中的 `graphb28_v6_fp16.mnn.weight`（sha256 与增量包外的那一份一致）。差异见 [docs/VOICEDESIGN_WEIGHTS.md](https://github.com/Nian27/VoiceDesign-MNN/blob/main/docs/VOICEDESIGN_WEIGHTS.md)。
*   **朗读权重（CosyVoice3 0.5B）**：`cosyvoice3-mnn-mobile-fp16-complete.zip`（1,399,083,563 B）与 `cosyvoice3-mnn-enrollment-extension.zip`（997,807,778 B），见 [CosyVoice3-MNN v1.0.0](https://github.com/Nian27/CosyVoice3-MNN/releases/tag/v1.0.0)。

字体层级、部署顺序与逐个文件的 sha256 见 [docs/RELEASE_v1.1.0.md](https://github.com/Nian27/VoiceDesign-MNN/blob/main/docs/RELEASE_v1.1.0.md)。

---

## ⚠️ 已知问题

*   **参考文本会被一并念出**：LM 生成的是「参考文本 + 正文」整段，App 把整段都当目标 token 交给 flow。要只读出正文，需要先确认 flow / conditioner 对「token 序列是否含提示区」的约定，不能凭猜改。
*   **`decodeMs` 与 `wallMs` 出现过互相矛盾的读数**（809 s vs 639 s），该字段的差分在某些路径上不可靠，诊断时请用 `wallMs` 或产物文件时间。
*   **`len266@[234-499]` 的重复起点**只有单次观测，不足以支撑结论。

---

## 📚 文档

*   [CHANGELOG.md](https://github.com/Nian27/VoiceDesign-MNN/blob/main/CHANGELOG.md) —— 问题 / 定位 / 解法 / 实测数字
*   [docs/LONG_TEXT_FIX_2026-09-18.md](https://github.com/Nian27/VoiceDesign-MNN/blob/main/docs/LONG_TEXT_FIX_2026-09-18.md) —— 长文本修复详解
*   [docs/VOICEDESIGN_WEIGHTS.md](https://github.com/Nian27/VoiceDesign-MNN/blob/main/docs/VOICEDESIGN_WEIGHTS.md) —— 权重现状与缺口
*   [docs/RELEASE_v1.1.0.md](https://github.com/Nian27/VoiceDesign-MNN/blob/main/docs/RELEASE_v1.1.0.md) —— 发布物清单
