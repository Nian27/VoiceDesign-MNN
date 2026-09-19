# 复现指南（download → 跑起来）

> 目标：**从零下载，不读源码也能装起来并跑通**。
> 下面每一步都给了可直接点/可复制的链接与校验值。

## 0. 两个仓库各自是什么

| 仓库 | 是什么 | 要下什么 |
|---|---|---|
| [`Nian27/CosyVoice3-MNN`](https://github.com/Nian27/CosyVoice3-MNN) | **合体 App**：朗读（CosyVoice3 0.5B）+ 文字设计音色（VoiceDesign / Qwen3-TTS 1.7B） | APK + 朗读模型 2 个包 + 设计权重 |
| [`Nian27/VoiceDesign-MNN`](https://github.com/Nian27/VoiceDesign-MNN) | **VoiceDesign 独立样例 + 本仓库文档/脚本** | 设计权重 + APK |

## 1. 装 APK

```text
# 合体版（推荐）
https://github.com/Nian27/CosyVoice3-MNN/releases/download/v1.2.0/CosyVoice3-MNN-v1.2.0-arm64.apk
  12,378,500 B   sha256 0151381BD96A3C0142242ED0E423CA1EADC3198AA9E7E8CE5B83E0EBE8C36CEB

# 独立示例版
https://github.com/Nian27/VoiceDesign-MNN/releases/download/v1.1.0/VoiceDesign-MNN-v1.1.0-arm64.apk
  12,378,500 B（同一个 APK 文件）
```

```bash
adb install -r CosyVoice3-MNN-v1.2.0-arm64.apk
```

> 自建签名版需要 `signing.properties` + keystore（**不在仓库里**，属私有凭据）；没有它只能出 debug 包，
> 见 `README.md` 的「从源码构建」章节。

## 2. 朗读（CosyVoice3）模型：2 个包，在 Release v1.0.0

```text
cosyvoice3-mnn-mobile-fp16-complete.zip    1,399,083,563 B
  https://github.com/Nian27/CosyVoice3-MNN/releases/download/v1.0.0/cosyvoice3-mnn-mobile-fp16-complete.zip
  sha256 B1C74DFC90972D82D8166813620A882FE37A0DC02964E19C4F33DAAFEFEB1C84

cosyvoice3-mnn-enrollment-extension.zip      997,807,778 B
  https://github.com/Nian27/CosyVoice3-MNN/releases/download/v1.0.0/cosyvoice3-mnn-enrollment-extension.zip
  sha256 59EF5C8810D3CEAD01FF21A64379CF0A819F0A72651A6BA2B2343E9DA5A72231
```

在 App 内用「导入模型包」安装；或按 `research/mnn-cosyvoice3/deploy-cosy-models.sh` / `deploy-ext-weights.sh` 手动铺到 App 私有目录。

## 3. 设计（VoiceDesign）权重

### 3.1 已在 v1.0.0 公开的 5 个分片（独立样例那一套，21 个文件 / 4,812,549,603 B）

```text
vd-weights-1-graphb-mnn.zip            503,734 B
vd-weights-2-graphb-w-part0.zip    1,500,000,312 B
vd-weights-3-graphb-w-part1.zip    1,321,857,592 B
vd-weights-4-decoder.zip             944,514,162 B
vd-weights-5-hosttables.zip        1,045,679,777 B
https://github.com/Nian27/VoiceDesign-MNN/releases/tag/v1.0.0
```

GraphB 常量池 2.82 GB 因超 GitHub 单附件 2 GB 上限被拆成 part0 / part1，拼回来：

```bash
cat graphb28_v6_fp16.mnn.weight.part0 graphb28_v6_fp16.mnn.weight.part1 > graphb28_v6_fp16.mnn.weight
sha256sum graphb28_v6_fp16.mnn.weight   # 应为 f23dbc06646349f3fd90aece8aaf78cdb8f73693d069a6b8d4f0da1031750528
```

### 3.2 合体版 App 需要的那一套 —— **目前还没上传**（见第 5 节缺口）

合体版与独立样例的权重差 6 个文件（`tokenizer_decoder_static_t96.mnn`、`codepred_prefill/step` 拆开的 `.mnn` + `.mnn.weight`、
`codec_head_fp16.mnn.weight`）。差异清单与 sha256 见 [`VOICEDESIGN_WEIGHTS.md`](VOICEDESIGN_WEIGHTS.md)。

## 4. 从源码构建

| 组件 | 文档 |
|---|---|
| 合体 App（Kotlin + JNI） | `CosyVoice3-MNN/README.md` →「从源码构建」/「编译原生库」 |
| MNN fork（`libllm.so` / Hexagon / QNN） | `CosyVoice3-MNN/mnn-patches/mnn-cosyvoice-npu-fork/README.md` + `llm-engine.patch` + `backend-hexagon-qnn.patch` |
| VoiceDesign 独立样例 | `docs/BUILD_AND_DEPLOY.md` + `patches/README.md` |
| 部署脚本 | `CosyVoice3-MNN/research/mnn-cosyvoice3/deploy-*.sh`、`VoiceDesign-MNN/scripts/` |

## 5. ⚠️ 当前缺口（诚实清单）

| # | 缺口 | 影响 | 状态 |
|---|---|---|---|
| 1 | **合体版设计权重增量包 `vd-integrated-delta.zip`（≈1.67 GB）未上传** | 下 v1.2.0 / v1.1.0 的 APK 后，**文字设计音色用不了**（v1.0.0 那 5 个分片是独立样例那一套，与合体版差 6 个文件） | 待打包上传（需从设备拉 457 MB 的 `tokenizer_decoder_static_t96.mnn`） |
| 2 | `VoiceDesign-MNN` 本地提交 `7a57e45`（发布说明的 7 分钟基线更正）**未推送** | 线上 v1.1.0 说明缺"7 分钟基线"那几行 | 待推送（需凭据） |
| 3 | `VoiceDesign-MNN` v1.1.0 的发布说明正文**未 PATCH 成最新版** | 同上 | 待 PATCH（需凭据） |

**结论：朗读链路现在下载即可复现；设计链路的权重还差一个增量包，补上才算"下载即可复现"。**
