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

### 3.2 合体版 App 需要的那一套 —— **已上传**（v1.1.0，切成 5 片）

合体版与独立样例的权重差 6 个文件（`tokenizer_decoder_static_t96.mnn`、`codepred_prefill/step` 拆开的 `.mnn` + `.mnn.weight`、
`codec_head_fp16.mnn.weight`），另有 `prompt_emb.f32` 只在合体版里存在。差异清单见 [`VOICEDESIGN_WEIGHTS.md`](VOICEDESIGN_WEIGHTS.md)。

```bash
# 1) 下载 v1.1.0 的 5 个分片 + 校验值
BASE=https://github.com/Nian27/VoiceDesign-MNN/releases/download/v1.1.0
for i in 0 1 2 3 4; do curl -LO "$BASE/vd-integrated-delta.zip.part$i"; done
curl -LO "$BASE/vd-integrated-delta-PARTS-SHA256SUMS.txt"
sha256sum -c vd-integrated-delta-PARTS-SHA256SUMS.txt

# 2) 拼回一个 zip（29.4 MB 的切片，不是 5 个 zip）
cat vd-integrated-delta.zip.part0 vd-integrated-delta.zip.part1 vd-integrated-delta.zip.part2 \
    vd-integrated-delta.zip.part3 vd-integrated-delta.zip.part4 > vd-integrated-delta.zip
sha256sum vd-integrated-delta.zip   # 应为 1b7f2053ecad63f1168ee053050f9b69f35784a608e556a4e07a9c1e124c2af3
# 期望大小 1,991,126,872 字节

# 3) 解到 App 私有目录（含 24 个模型文件 + SHA256SUMS.txt）
unzip vd-integrated-delta.zip -d <app私有目录>/files/cosyvoice3-mnn/voicedesign/
```

分片 sha256：

```text
316b75846d8bea7fb283535f0ac182a17a0ee05d50402869862ed0aa78bd0036 *vd-integrated-delta.zip.part0
2dd22ee0e3b2885c2db9616074e0938fda7abc8193482a2b295f5f6789c2005a *vd-integrated-delta.zip.part1
699f1e5c23dd0ebef37962d87f5d2907d28ffa24f085f84d0e3150c46c7c00c0 *vd-integrated-delta.zip.part2
4a1cae13736858f6c3ed56bccab58b51af13673898196bec02ab6ddab9dd2b18 *vd-integrated-delta.zip.part3
1d8beaaca7f857949c11bc7a0d8f0ace8aa97a03073cb63c3a3d9519596baf3e *vd-integrated-delta.zip.part4
```

> 注：合体版仍需要 v1.0.0 里的 `graphb28_v6_fp16.mnn.weight`（2.82 GB，由 part0 + part1 拼回），
> 或直接用增量包里没有的那一份 —— 两者 sha256 相同：`f23dbc06646349f3fd90aece8aaf78cdb8f73693d069a6b8d4f0da1031750528`。

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
| 1 | ~~合体版设计权重增量包未上传~~ | — | ✅ **已上传**（v1.1.0 的 `vd-integrated-delta.zip.part0..4`，拼回后 1,991,126,872 B / sha256 `1b7f2053…c2af3`） |
| 2 | ~~发布说明的 7 分钟基线更正未推送~~ | — | ✅ 已推送（`2f84bb7`） |
| 3 | ~~发布说明正文未 PATCH~~ | — | ✅ 已 PATCH（HTTP 200） |

**结论：朗读链路与设计链路现在都是「下载即可复现」。**（设计链路的权重自 v1.1.0 起已随 Release 提供，切成 5 片。）
