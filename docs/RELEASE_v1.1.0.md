# v1.1.0 发布物清单（design + cosyvoice 合体）

## 这一版要放上去的东西

| # | 附件 | 字节 | 说明 |
|---|---:|---|---|
| 1 | `VoiceDesign-MNN-v1.1.0-arm64.apk` | 12,378,500 | 合体 App（VoiceDesign + CosyVoice3 朗读）。`versionCode 3 / versionName 1.2.0`，
|   |  |  | `applicationId com.cosyvoice.app`；正式签名（`legado-signing/vicentrent-release.jks`） |
|   |  |  | sha256 `0151381BD96A3C0142242ED0E423CA1EADC3198AA9E7E8CE5B83E0EBE8C36CEB` |
| 2 | `vd-weights-1-graphb-mnn.zip` | 503,734 | 设计权重 · GraphB 图 |
| 3 | `vd-weights-2-graphb-w-part0.zip` | 1,500,000,312 | 设计权重 · GraphB 常量池 part0 |
| 4 | `vd-weights-3-graphb-w-part1.zip` | 1,321,857,592 | 设计权重 · GraphB 常量池 part1 |
| 5 | `vd-weights-4-decoder.zip` | 944,514,162 | 设计权重 · codec decoder |
| 6 | `vd-weights-5-hosttables.zip` | 1,045,679,777 | 设计权重 · host tables |
| 7 | `vd-integrated-delta.zip` | ≈1.67 GB（待打包） | **本版新增**：把设计权重从"独立样例那一套"对齐到"合体 App 那一套"的增量（19 个同名文件 + 6 个差异文件 + 设备上的 `tokenizer_decoder_static_t96.mnn`） |
| 8 | `cosyvoice3-mnn-mobile-fp16-complete.zip` | 1,399,083,563 | 朗读（CosyVoice3）主模型包 —— 与 `CosyVoice3-MNN` Release v1.0.0 同一个文件 |
| 9 | `cosyvoice3-mnn-enrollment-extension.zip` | 997,807,778 | 朗读（CosyVoice3）音色注册扩展包 —— 同上 |

所有附件均 < 2 GB，符合 GitHub Release 单附件上限。

## 2 / 3 / 4 / 5 / 6 号不用重打

`Nian27/VoiceDesign-MNN` 现有 Release **v1.0.0** 里的这 5 个分片就是本机
`E:/AndroidStudioProjects/vd-weights-out/` 里那 5 个 zip，sha256 与仓库清单逐条一致（已核对 7 个关键文件）。
可以直接复用，或在 v1.1.0 里重新挂一遍。

**其中 GraphB 常量池（2.82 GB）合体 App 也用同一个文件**：本机 `vd-weights-staging/graphb28_v6_fp16.mnn.weight`
的 sha256 = `f23dbc06646349f3fd90aece8aaf78cdb8f73693d069a6b8d4f0da1031750528`，
与 v1.0.0 清单声明的完全一致 → **这 2.82 GB 不重复上传**。

## 7 号（增量包）需要现打

```text
内容 = 19 个同名文件（vd-weights-staging/ 里除 graphb28_v6_fp16.mnn.weight 外的全部）
     + 6 个差异文件（vd-weights-ext/）
     + 设备上的 tokenizer_decoder_static_t96.mnn（456,941,592 B，本机没有，需从设备拉）
已知 sha256（vd-weights-ext/，本机实测）：
  codec_head_fp16.mnn            64b0e3b1…24c1   2,360
  codec_head_fp16.mnn.weight     725fbd3a…5a07   12,595,200
  codepred_prefill_fp16.mnn       4676b1a1…4804   123,768
  codepred_prefill_fp16.mnn.weight 64b78282…c6c1  161,830,912
  codepred_step_fp16.mnn          3cfaccaf…82c1   94,648
  codepred_step_fp16.mnn.weight   64b78282…c6c1   161,830,912
  frame_emb_fp16.mnn              3033cc55…9e88   138,430,844
```

差异的来龙去脉见 `docs/VOICEDESIGN_WEIGHTS.md`。

## 部署顺序（用户侧）

```bash
# 1. 装 APK
adb install -r VoiceDesign-MNN-v1.1.0-arm64.apk
# 2. 朗读模型（2 个 zip）按 App 内"导入模型包"走
# 3. 设计权重：解 vd-weights-1..5 + vd-integrated-delta 到
#    <app私有目录>/files/cosyvoice3-mnn/voicedesign/
```
