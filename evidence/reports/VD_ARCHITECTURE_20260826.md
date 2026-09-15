# VoiceDesign 子系统架构定案（2026-08-26）

## 系统定位（非最终目标）

Qwen3-TTS VoiceDesign 在 ReaderVoice-Mobile 中是**子系统**：

```text
角色文字描述 → SM8850 VoiceDesign → canonical reference voice WAV
```

它不是小说播放链的一部分，而是**新角色 VoiceIdentity 的创建器**。

## 全系统链

```text
ReaderDirector → RoleProfile → 缺 VoiceIdentity？
  YES → VoiceDesign → reference.wav → CosyVoice enrollment（角色主路）
                                 → Audio8 enrollment（旁白 + fallback）
  NO  → 直接 Canonical RenderUnit → PCM
```

## 桥接契约（关键决策）

- **不传内部 embedding/latent**：Qwen3-TTS latent ≠ CosyVoice latent ≠ Audio8 latent，私有空间不兼容
- **跨 TTS 通用交换格式**：Reference WAV + Reference transcript + VoiceProfile metadata
- 官方 VoiceDesign 接口天然产出此资产（voice description + target text → WAV）

## VoiceIdentity 包结构

```text
voices/role_yaolao/
├── manifest.json          # voiceIdentityId/roleId/source(desc+seed)/reference/derived
├── reference/canonical.wav + canonical.txt
├── cosyvoice/voicepack.bin  （角色主路）
└── audio8/reference_profile.bin（旁白/fallback，非默认）
```

## Gate 编号（VD 系列）

```text
VD-M0  PyTorch VoiceDesign baseline      [DONE]
VD-M1  Graph export                       [DONE]
VD-M2  MNN single modules                 [DONE]
VD-M3  SM8850 single-module equivalence   [DONE]
VD-M4  PC full VoiceDesign closure        [进行中] ← 当前
VD-M5  SM8850 full VoiceDesign closure    [待]
VI-M1  VoiceIdentity artifact             [待]
VI-M2  VD→CosyVoice VoicePack             [待]
VI-M3  VD→Audio8 profile                  [待]
VI-M4  identity consistency Gate          [待]
Voice Bridge Gate                          [待]
```

## 当前 VD-M4 卡点

- GraphB static_t35 图契约 = KV 固定 35 有效（34 prefill + 1 当前）
- torch 轨迹从空 KV 起步（cp=34 但 KV 只有 1）→ pad 后 cos 0.959 ≠ 1.0
- 修复：G4-A 重转动态 KV 版 GraphB / G4-B GraphA prefill 捕获权威 KV
