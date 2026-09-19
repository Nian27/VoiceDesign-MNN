#!/usr/bin/env bash
# 打 v1.1.0 的「合体版增量包」：把设计权重从独立样例那一套对齐到合体 App 那一套。
# 前提：设备已连 adb；本机有 vd-weights-staging/ 与 vd-weights-ext/。
set -euo pipefail
ADB="${ADB:-/c/Users/Administrator/AppData/Local/Android/Sdk/platform-tools/adb.exe}"
PKG=com.cosyvoice.app.debug
STAGE=/e/AndroidStudioProjects/vd-weights-staging
EXT=/e/AndroidStudioProjects/vd-weights-ext
OUT=/e/AndroidStudioProjects/vd-v110-delta
ZIP=/e/AndroidStudioProjects/vd-weights-out/vd-integrated-delta.zip

rm -rf "$OUT"; mkdir -p "$OUT"

# 1) 19 个同名文件（除 2.82 GB 的 GraphB 常量池，它复用已发布的 part0/part1）
for f in graphb28_v6_fp16.mnn codec_emb_fp16.mnn codec_emb_weight.f32 lm_head_weight.f32 \
         prompt_emb.f32 talker_codec_emb.f32 talker_rope_cos.f32 talker_rope_sin.f32 \
         codepred_rope_cos.f32 codepred_rope_sin.f32 text_embedding.fp16 \
         text_proj_fc1.fp16 text_proj_fc1_bias.f32 text_proj_fc2.fp16 text_proj_fc2_bias.f32 \
         tokenizer.bin; do
  cp -v "$STAGE/$f" "$OUT/$f"
done

# 2) 6 个差异文件（合体 App 用的是 .mnn + .mnn.weight 拆开的那一套）
for f in codec_head_fp16.mnn codec_head_fp16.mnn.weight \
         codepred_prefill_fp16.mnn codepred_prefill_fp16.mnn.weight \
         codepred_step_fp16.mnn codepred_step_fp16.mnn.weight \
         frame_emb_fp16.mnn; do
  cp -v "$EXT/$f" "$OUT/$f"
done

# 3) 设备上的 t96（本机没有对应物，必须从设备拉）
echo "拉取 tokenizer_decoder_static_t96.mnn (456,941,592 B) …"
"$ADB" exec-out "run-as $PKG cat files/cosyvoice3-mnn/voicedesign/tokenizer_decoder_static_t96.mnn" \
  > "$OUT/tokenizer_decoder_static_t96.mnn"
echo "bytes=$(stat -c%s "$OUT/tokenizer_decoder_static_t96.mnn") 期待=456941592"

# 4) 校验和 + 打包（<2 GB，可直接放 Release）
( cd "$OUT" && sha256sum * > SHA256SUMS.txt )
rm -f "$ZIP"
( cd "$OUT" && zip -0 -X "$ZIP" SHA256SUMS.txt ./*.mnn ./*.mnn.weight ./*.f32 ./*.fp16 ./*.bin 2>/dev/null || zip -0 -X -r "$ZIP" . )
echo "zip bytes=$(stat -c%s "$ZIP")  上限 2000000000"
