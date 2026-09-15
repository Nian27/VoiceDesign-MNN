#!/usr/bin/env bash
# pack_weights.sh - 把 VoiceDesign 模型打成 <2GB 的分包，并生成 SHA-256 校验
# usage: bash pack_weights.sh <模型目录> <输出目录>
set -euo pipefail

SRC="${1:?usage: pack_weights.sh <模型目录> <输出目录>}"
OUT="${2:?usage: pack_weights.sh <模型目录> <输出目录>}"
mkdir -p "$OUT"

PART1=(graphb28_v6_fp16.mnn graphb28_v6_fp16.mnn.weight)
PART2=(tokenizer_decoder_static_t300.mnn frame_emb_fp16.mnn codec_head_fp16.mnn codec_emb_fp16.mnn codepred_prefill_fp16.mnn codepred_step_fp16.mnn)
PART3=(lm_head_weight.f32 codec_emb_weight.f32 text_embedding.fp16 text_proj_fc1.fp16 text_proj_fc2.fp16 text_proj_fc1_bias.f32 text_proj_fc2_bias.f32 talker_codec_emb.f32 tokenizer.bin talker_rope_cos.f32 talker_rope_sin.f32 codepred_rope_cos.f32 codepred_rope_sin.f32)

echo "== 1) 校验文件齐全 =="
miss=0
for f in "${PART1[@]}" "${PART2[@]}" "${PART3[@]}"; do
  if [ ! -f "$SRC/$f" ]; then echo "  MISSING: $f"; miss=$((miss+1)); fi
done
if [ "$miss" -ne 0 ]; then echo "缺 $miss 个文件，终止"; exit 1; fi
echo "  21/21 齐全"

echo "== 2) 打包（分 3 包，每个 <2GB） =="
cd "$SRC"
zip -q -0 "$OUT/vd-weights-part1-graphb.zip"     "${PART1[@]}"; echo "  part1 done"
zip -q -0 "$OUT/vd-weights-part2-decoder.zip"    "${PART2[@]}"; echo "  part2 done"
zip -q -0 "$OUT/vd-weights-part3-hosttables.zip" "${PART3[@]}"; echo "  part3 done"

echo "== 3) SHA-256 =="
cd "$OUT"
: > CHECKSUMS.txt
for z in vd-weights-part*.zip; do
  sz=$(stat -c %s "$z" 2>/dev/null || stat -f %z "$z")
  if [ "$sz" -gt 2000000000 ]; then echo "  !! $z 超过 2GB 上限 ($sz)"; fi
  h=$(sha256sum "$z" | cut -d' ' -f1)
  printf '%s  %s  %s bytes\n' "$h" "$z" "$sz" | tee -a CHECKSUMS.txt
done

echo
echo "== 完成，产物在 $OUT =="
ls -la "$OUT"
