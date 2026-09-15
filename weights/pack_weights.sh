#!/usr/bin/env bash
# pack_weights.sh - 把 VoiceDesign 模型打成 <2GB 的分包，并生成 SHA-256 校验
# usage: bash pack_weights.sh <模型目录> <输出目录>
#
# 约束: GitHub Release 单个 asset 上限 2GB
#   graphb28_v6_fp16.mnn.weight 单文件 2.82GB -> 拆成 .part0/.part1 并各自打包
#   部署时还原:  cat graphb28_v6_fp16.mnn.weight.part* > graphb28_v6_fp16.mnn.weight
set -euo pipefail

SRC="${1:?usage: pack_weights.sh <模型目录> <输出目录>}"
OUT="${2:?usage: pack_weights.sh <模型目录> <输出目录>}"
mkdir -p "$OUT"
SPLIT_BYTES=1500000000

PART2=(tokenizer_decoder_static_t300.mnn frame_emb_fp16.mnn codec_head_fp16.mnn codec_emb_fp16.mnn codepred_prefill_fp16.mnn codepred_step_fp16.mnn)
PART3=(lm_head_weight.f32 codec_emb_weight.f32 text_embedding.fp16 text_proj_fc1.fp16 text_proj_fc2.fp16 text_proj_fc1_bias.f32 text_proj_fc2_bias.f32 talker_codec_emb.f32 tokenizer.bin talker_rope_cos.f32 talker_rope_sin.f32 codepred_rope_cos.f32 codepred_rope_sin.f32)

echo "== 1) 校验 =="
miss=0
for f in graphb28_v6_fp16.mnn graphb28_v6_fp16.mnn.weight "${PART2[@]}" "${PART3[@]}"; do
  [ -f "$SRC/$f" ] || { echo "  MISSING: $f"; miss=$((miss+1)); }
done
[ "$miss" -eq 0 ] || { echo "缺 $miss 个"; exit 1; }
echo "  全部齐全"

echo "== 2) 拆大文件并校验可还原 =="
cd "$SRC"
rm -f graphb28_v6_fp16.mnn.weight.part*
split -b "$SPLIT_BYTES" -d -a 1 graphb28_v6_fp16.mnn.weight graphb28_v6_fp16.mnn.weight.part
cat graphb28_v6_fp16.mnn.weight.part* > /tmp/_rejoin.bin
a=$(sha256sum /tmp/_rejoin.bin | cut -c1-16); b=$(sha256sum graphb28_v6_fp16.mnn.weight | cut -c1-16)
rm -f /tmp/_rejoin.bin
[ "$a" = "$b" ] && echo "  还原校验 OK $a" || { echo "  FAIL $a != $b"; exit 1; }

echo "== 3) 打包（每包单独 zip，各 <2GB） =="
zip -q -0 "$OUT/vd-weights-1-graphb-mnn.zip"      graphb28_v6_fp16.mnn
zip -q -0 "$OUT/vd-weights-2-graphb-w-part0.zip"  graphb28_v6_fp16.mnn.weight.part0
zip -q -0 "$OUT/vd-weights-3-graphb-w-part1.zip"  graphb28_v6_fp16.mnn.weight.part1
zip -q -0 "$OUT/vd-weights-4-decoder.zip"         "${PART2[@]}"
zip -q -0 "$OUT/vd-weights-5-hosttables.zip"      "${PART3[@]}"
echo "  done"

echo "== 4) SHA-256 + 体积检查 =="
cd "$OUT"
: > CHECKSUMS.txt
bad=0
for z in vd-weights-*.zip; do
  sz=$(stat -c %s "$z")
  if [ "$sz" -ge 2000000000 ]; then echo "  !! $z 超 2GB ($sz)"; bad=1; fi
  h=$(sha256sum "$z" | cut -d' ' -f1)
  printf '%s  %s  %s\n' "$h" "$z" "$sz" | tee -a CHECKSUMS.txt
done
[ "$bad" -eq 0 ] && echo "  全部 <2GB OK" || echo "  有超限文件!"
echo
ls -la "$OUT"
