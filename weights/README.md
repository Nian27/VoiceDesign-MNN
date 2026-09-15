# 权重打包与分发

## 为什么不能直接进仓库

| 限制 | 数值 |
|---|---|
| GitHub 单文件（普通 git） | **100 MB** |
| GitHub Release 单个 asset | **2 GB** |
| 我们最大的单个文件 | **2.82 GB**（`graphb28_v6_fp16.mnn.weight`）|

所以：**权重不进 git，走 Release asset，且必须分包**。

## 分包方案（已验证可行）

```text
part1  权重-核心.zip   ~1.6 GB   graphb28_v6_fp16.mnn + .weight
part2  权重-解码.zip   ~1.5 GB   tokenizer_decoder_static_t300.mnn
                                frame_emb / codec_head / codec_emb
                                codepred_prefill / codepred_step
part3  权重-host表.zip ~1.0 GB   lm_head_weight.f32
                                codec_emb_weight.f32
                                text_embedding.fp16 (+proj)
                                talker_codec_emb.f32
                                tokenizer.bin + rope 表
```

每个都 **< 2 GB** ✓

## 打包

```bash
bash weights/pack_weights.sh <模型目录> <输出目录>
```

脚本会：
1. 校验 21 个文件是否齐全（按 `release-manifest.json`）
2. 分成 3 个 zip
3. 对每个 zip 算 **SHA-256**
4. 生成 `weights/CHECKSUMS.txt`

## 上传

仓库建好后，用 GitHub Release 上传（token 需要 `Contents: write`，本项目的 PAT 已具备 `push` 权限）：

```bash
TAG=v1.0.0
# 建 release
curl -X POST -H "Authorization: Bearer $T" \
     -H "Accept: application/vnd.github+json" \
     -d '{"tag_name":"'"$TAG"'","name":"VoiceDesign-MNN v1.0.0","body":"模型包 + APK"}' \
     https://api.github.com/repos/Nian27/VoiceDesign-MNN/releases

# 上传 asset（每个 < 2GB）
curl -X POST -H "Authorization: Bearer $T" -H "Content-Type: application/zip" \
     --data-binary @权重-核心.zip \
     "https://uploads.github.com/repos/Nian27/VoiceDesign-MNN/releases/<id>/assets?name=权重-核心.zip"
```

> ⚠️ 4.7 GB 上传耗时长（取决于上行带宽）。建议**后台跑 + 断点续传**，
> 或用 HuggingFace 托管（参考项目就是这么做的：`docs/HUGGINGFACE_README.md`）。

## 部署到手机

```bash
# 解包后，用 run-as 管道写进 App 私有目录（不要 cp 到 sdcard！）
for f in *; do
  cat "$f" | adb shell "run-as com.voicedesign.app sh -c 'cat > files/voicedesign/$f'"
done
```

详见 `../docs/BUILD_AND_DEPLOY.md` 第 5 节。
