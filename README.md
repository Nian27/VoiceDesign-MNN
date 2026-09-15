# VoiceDesign-MNN

> **Qwen3-TTS VoiceDesign 1.7B 端侧化**：在 Android arm64 手机上，**纯文字**设计音色并生成语音。
> 不需要任何参考音频 —— 用自然语言描述音色（"苍老女性，低沉略沙哑，语气沉稳"），模型当场凭空造出一个音色。

**真机验证**：Honor Magic8 Pro（BKQ-AN90 / SM8850 / Android 17 / arm64-v8a），OpenCL(GPU) 后端，端到端跑通。

---

## 1. 这是什么

| | |
|---|---|
| **输入** | 两段文字：`音色描述`（instruct）+ `要说的内容`（text） |
| **输出** | 一段 WAV（24 kHz / 单声道 / 16-bit） |
| **不需要** | 参考音频 / 预置说话人 / 网络 / 云端 —— 统统不需要 |
| **模型** | Qwen3-TTS-12Hz-1.7B-VoiceDesign（talker 28L + code_predictor + 12.5Hz codec decoder） |

### 与其它 TTS 路线的区别

| 模式 | 需要什么 | 说明 |
|---|---|---|
| **VoiceDesign**（本项目） | **只要文字** | 用文字**创造**音色。`tts_model_type=voice_design`，`spk_id={}`（连预设音色都没有） |
| Voice Clone | 参考音频 + 文本 | 克隆已有音色（`generate_voice_clone`） |
| Custom Voice | 预设说话人名 | 用内置音色（`generate_custom_voice`） |

> **术语澄清**：早期笔记把产物叫 "reference.wav" 是**错误叫法**，容易与"参考音频"混淆。
> 正确叫法是 **designed voice sample（音色样张）** —— 它是"造出来的"，不是"拿来当参考的"。

---

## 2. 整体架构

### 2.1 生成链（16 码本 / 帧 @ 12.5 Hz）

```text
文字 --tokenize--> prompt_emb (L x 2048)
                      |
                      v
        +-------- L 次 GraphB 单步（prefill，写 KV slot 0..L-1）--------+
        |                                                              |
        v                                                              |
   [ talker 28L · hidden 2048 · KV cache 768 slot ]                   |
        |                                                              |
        +--> codec_head --> logits(3072) --采样--> code0                |
        |                                          |                   |
        |                                          v                   |
        |                              codepred prefill (2 tok)        |
        |                                          |                   |
        |                              codepred step x14 (KV cap 16)   |
        |                                          |                   |
        |                                   code1..code15              |
        |                                          |                   |
        +-------------- frame_emb([code0..15]) <---+                   |
                      |                                                |
                      +--> GraphB step (cp = L+n) ---------------------+
                                    |
                          code0 == 2150 (EOS)?
                                    |
                                    v
                    tokenizer_decoder (静态 300 帧) --> WAV
```

### 2.2 关键设计决策

| 决策 | 原因 |
|---|---|
| **host 采样**（不放图里） | 原方案图里用 `torch.argmax` 做 AR 前缀 → 与官方采样语义不符，且放大 stage-0 误差（**Bug D**） |
| **codepred 走 host 驱动 + 自带 KV cache** | 原方案把 15 级展开成一张大图（O(n^2) attention），慢且浪费。改成 prefill+step 后 KV 只需 16 |
| **只导出 prefill + step 两张图** | 接口稳定、可复用；lm_head / embedding 查表留在 host（fp32，无精度损失） |
| **大表一律 mmap** | 622MB / 126MB / 252MB 的表整块 `read()` 会把 App 内存压爆（实测 VmHWM 2.08GB 后挂死） |
| **prefill 复用 GraphB 自身** | L 次单步写 KV slot 就是 prefill。省掉一整张 2.8GB 的 GraphA |

### 2.3 codepred 契约（官方语义，已逐位对齐）

```text
官方 forward 语义：
  prefill:  inputs_embeds = cat(past_hidden, code0_emb)  -> generation_steps = 0
  之后第 g 步: inputs_embeds = codec_embedding[g-1](code_g)
  logits   = lm_head[g](hidden)

所以：
  stage 0 (prefill) -> lm_head[0] -> code1
  stage g (step)    -> lm_head[g] -> code_{g+1}
```

---

## 3. 模型清单

完整 hash 见 release-manifest.json。

| 文件 | 大小 | 说明 |
|---|---|---|
| `graphb28_v6_fp16.mnn` + `.weight` | 0.5 MB + **2.82 GB** | talker 28L 单步（KV 768 slot） |
| `codepred_prefill_fp16.mnn` | 162 MB | code_predictor prefill（2 token） |
| `codepred_step_fp16.mnn` | 162 MB | code_predictor step（KV cap 16） |
| `tokenizer_decoder_static_t300.mnn` | 457 MB | 12.5Hz codec → 波形（静态 300 帧） |
| `frame_emb_fp16.mnn` | 138 MB | 16 码本 → 2048 维帧嵌入 |
| `codec_head_fp16.mnn` / `codec_emb_fp16.mnn` | 12.6 MB x2 | codec logits / codec 嵌入 |
| `lm_head_weight.f32` | 126 MB | 15 个 lm_head（host fp32） |
| `codec_emb_weight.f32` | 252 MB | 15 个 codec embedding 表（host fp32） |
| `text_embedding.fp16` | **622 MB** | 文本词表 151936x2048 |
| `text_proj_fc1/fc2.fp16` + `_bias.f32` | 8.4 MB x2 + 8 KB x2 | text_projection MLP（**bias=True，别漏**） |
| `talker_codec_emb.f32` | 25 MB | talker 的 codec embedding（3072x2048） |
| `tokenizer.bin` | 3.5 MB | 自研紧凑分词器（vocab 151643 + merges 151387 + added 33） |
| `talker_rope_cos/sin.f32` | 192 KB x2 | 合并后的 mRoPE 表（384 位置） |
| `codepred_rope_cos/sin.f32` | 8 KB x2 | codepred RoPE（16 位置） |

**总计 ~4.7 GB**，部署在 App 私有目录，**不打进 APK**。

---

## 4. 构建

### 4.1 前置

```text
Android SDK + NDK r27 (27.0.12077973)
MNN 3.6.1 源码（MNN_HEXAGON=ON 或 MNN_OPENCL=ON 的构建）
Python 3 + torch（仅导出模型时需要）
Qwen3-TTS-12Hz-1.7B-VoiceDesign 原始权重
```

### 4.2 编译 App

```bash
./gradlew :app:assembleDebug
```

`app/src/main/cpp/CMakeLists.txt` 里需要改这两个路径：

```cmake
set(MNN_SRC "你的/MNN-3.6.1")
set(PREBUILT .../jniLibs/${ANDROID_ABI})
```

### 4.3 导出模型（可选，仓库不含权重）

```bash
python scripts/vd_b28_export_v6.py      # GraphB 28L
python scripts/vd_cpr1_export2.py       # codepred prefill + step
python scripts/vd_app_prep.py           # rope 表 + prompt_emb
python scripts/vd_text_prep.py          # 分词器 + text_embedding + projection
python scripts/vd_codec_emb.py          # talker codec embedding
```

---

## 5. 部署（模型不进 APK）

App 读的是**自己私有目录**，adb 无法直接写（属主/权限问题），必须用 `run-as` 管道：

```bash
PKG=com.voicedesign.app
# 1) 先推到 /data/local/tmp（shell 可读）
adb push graphb28_v6_fp16.mnn.weight /data/local/tmp/vd/
# 2) 用 run-as 以 App 身份写入
cat /data/local/tmp/vd/graphb28_v6_fp16.mnn.weight | \
  adb shell "run-as $PKG sh -c 'cat > files/voicedesign/graphb28_v6_fp16.mnn.weight'"
```

> **不要**直接把文件 `cp` 到 `/sdcard/Android/data/<pkg>/files/` —— adb(shell) 建的目录属主是 `shell:ext_data_rw`，
> App **没有权限读**（实测 `run-as ... ls` → Permission denied）。要用 `filesDir`（`/data/user/0/<pkg>/files/`）。

---

## 6. 运行

```kotlin
val eng = VoiceDesignEngine()
VoiceDesignEngine.prepareAdsp(applicationInfo.nativeLibraryDir)   // 见第 8 节
eng.load(modelDir, backend = "opencl", nativeLibDir = applicationInfo.nativeLibraryDir)

val status = eng.run(
    modelDir   = modelDir,
    outWavPath = "/data/user/0/com.voicedesign.app/files/voicedesign/out.wav",
    maxFrames  = 300,
    instruct   = "苍老女性，低沉略沙哑，声音有岁月感，语气沉稳、有威严，但并不凶狠。",
    text       = "小家伙，修炼之事急不得，根基若是不稳，日后吃亏的还是你自己。",
    lang       = 2055,          // codec_language_id["chinese"]
)
// status: "OK steps=109 stop=EOS prefill=71249ms decode=272627ms decoder=57076ms wav=209280 samples (8.72s) finite=yes ..."
```

### 后端选择

| backend | 真机结果 |
|---|---|
| `opencl` | **完整跑通**（GPU） |
| `cpu` | 可跑（慢，RSS ~7GB 易被杀） |
| `htp` (NPU) | App 内 DSP stop-execute / 挂死（CLI 探针正常，见 docs/NPU_HEXAGON_NOTES.md） |

> **推荐**：VoiceDesign 走 **OpenCL**，把 **NPU 留给 LLM**（同机的 ReaderDirector / Qwen 推理）。

---

## 7. 实测性能（Honor Magic8 Pro）

| 阶段 | OpenCL | 说明 |
|---|---|---|
| `load` | 1.6-6.1 s | 含 4.7 GB mmap |
| `tokenize + prompt` | ~10 s | 62 token 的 projection 是纯 C++ 标量实现，可优化 |
| `prefill` | **71 s** / 60-62 token | ~1.15 s/token |
| `decode` | **273 s** / 109 帧 | ~2.5 s/帧 |
| `decoder` | 57 s | 静态 300 帧 |
| **合计** | **~7 分钟 / 条** | RTF ~47 |

参考：同一设备上 **HTP 的 GraphB 单步只要 126 ms**（CLI 探针实测），codepred 整帧 71 ms —— 所以 **App 内 HTP 一旦修好，帧时间可从 2.5 s 降到 ~0.13 s**。

---

## 8. 踩坑与修复（摘要）

完整版见 docs/PITFALLS_AND_FIXES.md。两个"真凶"：

### 真凶 1：`text_projection` 是 `bias=True` 的 MLP，漏了 bias

```text
缺 bias:  fc1 stage cos = 0.9876823   full cos = 0.8335207
加 bias:  full cos = 1.0000000        maxabsdiff = 0.000000
```

后果：设备构造的 prompt 全错（cos 0.78）→ 生成退化 → **永不吐 EOS**。
（当时误判成"缺 repetition_penalty"，白追了一轮。）

### 真凶 2：`mmap` 长度把 float **个数**当 **bytes**

```cpp
mmap(NULL, 15*2048*1024, ...)      // 错：只映射了 1/4
mmap(NULL, 15*2048*1024*4, ...)    // 对
```

后果：索引一超 1/4 就越界 → `SIGSEGV(SEGV_MAPERR)`，且**故障地址正好页对齐**（映射末端）—— 这个特征就是判据。

### 其余 5 个

| # | 问题 | 修法 |
|---|---|---|
| 3 | `ADSP_LIBRARY_PATH` 只设了 App lib 目录 | 必须**分号分隔**且含 `/vendor/lib/rfsa/adsp;/system/lib/rfsa/adsp`，且要在 **native 库加载之前**（Kotlin `Os.setenv`） |
| 4 | 后端 stop-execute 时 `onForward` 返回空 vector | 7 处 `outOk()` 防御，报 `ERR BACKEND_STOP_EXECUTE: <模块>` |
| 5 | tts special token 名字想当然 | 真名是 `<tts_text_bos>` / `<tts_text_eod>` / `<tts_pad>`（**不带竖线**） |
| 6 | 622MB 文本表整块读入 | 改 mmap（VmHWM 从 2.08GB 降下来） |
| 7 | 缺 `repetition_penalty=1.05` | 补上，code0 不再重复 |

---

## 9. 验证（Gate）

完整数据见 docs/VALIDATION.md。

```text
(1) 分词器    设备 C++ vs HuggingFace       10/10 逐 id 相同          PASS
(2) Prompt    设备构造 vs PC 参考           cos = 1.0000000          PASS
                                            maxabsdiff = 1.997e-06
(3) GraphB    28L 单步 CPU vs HTP           hidden cos 0.9983157     PASS
(4) codepred  prefill + 14 级 forced-prefix hidden min 0.9999223     PASS
                                            logits min 0.9999907
                                            top1 14/14 一致
(5) 端到端    真机文字 -> WAV               8.72s, finite, stop=EOS   PASS
```

---

## 10. 已知限制

1. **速度**：OpenCL 上 ~7 分钟/条（RTF 47）。瓶颈是 GPU 上 2.8GB 权重的带宽。
2. **HTP 在 App 内不可用**：CLI 探针正常，App 内 DSP stop-execute。已排除 `ADSP_LIBRARY_PATH`。
3. **单条生成**：无批量、无流式。
4. **decoder 固定 300 帧**：输出上限 300x1920/24000 = 24 s；超过会被截断。
5. **prompt 上限 512 token**：`buildPrompt` 里 `maxTok=512`，长文本会返回 `ERR prompt build failed`。
6. **`buildPrompt` 缺越界保护**：文本很长时 body 段可能超过 `maxTok`（只有 instruct 段做了检查）。

---

## 11. 路线图

- [ ] **修 HTP**：查清 App 内 DSP 失败原因（帧时间 2.5s -> ~0.13s）
- [ ] **提速 OpenCL**：GraphB 量化（int8/int4），权重带宽降 2-4 倍
- [ ] **prompt 构建加速**：projection 用 NEON，或导出成 MNN 小图
- [ ] **界面**：进度条、音色历史、批量生成、分享
- [ ] **多角色**：与 ReaderDirector 结合，按角色设计不同音色

---

## 12. 目录结构

```text
VoiceDesign-MNN/
├── README.md                    本文件
├── AGENTS.md                    Agent 上下文约定
├── MEMORY.md                    关键结论速查
├── release-manifest.json        模型清单 + hash
├── app/                         Android 应用模块（Kotlin + JNI）
│   ├── build.gradle.kts
│   └── src/main/
│       ├── AndroidManifest.xml
│       ├── java/com/voicedesign/app/
│       │   ├── VoiceDesignActivity.kt    输入界面 + 日志
│       │   ├── VoiceDesignService.kt     前台服务 + 自动播放
│       │   └── VoiceDesignEngine.kt      native 封装
│       └── cpp/
│           ├── voicedesign_jni.cpp       整链 JNI
│           ├── vd_text.cpp/.h            分词器 + prompt 拼装
│           └── CMakeLists.txt
├── mnn-jni/                     同上（单独一份便于复用）
├── probes/                      诊断探针（CLI 形态，可独立跑）
│   ├── cp_r3_probe.cpp          codepred prefill+step CPU vs HTP
│   ├── cp_r4_probe.cpp          codepred 自驱 16 码轨迹
│   └── vd_full_chain.cpp        整链 CLI（子进程方案备用）
├── scripts/                     导出与验证
└── docs/
    ├── ARCHITECTURE.md
    ├── DEVELOPMENT_STORY.md
    ├── PITFALLS_AND_FIXES.md
    ├── VALIDATION.md
    ├── BUILD_AND_DEPLOY.md
    └── NPU_HEXAGON_NOTES.md
```

---

## 13. 许可与致谢

- 本项目代码：**Apache-2.0**（见 LICENSE）
- 模型权重来自 **Qwen3-TTS**（Alibaba Qwen），遵循其原始许可，**本仓库不包含权重**
- 推理引擎：[**MNN**](https://github.com/alibaba/MNN)（Apache-2.0）
- 参考项目：[**Nian27/CosyVoice3-MNN**](https://github.com/Nian27/CosyVoice3-MNN)（同设备路线，工程结构参考）
- Hexagon 配方来自 MNN 仓库 `source/backend/hexagon/README.md` 与官方 `apps/Android/MnnLlmChat/.../QnnModule.kt`

**免责**：本项目为工程验证用途。用他人音色生成语音涉及法律与伦理问题，请遵守当地法规。
