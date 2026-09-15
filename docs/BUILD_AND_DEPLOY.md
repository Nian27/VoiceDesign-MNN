# 构建与部署

## 1. 环境

```text
Android SDK + NDK r27 (27.0.12077973)
CMake 3.22+
JDK 17
MNN 3.6.1（需要 OpenCL 或 Hexagon 后端，见第 3 节）
Python 3.10+ / torch / qwen_tts（仅导出模型时）
```

真机验证环境：Honor Magic8 Pro（BKQ-AN90 / SM8850 / Android 17 / arm64-v8a）。

## 2. 编译 App

```bash
./gradlew :app:assembleDebug
# 产物 app/build/outputs/apk/debug/app-debug.apk
```

`app/src/main/cpp/CMakeLists.txt` 里两处路径要改成你自己的：

```cmake
set(MNN_SRC "你的/MNN-3.6.1")
set(PREBUILT ${CMAKE_CURRENT_SOURCE_DIR}/../jniLibs/${ANDROID_ABI})
```

## 3. MNN 库怎么来

### 3.1 OpenCL 构建（**推荐**，本项目的验证路径）

```bash
cd MNN-3.6.1 && mkdir -p build-opencl-arm64 && cd build-opencl-arm64
cmake .. -DCMAKE_TOOLCHAIN_FILE=<NDK>/build/cmake/android.toolchain.cmake \
         -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-26 \
         -DMNN_OPENCL=ON -DMNN_BUILD_SHARED_LIBS=ON \
         -DMNN_SEP_BUILD=OFF -DMNN_BUILD_LLM=ON
make -j
# 需要：libMNN.so / libMNN_Express.so / libMNN_CL.so
```

### 3.2 Hexagon 构建（NPU，App 内暂时不可用）

```bash
cmake .. -DMNN_HEXAGON=ON ... 
# 额外需要：libMNN_htpops.so（AArch64 stub） + libMNN_htpops_skel.so（DSP 侧）
# skel 由 source/backend/hexagon/htp-ops-lib/build.sh v79 产出
```

> ⚠️ 若 `libMNN_htpops.so` 缺失，运行期会 `dlopen failed: library not found`。
> 把它放进**私有 lib 目录**，并确保 `ADSP_LIBRARY_PATH` 含 **skel 所在目录 + 厂商 rfsa 路径**。

### 3.3 jniLibs 要放什么

```text
app/src/main/jniLibs/arm64-v8a/
  libMNN.so
  libMNN_Express.so
  libMNN_CL.so              # OpenCL 后端
  libMNN_htpops.so          # 可选（Hexagon）
  libMNN_htpops_skel.so     # 可选（Hexagon）
  libc++_shared.so
```

## 4. 导出模型

```bash
# 需要先能 from qwen_tts import Qwen3TTSModel，并准备好原始权重目录
python scripts/vd_b28_export_v6.py     # -> graphb28_v6_fp16.mnn (+ .weight)
python scripts/vd_cpr1_export2.py      # -> codepred_prefill_fp16.mnn / codepred_step_fp16.mnn
python scripts/vd_app_prep.py          # -> talker_rope_cos/sin.f32, codepred_rope_*.f32, prompt_emb.f32
python scripts/vd_text_prep.py         # -> tokenizer.bin, text_embedding.fp16, text_proj_*.fp16
python scripts/vd_codec_emb.py         # -> talker_codec_emb.f32
```

导出后请核对：
```text
codepred_*.mnn 里 ArgMax / ScatterND / IsNaN 应全为 0
（用 MNNConvert 转完可以拿 netron 或 MNN 自带工具查；本项目实测 ArgMax=0 ScatterND=0）
```

## 5. 部署模型（关键：不要用 cp 到 sdcard）

### 5.1 错误做法
```bash
adb shell mkdir -p /sdcard/Android/data/<pkg>/files/voicedesign
adb push model.bin /sdcard/Android/data/<pkg>/files/voicedesign/    # 看似成功
```
结果：App **读不到**（目录属主 `shell:ext_data_rw`，App uid 不在其中）。

### 5.2 正确做法：run-as 管道

```bash
PKG=com.voicedesign.app
DST=files/voicedesign

# 1) 先推到 shell 可读的位置
adb shell mkdir -p /data/local/tmp/vd
adb push graphb28_v6_fp16.mnn.weight /data/local/tmp/vd/

# 2) 以 App 身份写进私有目录
adb shell "run-as $PKG mkdir -p $DST"
adb shell "cat /data/local/tmp/vd/graphb28_v6_fp16.mnn.weight | run-as $PKG sh -c 'cat > $DST/graphb28_v6_fp16.mnn.weight'"

# 3) 核验大小
adb shell "run-as $PKG ls -la $DST"
```

批量部署脚本见 `scripts/` 下的 shell（本项目用过的 20 文件版本）。

## 6. 安装（避坑）

```bash
# 设备可能有多用户（Honor 的"平行空间"）。显式指定 user 0：
adb push app-debug.apk /data/local/tmp/vd.apk
adb shell "pm install --user 0 -r /data/local/tmp/vd.apk"

# 核验装上了
adb shell "dumpsys package com.voicedesign.app | grep lastUpdateTime"

# 启动也要带 user
adb shell "am start --user 0 -n com.voicedesign.app/.VoiceDesignActivity"
```

> ⚠️ 这台设备装 APK 会弹权限确认框。如果不点"允许"，会看到
> `Failure [INSTALL_FAILED_ABORTED: User rejected permissions]`，
> 而**旧版本还在跑** —— 我就这么白跑过两轮。装完一定要用 `lastUpdateTime` 核验。

## 7. 运行与取结果

```bash
# 触发（后台服务跑整链）
adb shell "am start --user 0 -n com.voicedesign.app/.VoiceDesignActivity \
  --ez autorun true --es backend opencl \
  --es instruct '苍老女性，低沉略沙哑' --es text '今天天气不错'"

# 看进度（App 自己写的日志文件，比 logcat 可靠）
adb shell "run-as com.voicedesign.app cat files/voicedesign/vd_run.log"

# 取 WAV
adb exec-out run-as com.voicedesign.app cat files/voicedesign/<name>.wav > out.wav
```

> 日志为什么写文件：logcat 的 ring buffer 只有 256KB，会被 HTP/OpenCL 的打印刷爆。
> 需要时先 `adb logcat -G 16M`。

## 8. 常见故障速查

| 现象 | 先查 |
|---|---|
| `load failed` | 模型是否齐（20 个文件）、是否在 `filesDir` |
| `ERR prompt build failed` | 文字是否为空、`buildPrompt` 的 token 上限 |
| 卡在 load 不出日志 | 是否还有大表在用 `read()` 而非 mmap |
| 跑一半进程消失 | 内存策略（见 PITFALLS P9）；一次只跑一个重任务 |
| 装完行为没变 | `dumpsys package ... lastUpdateTime` 核验是否真装上 |
