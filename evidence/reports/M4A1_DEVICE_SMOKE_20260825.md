# M4-A1 tokenizer-decoder Android 真机 Smoke 报告（2026-08-25）

## 结论

**M4-A1 PASS（CPU 后端五证其四 + Backend 证待 HTP）——VoiceDesign 首次在 Android 真机端到端产出可用 PCM/WAV。**

- 设备：SM8850（adb 10.40.128.150:42759，API 37）
- 模型：tokenizer_decoder_static_t300.mnn（静态 T=300 变体，fp32，457MB push 后）
- 载体：audio8_mnn_probe（M4-A probe 级，非 M4-B ReaderVoice App）+ monolith libMNN.so (3.6.1)

## 五证明细

| Gate | 结果 | 证据 |
|---|---|---|
| Load | PASS | load_ms=1168（228MB .mnn） |
| Backend | CPU（HTP 待补） | execution_path=cpu, backend_actual=0；HTP context 生成卡 StridedSlice host 注册问题 → WSL 路线待跑 |
| Execute | PASS | run_ms=18120；finite 576000/576000；min -0.868/max 0.994/rms 0.105 |
| Numeric | PASS | vs PC-MNN 同模型同输入：cosine 0.999867 / mae 6.35e-04 / SI-SDR 35.41dB / rel_l2 0.017 / maxabs 0.203 |
| Audio | PASS | decoder_device_smoke.wav（24.0s @24kHz PCM16 可播放） |

## 性能

- 真机 CPU RTF = 0.755（24s 音频 / 18.1s 合成）；session_memory ≈ 1494MB
- 对选角师场景（低频一次性、3 候选）CPU 已可接受；HTP 用于进一步降 RTF/功耗

## 过程修复链（防重踩）

1. 动态 T=-1 被 probe 拒绝（input non-positive dim）→ 导出静态 T=300 变体（export_decoder_static_t300.py）
2. probe 与 libMNN ABI 配对：必须用 monolith 构建（build-qnn-plugin-android-arm64-16k-monolith）的 libMNN.so，
   plugin 版缺 RuntimeManager::destroy 符号
3. strip_isnan_generic.py 对静态版再移除 16×IsNaN（822 nodes）后 MNNConvert Success

## 产物

```text
artifacts/device-smoke/tokenizer_decoder_static_t300.{onnx,_nonan.onnx,.mnn}
artifacts/device-smoke/{decoder_codes_i32.raw, decoder_pcm_pc_f32.raw, decoder_pcm_dev_cpu_f32.raw}
artifacts/device-smoke/decoder_device_smoke.wav
artifacts/device-smoke/decoder_pc_meta.json
```

## 边界（诚实声明）

- 本 Gate = M4-A（probe/runtime 级）。M4-B（ReaderVoiceMobile Activity/JNI/播放最小 App）另立 Gate。
- Backend 证 = CPU。HTP 四证（V81 skel/FastRPC/CDSP/dspqueue）需先解决 QNN context 生成的 WSL 路线。
- 数值参照 = PC-MNN 同模型（PC-MNN vs PyTorch 已另证 cos 0.999936），数值守恒成立。