# NPU / Hexagon 专题笔记

**结论先行**：**HTP(NPU) 在 App 内不可用；在 CLI 探针里完全可用。**

| 环境 | HTP 结果 |
|---|---|
| CLI 可执行文件（`adb shell` 里直接跑） | **可用**。GraphB 126 ms/step，RSS 1.29 GB；codepred 整帧 71 ms |
| Android App（JNI + OpenCL 构建） | **不可用**。DSP stop-execute / 挂死 / 空 VARP 崩溃 |
| Android App（换成 HEXAGON 构建的 libMNN） | 同上 |

因此本项目的产品决策是：**VoiceDesign 走 OpenCL(GPU)，把 NPU 留给 LLM。**

---

## 1. 已排除的原因

### 1.1 `ADSP_LIBRARY_PATH`（已修，但仍不通）
见 PITFALLS P4。修完后从"静默挂死"变成"可诊断的崩溃"，但 HTP 仍未跑通。

### 1.2 进程 SELinux 域
CLI 跑在 `shell` 域，App 跑在 `untrusted_app` 域。
DSP 的 FastRPC（`libcdsprpc.so`）在 `untrusted_app` 下可能受 dma-buf / 内存配额限制。
**未证实，是下一步的排查方向。**

### 1.3 会不会是图本身有问题？
不会。同一批 `.mnn` 在 CLI 里跑出了正确结果（见 VALIDATION V4/V5）。

---

## 2. 关键实测数据（CLI，仍然有效）

```text
GraphB v6 28L 单步：
  延迟   HTP ~126 ms/step   vs  CPU ~1190 ms/step   (9.4x)
  内存   HTP 1.29 GB        vs  CPU 6.08 GB
  hidden cos 0.9983157  cur_k 0.9999233  cur_v 0.9983765

codepred（prefill + 14 step）：
  prefill 1.95 ms   step 稳态 ~5.3 ms   整帧 ~71 ms   RSS 367 MB
```

对比 App 内 OpenCL 的 2.5 s/帧 —— **HTP 一旦修好是 20 倍提速**。

---

## 3. 设备/HTP 环境要点

### 3.1 库的摆法
必须放在**私有 lib 目录**，不要动 `/data/local/tmp/MNN/libMNN.so`
（该文件可能被并行工作流替换，导致 shdr 失效）：
```text
/data/local/tmp/qwen3tts/vd_m5b/lib/      libMNN.so / libMNN_Express.so / libMNN_htpops.so
/data/local/tmp/qwen3tts/vd_m5b/adsp/     libMNN_htpops_skel.so
```
运行时：
```bash
LD_LIBRARY_PATH=<lib>:. ADSP_LIBRARY_PATH=<adsp> ./probe ...
```

### 3.2 `libMNN_htpops.so` 找不到会怎样
`dlopen failed: library not found` → 放进私有 lib 目录后，DSP 算子集变丰富，但输出不变。

### 3.3 我踩过并绕开的 3 个 MNN Hexagon lowering bug

| 代号 | 现象 | 绕法 |
|---|---|---|
| **Bug A** | stride-3 交错的 mRoPE slice/scatter 降低错误 | **host 预先交错好 RoPE** 再喂图 |
| **Bug B** | fp16 下 `-inf` 掩码被钳到 65504 | **host 做 mask**，用 `-1e4` 代替 `-inf` |
| **Bug C** | `ScatterND` / 索引覆写**不保留目标原值** | 避免 `ScatterND`，改用 **one-hot mask 广播加** |

> 注：`transpose` / `matmul` **从未被证明有 bug** —— 当时那个"幽灵"是 Bug C 造成的假象。

**上游没有修这三个**：对比 `MNN-master` 与 `MNN-3.6.1`，`HexagonRaster.cpp / HexagonBinary.cpp /
HexagonAttention.cpp / HexagonKVCacheManager.cpp` 的 diff **全为 0 行**。

### 3.4 被证伪的诊断手段（别再浪费时间）
| 手段 | 结果 |
|---|---|
| `MNN_HEX_HIDDEN_DUMP` 回读 | **垃圾数据**（hexdump 显示是 `89,90,91,92,...` 递增序列）|
| `MNN_HEX_ATTN_OFF=1` | **输出逐位不变** → 对标准 attention 无效 |

---

## 4. 下一步排查建议

1. 在 App 内抓 **FastRPC / adsprpc / dma-buf** 相关日志（`logcat | grep -iE "fastrpc|adsprpc|dma.?buf|cdsprpc"`）
2. 用 `dmabuf_dump <pid>` 看 App 的 DMA-BUF 用量（MNN `skills/hexagon/SKILL.md` 给的方法）
3. 试 `android:largeHeap="true"` 与 `<uses-native-library name="libcdsprpc.so"/>`（已加，无效）
4. 若确认是 `untrusted_app` 域限制，可考虑**子进程方案**（`probes/vd_full_chain.cpp` 就是为此写的）
   —— 由 App 解出一个可执行文件再 exec，进程环境与 CLI 探针一致

---

## 5. 有用的上游资料

| 资料 | 位置 |
|---|---|
| Hexagon 后端部署说明（**ADSP_LIBRARY_PATH 的权威写法**） | `MNN/source/backend/hexagon/README.md` |
| 官方 App 的 `Os.setenv` 写法 | `MNN/apps/Android/MnnLlmChat/.../QnnModule.kt:109` |
| Hexagon 优化/调试技能（含崩溃正则、DMA-BUF 测量法） | `MNN/skills/hexagon/SKILL.md` |
| QNN 调试技能 | `MNN/skills/qnn-debug/` |
| HVX FP16 分段线性激活（Sigmoid 16 段 / Tanh 12 / GELU 12 / SiLU 8） | `MNN/docs/perf/hexagon_pwl_activations.md` |

> 注意：MNN 仓库的 `AGENTS.md` 声明 `schema/private/` 与 `source/internal/` 为**内部专有代码，不得读取/修改/引用** —— 本项目遵守。

---

## 6. 崩溃诊断正则（来自 `skills/hexagon/SKILL.md`）

```bash
adb logcat -d | grep -aiE "execute_command_group_profile failed|qurt|sysfatal|fatal|crash|tlb|cdsp.*crash|adsp.*crash|segv|signal 11"
```

配合 tombstone 定位：
```bash
T=$(adb shell "ls -t /data/tombstones/tombstone_*[0-9] | head -1")
adb shell "grep -aE 'Cmdline|signal|#0[0-3]' $T"
# 映射到源码行（需要带调试符号的 .so）
llvm-addr2line -e libvoicedesign_jni.so -f -C 0x<pc>
```
