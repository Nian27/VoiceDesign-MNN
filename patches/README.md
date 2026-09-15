# MNN 补丁

本目录存放**对 MNN 源码的改动**，便于复现构建。

目前**没有必需的补丁** —— 本项目在**未修改**的 MNN 3.6.1 上跑通（OpenCL 后端）。

## 可选插桩（调优/诊断用，非必需）

用 Hexagon/NPU 后端时，本机调试曾加过这些插桩（**不是修复，只是诊断**）：

| 插桩 | 作用 | 结论 |
|---|---|---|
| `MNN_HEX_COPY_DIAG` | 打印 host→device 拷贝的 src CRC / dst fd / 偏移 | 有用 |
| `MNN_HEX_HIDDEN_DUMP` | 尝试回读中间张量 | **返回垃圾数据，勿依赖** |
| `MNN_HEX_ATTN_OFF` | 关闭 attention 优化 | **输出逐位不变，无效** |

## 已知的上游问题（未修，靠 host 侧绕开）

见 `../docs/NPU_HEXAGON_NOTES.md` 第 3.3 节：

- **Bug A**：stride-3 交错 mRoPE 的 slice/scatter 降低错误 → host 预交错
- **Bug B**：fp16 下 `-inf` 掩码被钳到 65504 → host 做 mask，用 `-1e4`
- **Bug C**：`ScatterND` 不保留目标原值 → 改 one-hot mask 广播加

> 对比 `MNN-master` 与 `MNN-3.6.1`：`HexagonRaster.cpp / HexagonBinary.cpp /
> HexagonAttention.cpp / HexagonKVCacheManager.cpp` 的 diff **全为 0 行** —— **上游没有修这三个**。
