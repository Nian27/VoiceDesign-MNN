# HTP 支线取证（VD-M5A/M5B，2026-08-26）

## 结论速览
- HTP-P0 = PASS：HTP 确实在真机执行（DSP 端 op 级 profile 为铁证）
- HTP-P1 = 未完成：first-divergence 尚未定位；已排除若干假设，且发现取证工具本身不可用

## HTP-P0 证据（PASS）

### 1. DSP 端 op 级 profile（只有 DSP 侧执行才会输出）
    I MNNJNI : Hexagon DSP Profile:
      DSPOpType CONV1X1_DIRECT_FP16 (17): 124.966 ms
      DSPOpType BINARY_ELEMENTWISE  (19): 16802.324 ms   <- 主导，异常
      DSPOpType RASTER_BLIT (3):           24.348 ms
      DSPOpType TENSOR_CONVERT (7):         3.373 ms
      DSPOpType TENSOR_CONVERT_TYPE0_NC4HW4_TO_NCHW (100): 3.153 ms
      DSPOpType LAYER_NORM_PACKED (9):      0.272 ms
      DSPOpType REDUCTION (29):            25.704 ms
    I MNNJNI : Hexagon onCopyBuffer Profile:
      HEXAGON_TO_CPU: calls=162, bytes=2142596, time=3174 ms, flush=3173 ms
      CPU_TO_HEXAGON: calls=126, bytes=1363042, time=0.378 ms

### 2. DSP RPC 会话确凿
    remote_handle_close_domain ... skel unload time 382 us
    dspqueue_close: destroying queues and signals for domain 100000

### 3. 发现并修掉的缺陷
    E MNNJNI : [MNN::Hexagon] Open libMNN_htpops.so, error=dlopen failed: library not found
  修复：把 htpops(+skel) 放进私有库目录
    私有 lib/libMNN_htpops.so
    私有 adsp/libMNN_htpops_skel.so
  修复后 DSP op 类型集合变丰富（出现 TENSOR_CONVERT / LOOP_BLIT / CAST），但输出数值不变。

### 启动模板（HTP）
    cd /data/local/tmp/qwen3tts/vd_m5a
    LD_LIBRARY_PATH=/data/local/tmp/qwen3tts/vd_m5b/lib:.
    ADSP_LIBRARY_PATH=/data/local/tmp/qwen3tts/vd_m5b/adsp
    ./graphb_module_probe <model> <cp> <emb> <kvk> <kvv> <pos> <outdir> 1 htp

## HTP-P1 现状（未定位）

### 已知数值
| 运行 | hidden maxabs | crc32 | cos vs ref |
|---|---|---|---|
| device CPU (golden) | 24.5427 | a009d558 | 0.9999731 PASS |
| device HTP | 14.8203 | dadd0af4 | 0.2038 FAIL |
| HTP + htpops 已加载 | 14.8203 | dadd0af4 | 不变 |
| HTP + MNN_HEX_ATTN_OFF=1 | 14.8203 | dadd0af4 | 不变 |

### 已排除
1. htpops 缺失：修好后数值不变 -> 不是自定义算子缺失导致
2. attention：MNN_HEX_ATTN_OFF=1 输出 crc 逐位相同 -> 该开关对本模型标准 attention 无效（为 LLM FA/LA 所写），未取得注意力是根因的证据
3. 性能：HTP 1049.7 ms 约等于 CPU 1056 ms，零加速；RSS 仅 212-595 MB

### 取证工具本身不可用（重要）
MNN_HEX_HIDDEN_DUMP=1 + MNN_HEX_LOGITS_DIR=<dir> 对 GraphB dump 出的是垃圾：
- 文件名/大小(4096B) 都对，但内容为 int32 递增序列 89,90,91,92,...（hexdump 实证）
- 即 debugReadback 指针未指向真实张量 -> 与 M2-3 记录的 readback 假象/循环论证同源
- logcat 中同批 f8 值同样不可信

### 可疑方向（下一步候选）
1. 5D 张量 (kv 28x1x8x768x128) 在 Hexagon 路径的 layout 处理：同模型用 Interpreter API 喂 5D 亦算错(cos -0.006)，而 Module API on CPU 正确 -> 两条错路都指向 5D 处理
2. BINARY_ELEMENTWISE 16.8 s：注意力掩码 positions<=cache_position 在 768 槽上广播(28x8x768x128 元素)，量级异常
3. 需要可信的 HTP 中间值 dump：按 M2-3 文档自身建议，改用 createHostTensorFromDevice(经 onCopyBuffer 写入独立 CPU buffer)，而非 mapped getPtr 直读

## 待办
- HTP-P1：建立可信逐层 dump（重建插桩 或 导出带逐层输出的 GraphB）
- HTP-P1：定位 first_divergence_layer / op
- HTP-P2：仅分析首个 divergence 的 producer/consumer
- 注：GraphB 逐层输出的 ONNX 导出超过 600 s 工具时限，需离线/分块方式
