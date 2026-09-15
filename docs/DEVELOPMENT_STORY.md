# 开发历程

按时间线记录**真实的推进过程**，包括走过的弯路。

## 阶段 0 · 目标

把 Qwen3-TTS-12Hz-1.7B-VoiceDesign 搬到 Android 手机上，**纯文字**设计音色。
与同机的 CosyVoice3 栈并行（那条线做克隆，这条线做"文字造音色"）。

## 阶段 1 · 图导出与 PC 对齐

把模型拆成可端侧执行的几张图。踩的坑：

- **Bug D**：第一版 `codepred_ar` 图里用 `torch.argmax` 做 AR 前缀（烘焙了 14 个 `ArgMax` 节点）
  → 与官方采样语义不符，且把 stage-0 误差逐级放大。
  **修**：改成 host 驱动采样。
- **Bug E**：attention 输出漏了 `.transpose(1,2)` 再 reshape。
  S=1（decode）时无害，**S=2（prefill）才暴露** —— codepred prefill cos 只有 0.09。
  **修**：补 transpose，之后 ONNX vs torch `hidden_last cos = 1.0000000`。
- 因果掩码写错（`am2[0,0,0,1]=0` 非因果）→ 修。

**结论**：CP-R1（重导）+ CP-R2（PC MNN vs Torch）**PASS**，15/15 codes 逐位相同。

## 阶段 2 · HTP（NPU）攻坚

真机跑 GraphB 28L，找到并绕开 3 个 MNN Hexagon lowering bug（Bug A/B/C，见 NPU 笔记）。
修完后 **1L 绿了**，再推到 28L 全绿：

```text
GraphB v6 28L 单步  HTP hidden cos 0.9983157   126 ms/step   RSS 1.29 GB
                     CPU hidden cos 1.0000000  1190 ms/step   RSS 6.08 GB
```

**但这里我犯了一个错**：在 CPU+HTP 并发跑两个进程时把设备内存打爆，设备卡死。
教训写进了纪律：**一次只跑一个重任务**。

## 阶段 3 · codepred 重导（CP-R1 → CP-R4）

原 `codepred_ar` 是"15 级全展开"的大图（506 MB，O(n^2) attention，每帧 1682 ms，
占整帧 93%）。**重新设计**为 host 驱动的 prefill + step，自带 KV cache，KV 容量只要 16。

```text
CP-R1  重导（host 采样 + KV cache + 无 ArgMax）   ArgMax=0 ScatterND=0   PASS
CP-R2  PC MNN vs Torch                          15/15 codes exact       PASS
CP-R3A prefill (S=2)          hidden 0.9999958  kv 0.9999997           PASS
CP-R3B forced-prefix 14 级    hidden min 0.9999223
                              logits min 0.9999907  top1 14/14         PASS
CP-R3  性能                   prefill 1.95ms / step 4.9ms / 整帧 71ms
                              旧图 1682ms -> 23.7x                     大幅改善
CP-R4  自驱 16 码轨迹         frame0 在 code_index=7 分叉（boundary flip）
```

**CP-R4 的判定**：分叉是 **sampling-boundary flip**（CDF 只差 2.7e-4），不是后端数学崩坏。
结论：**fp16 下无法逐位复现 CPU 轨迹，可接受**。

## 阶段 4 · 里程碑：真机 WAV

第一次端到端出 WAV（OpenCL 后端，8.48 s）。此时还只支持"预置 prompt"。

## 阶段 5 · 做进 App（这一步最难）

把链搬进 Android App，遇到一串问题（详见 PITFALLS）：

```text
P4  ADSP_LIBRARY_PATH 顺序+厂商路径   -> 从静默挂死变成可诊断
P5  空 VARP 崩溃                      -> 7 处防御
P2  mmap 长度单位错                   -> SIGSEGV 修复
P3  622MB 表整块读入                  -> 改 mmap
P9  厂商内存策略杀进程                -> 前台服务 + mmap
```

## 阶段 6 · 文字输入（真正的产品形态）

要支持任意文字，必须补齐三块：**分词器 + 文本嵌入 + prompt 拼装**。

```text
① Qwen2 BPE 分词器（C++ 从零实现）
   实现 GPT-2 byte-level 字母表 + 预分词正则 + merges 按 rank 合并
   验证：与 HuggingFace 逐 id 比对  10/10   PASS

② text_embedding (622MB) + text_projection (16.8MB) 导出

③ prompt 拼装（在 PC 上先复刻官方逻辑）
   验证：exact_equal = True（逐位相同）  PASS
```

**然后是本项目最贵的一个 bug**（P1）：

```text
设备构造的 prompt: cos vs PC 参考 = 0.7824020    <- 结构对(62 token)、内容全错
现象: code0 反复重复、永不吐 EOS
误判: 以为是缺 repetition_penalty（白追一轮）
真因: text_projection 是 bias=True 的 MLP，我漏了 bias
修后: cos = 1.0000000   maxabsdiff = 1.997e-06
```

## 阶段 7 · 闭环

```text
完成：结果 = OK steps=109 stop=EOS prefill=71249ms decode=272627ms decoder=57076ms
       wav=209280 samples (8.72s) finite=yes
完成：8.72 秒 / 418604 字节 —— 正在播放
播放结束
```

用户在手机上自己输入"苍老女性，低沉略沙哑…"，得到 8.72 s 语音。

---

## 复盘：哪些判断错了

| 当时的判断 | 实际 |
|---|---|
| "HTP 的 stage0 codepred 误差是材料级扰动" | **错**。是 Bug D/E 的产物，修完 cos 从 0.9956 升到 0.99999 |
| "永不吐 EOS 是因为缺 repetition_penalty" | **错**（是其中一个因素，但主因是 bias 缺失导致 prompt 全错） |
| "mmap 失败会报错" | **错**。它静默返回一个过短的映射，直到越界才崩 |
| "App 内 HTP 挂死是 ADSP_LIBRARY_PATH 问题" | **部分对**。修完不挂了，但换成了空 VARP 崩溃，至今未通 |
| "把模型 cp 到 sdcard 的 app 目录就能用" | **错**。属主不对，App 读不到 |

## 有效的方法论（值得复用）

1. **逐位对齐**优先于"看起来对"：cos=1.0 / exact_equal=True 才是证据
2. **dump 中间产物**再比对：prompt、hidden、kv、logits 都 dump 过
3. **addr2line 定位崩溃**：带调试符号编译，崩溃点精确到源码行
4. **故障地址的形态是线索**：页对齐 → 映射边界；0x0 → 空指针
5. **先二分再深挖**：加 `M1..M8` 标记，一次就能定位到具体语句
6. **换环境对比**：CLI 能跑、App 不能 → 差异在进程环境，不在图
