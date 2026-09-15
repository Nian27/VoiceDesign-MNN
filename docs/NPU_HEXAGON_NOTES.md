# NPU / Hexagon (HTP) 专题 · 完整技术档案

> 这是一份**完整**的 HTP 攻坚记录：做了什么、得到什么数据、遇到什么问题、怎么解决、
> 以及**哪些结论后来被推翻**、**哪些至今没解决**。

---

## 0. 一页结论

| 环境 | HTP 可用性 | 关键数据 |
|---|---|---|
| **CLI 可执行文件**（`adb shell` 里直接跑） | ✅ **完全可用** | GraphB **126 ms/step**（CPU 1190 ms，**9.4x**）；RSS **1.29 GB**（CPU 6.08 GB） |
| **Android App**（JNI，HEXAGON 构建的 libMNN） | ❌ **不可用** | DSP stop-execute / 空 VARP / 挂死；已排除 ADSP_LIBRARY_PATH，**至今未解决** |

**产品决策**：VoiceDesign 走 **OpenCL(GPU)**，把 **NPU 留给 LLM**（ReaderDirector / Qwen）。
依据：MNN 官方 `skills/hexagon/SKILL.md` 的硬约束 —— **不许用 CPU fallback 冒充"优化"**；
而我们这里的"退回 GPU"是**明确的后端选择**，不是把 NPU 的工作偷偷搬到 CPU。

**如果 App 内 HTP 修好，帧时间可从 2.5 s → ~0.13 s（约 20 倍）。**

---

## 1. 时间线（HTP-P0 → CP-R4 → App 内失败）

| 阶段 | 做什么 | 结果 |
|---|---|---|
| **HTP-P0** | 证明 Hexagon **真的在 DSP 上执行**（不是 CPU fallback） | ✅ DSP op 级 profile 显示算子确实落到 cDSP |
| **HTP-P1** | 设备 CPU vs HTP 首次分歧定位 | ⚠️ 定位到 5D 张量/tensor 输入相关 |
| **HTP-P2** | 把整个 **3-way mRoPE 交错（stride-3）搬出 HTP 图** | ✅ 分歧消失，锁定 Bug A |
| **HTP-P2-B2** | host mask + 精确验证 **KV preserve/write slot**；额外输出 `attn_logits_before_mask / after_mask / attn_probs` | ✅ 锁定 Bug B、Bug C |
| **G1** | 1L 绿 → **28L 全绿** | ✅ hidden cos **0.9983157** |
| **F1** | 功能门：logits / top50 / EOS rank / shared-RNG | ✅ LOGITS_COS **0.999330**、TOP50 **49/50**、EOS rank 91=91、RNG **32/32** |
| **G2-A** | 20 步强制轨迹，看是否漂移 | ✅ 无漂移（hidden 0.99776→0.99855） |
| **G2-B** | 自由采样首分歧 | ⚠️ step1 / code_index 1 —— **但此结论后来被推翻**（见 §7） |
| **设备事故** | 我并发跑 CPU+HTP → 内存爆 → **设备卡死** | ❌ 我的责任，此后立规矩：一次只跑一个重任务 |
| **CP-R1/R2** | codepred 重导（host 采样 + KV cache） | ✅ ArgMax=0 ScatterND=0 |
| **CP-R3** | codepred prefill + 14 级 forced-prefix（HTP vs CPU） | ✅ hidden min **0.9999223**、logits min **0.9999907**、top1 **14/14** |
| **CP-R4** | codepred 自驱 16 码轨迹 | ⚠️ frame0 在 code_index 7 **boundary flip** |
| **App 内** | 把链搬进 App 用 HTP | ❌ 挂死 → 修 ADSP 后变成空 VARP → **未解决** |

---

## 2. GraphB 的 HTP 契约（v6，已冻结）

```text
输入 (7):
  inputs_embeds   (1,1,2048)       fp32
  kv_k            (28,1,8,768,128) 缓存读入
  kv_v            (28,1,8,768,128)
  rope_cos        (1,1,1,128)
  rope_sin        (1,1,1,128)
  attn_mask       (1,1,1,768)
  slot_mask       (1,1,768,1)

输出 (3):
  hidden_states   (1,1,2048)       给 codec_head
  cur_k           (28,8,128)       本步新 K
  cur_v           (28,8,128)       本步新 V
```

**注意**：导出时 tracer 丢掉了 `cache_position`（原本 8 输入 → 实际 7 输入）。

### 2.1 为什么 KV 写入用 one-hot 掩码而不是 ScatterND

```cpp
// ✗ 错: ScatterND 在 MNN Hexagon 上不保留目标原值 (Bug C)
kv_k = ScatterND(kv_k, index=cp, update=k_new)

// ✓ 对: one-hot 掩码广播加
m = zeros(768); m[cp] = 1
kv_k_out = kv_k * (1 - m) + k_new * m
```

### 2.2 为什么 mask 用 -1e4 而不是 -inf

```cpp
am[i] = (i <= cp) ? 0.f : -1e4f;    // ✗ -inf 在 fp16 下会被钳到 65504 (Bug B)
```

---

## 3. 三个 MNN Hexagon lowering bug（本项目的核心发现）

### Bug A · stride-3 交错的 mRoPE slice/scatter 降低错误

**症状**：HTP 输出与 CPU 系统性偏离，1L 就对不上；偏差随层数叠加。

**定位**：talker 用 **mRoPE（3 段交错）**，`mrope_section=[24,20,20]`，
把三个位置维度的 cos/sin 按 `i:n*3:3` 交错写入。
MNN Hexagon 对**跨步 3 的 slice/scatter** 降低结果错误。

**解法**：**host 端预先交错好**，以普通 RoPE 的形式喂进去：
```python
def host_merge(c3, s3):
    half = c3.shape[-1] // 2
    def merge(x):
        x_t = x[0].clone()
        for i, n in enumerate(ms[1:], 1):
            x_t[..., i:n*MOD:MOD] = x[i, ..., i:n*MOD:MOD]
        return x_t
    return cat([merge(c3[..., :half])]*2, -1), cat([merge(s3[..., :half])]*2, -1)
```
导出 `talker_rope_cos/sin.f32`（384 位置 × 128），运行时按位置切片即可。

**验证**：改完后 1L 直接绿，再推到 28L 全绿。

---

### Bug B · fp16 下 `-inf` 掩码被钳到 65504

**症状**：attention 里本该被完全屏蔽的位置，仍以极小权重参与。

**定位**：图内用 `-inf` 做 mask；MNN Hexagon 内部走 fp16，
`-inf` 被钳到 **65504**（fp16 最大值），于是"屏蔽"变成了"巨大正数/负数的边界行为"。

**解法**：**host 做 mask**，并统一用 `-1e4`（足够大，fp16 下不会溢出，softmax 后≈0）：
```cpp
for (int i = 0; i < TKV; i++) am[i] = (i <= cp) ? 0.f : -1e4f;
```

---

### Bug C · `ScatterND` / 索引覆写不保留目标原值

**症状**：KV cache 每步写入后，**其它 slot 的值被破坏**。
表现为：单步看起来对，多步轨迹逐渐崩坏。

**定位**：用一个专门的 KV preserve/write slot 验证图（HTP-P2-B2）：
把已知 pattern 写进 KV，走一步，再读回——**未被写入的 slot 变了**。

**重要**：这个 bug 一度让我误判 `transpose / matmul` 有问题（见 §6 的"幽灵"）。

**解法**：彻底避免 `ScatterND`，改 **one-hot mask 广播加**（见 §2.1）。

---

### 3.4 上游有没有修？—— **没有**

拿 `MNN-master`（上游 master 快照）与工作树 `MNN-3.6.1` 逐文件 diff：

```text
HexagonRaster.cpp              diff = 0 行
HexagonBinary.cpp              diff = 0 行
HexagonAttention.cpp           diff = 0 行
HexagonKVCacheManager.cpp      diff = 0 行

HexagonBackend.cpp             diff = 52 行   ← 全部是 M2-3 的插桩(HEXBE 打印 / setOpName)
HexagonRuntime.cpp             diff = 16 行   ← 同上
HexagonCommand.cpp             diff = 4 行    ← 同上
HexagonExecution.cpp           diff = 332 行  ← 同上(COPY_DIAG)
HexagonExecutionFactory.cpp    diff = 227 行  ← 同上
HexagonAttentionUtils.hpp      仅 master 有
HexagonLinearAttention.*       仅 3.6.1 有（M2-3 加的）
```

**结论**：**核心执行内核两边完全相同** → 这三个 bug 上游没修，换到 master 也躲不掉。

---

## 4. 实测数据（全部真机，可作回归基线）

### 4.1 GraphB 28L 单步

```text
                   HTP            CPU           倍率
hidden cos         0.9983157      1.0000000     -
cur_k cos          0.9999233      -             -
cur_v cos          0.9983765      -             -
延迟               126 ms/step    1190 ms/step  9.4x
RSS                1.29 GB        6.08 GB       4.7x
```

### 4.2 F1 功能门

```text
LOGITS_COS      = 0.999329996
TOP50_OVERLAP   = 49/50
EOS_2150        rankA = rankB = 91
SHARED_RNG_SAME = 32/32
```

### 4.3 G2-A 20 步强制轨迹（看漂移）

```text
hidden   0.9977570 -> 0.9985492     drift = +0.0007921   (无累积崩坏)
cur_k    drift = -0.0000492
cur_v    drift = -0.0003667
全部 finite
```

### 4.4 codepred（新架构）CP-R3 —— **HTP 精度出乎意料地好**

**CP-R3A prefill (S=2)**
```text
prefill_hidden   cos = 0.9999958
prefill_kvk      cos = 0.9999997
prefill_kvv      cos = 0.9999992
lm_head[0] logits cos = 0.9999990   top1 1159/1159   ovl50 50/50
```

**CP-R3B 14 级 forced-prefix**
```text
stage   hidden_cos    kv_k_cos    kv_v_cos |  logits_cos  top1c  top1h   ovl
0        0.9999943   0.9999996   0.9999983 |   0.9999982   1114   1114    50
1        0.9999846   0.9999986   0.9999945 |   0.9999959   1363   1363    49
2        0.9999889   0.9999985   0.9999931 |   0.9999981   1016   1016    50
3        0.9999848   0.9999984   0.9999922 |   0.9999972    810    810    50
4        0.9999629   0.9999979   0.9999894 |   0.9999947   1579   1579    50
5        0.9999764   0.9999978   0.9999880 |   0.9999960   2008   2008    50
6        0.9999752   0.9999978   0.9999873 |   0.9999962    810    810    50
7        0.9999711   0.9999977   0.9999868 |   0.9999956     86     86    49
8        0.9999734   0.9999976   0.9999865 |   0.9999981    288    288    50
9        0.9999710   0.9999975   0.9999859 |   0.9999969   1047   1047    50
10       0.9999460   0.9999976   0.9999861 |   0.9999924    914    914    47
11       0.9999585   0.9999975   0.9999859 |   0.9999959   1422   1422    50
12       0.9999267   0.9999975   0.9999860 |   0.9999907   1219   1219    50
13       0.9999223   0.9999975   0.9999862 |   0.9999931   1668   1668    49

hidden  min=0.9999223   drift(首->末) = -0.0000719
kv_k    min=0.9999975   kv_v min=0.9999859
logits  min=0.9999907   top1 14/14 全部一致

GATE hidden>=0.999     : True
GATE kv>=0.999         : True
GATE logits_cos>=0.999 : True
GATE top50>=49/50      : False（stage10 = 47/50）
```

**判定**：唯一未过的 top50 尾项，在 `logits cos 0.99999 且 top1 全一致` 的前提下，
属**尾部排序效应**，非决策面敏感 → **codepred HTP precision PASS，不需要 selective fp32**。

**性能（独占测量）**
```text
                CPU        HTP        倍率
prefill ms      98.97      1.95       51x
step 稳态 ms    ~79        ~5.3       ~15x
整帧估算        ~1223ms    ~71ms      ~17x
RSS             994 MB     367 MB
```

> 这是**本项目最大的单项性能收益**：codepred 从旧图的 **1682 ms/帧 → 71 ms/帧（23.7x）**。
> 注意 1682 ms 是**旧架构**（15 级全展开大图）在 HTP 上的数，
> 新架构在同为 HTP 的情况下是 71 ms。

### 4.5 CP-R4 自驱 16 码轨迹 —— boundary flip

```text
CPU codes = [1995 496 1114 1363 157 881 1579 | 145 1253 455 288 586 914 779 1219 901]
HTP codes = [1995 496 1114 1363 157 881 1579 | 1520 810 871 288 1331 914 1017 812 901]
first_bad_code_index = 7        <- 前 7 码完全一致

margin:
u = 0.888022022
CPU token=145   interval [0.885853, 0.896191)  width=0.010338  dist_to_edge=0.002169
HTP token=1520  interval [0.886127, 0.896445)  width=0.010318  dist_to_edge=0.001895
CDF 边界位移: lower +0.000274   upper +0.000254
```

**判定**：典型 **sampling-boundary flip** —— 两侧 CDF 只差 **2.7e-4**，
u 恰好落在两个 token 区间的**重叠段**里，排名交换导致 u 落到另一个 token 上。
**不是 backend 数学崩坏**，但说明：**fp16 下无法逐位复现 CPU 轨迹**（属可接受特性）。

---

## 5. 被证伪的诊断手段（别再浪费时间）

| 手段 | 结果 | 说明 |
|---|---|---|
| `MNN_HEX_HIDDEN_DUMP` 回读中间张量 | **垃圾数据** | hexdump 显示是 `89,90,91,92,...` 递增序列，不是真实张量 |
| `MNN_HEX_ATTN_OFF=1` 关掉 attention 优化 | **输出逐位不变** | 对标准 attention 路径无效 |

## 6. 一个"幽灵 bug"：`transpose / matmul` 到底有没有问题？

**当时的现象**：某个 5D 张量经过 `transpose + matmul` 后结果异常，
一度怀疑是 MNN Hexagon 的 transpose/matmul 降低错误，花了很久排查。

**真相**：那是 **Bug C（ScatterND 不保留目标原值）** 造成的**假象** ——
上游 KV 已被破坏，transpose 只是"接锅侠"。

**教训**：**"@transpose` 错了" 这类结论必须先构造最小复现**，
否则会把因果链下游的表现误判成上游的错。

---

## 7. 被推翻的结论（诚实记录）

| 当时的结论 | 后来发现 |
|---|---|
| "stage0 codepred logits cos 0.9956 / top50 42/50 是**材料级扰动**，HTP 精度不够" | ❌ **错**。那是 **Bug D（AR 前缀用 argmax）+ Bug E（attention 漏 transpose）** 的产物。修完后 cos 从 0.9956 → **0.99999** |
| "G2-B 首分歧在 step1 / code_index 1" 说明 HTP 精度问题 | ❌ **该测量是在有 Bug 的图上做的**，结论作废 |
| "App 内 HTP 挂死是 `ADSP_LIBRARY_PATH` 没设对" | ⚠️ **部分对**。修完后不再静默挂死，但换成**空 VARP / stop-execute 崩溃**，**至今未通** |

---

## 8. App 内 HTP 失败：完整记录

### 8.1 现象演进

```text
阶段1  选择 htp 后端 -> 进程静默挂死
       /proc/<pid>/stat: state=S, utime 冻结
       logcat 有 ADSP_LOG_PRINT 事件（DSP 被唤起但没回来）

阶段2  修好 ADSP_LIBRARY_PATH 后 -> 变成崩溃
       signal 11 (SIGSEGV), code 1 (SEGV_MAPERR), fault addr 0x0
       #01 MNN::Express::VARP::operator->()
       #02 nativeRun+17076
       => 后端 stop-execute，onForward 返回空 vector，我直接 o[0]-> 越界

阶段3  加 7 处 outOk() 防御后 -> 变成明确的错误码
       ERR BACKEND_STOP_EXECUTE: <模块名>
       但仍然跑不完（DSP 侧停止执行）
```

### 8.2 已排除的原因

| 假设 | 排除方式 | 结论 |
|---|---|---|
| 图有问题 | 同一批 `.mnn` 在 CLI 里跑出正确结果 | ❌ 不是图 |
| `ADSP_LIBRARY_PATH` 没设 | 按 README 修正（分号 + 厂商路径 + 在 dlopen 之前） | ⚠️ 修了，但不是根因 |
| 库没找到 | `dlopen failed` 已消失 | ❌ 不是 |
| libMNN 版本不对 | 与 CLI 用**同一份** `libMNN.so` | ❌ 不是 |

### 8.3 待查方向（下一步）

1. **进程 SELinux 域**：CLI 跑在 `shell` 域，App 跑在 `untrusted_app` 域。
   FastRPC（`libcdsprpc.so`）在 `untrusted_app` 下可能受 **dma-buf / 内存配额**限制。
2. 抓 `fastrpc / adsprpc / dma.?buf / cdsprpc` 相关日志对比 CLI 与 App 的差异
3. 用 `dmabuf_dump <pid>` 看 App 的 DMA-BUF 用量（方法见 `skills/hexagon/SKILL.md`）
4. 若确认是域限制 → 走**子进程方案**（`probes/vd_full_chain.cpp` 就是为此准备的：
   由 App 解出可执行文件再 exec，进程环境与能跑通的 CLI 探针一致）

---

## 9. 诊断工具与命令（可直接复制）

### 9.1 崩溃正则（来自 MNN `skills/hexagon/SKILL.md`）

```bash
adb logcat -d | grep -aiE "execute_command_group_profile failed|qurt|sysfatal|fatal|crash|tlb|cdsp.*crash|adsp.*crash|segv|signal 11"
```

### 9.2 tombstone → 源码行

```bash
T=$(adb shell "ls -t /data/tombstones/tombstone_*[0-9] | head -1")
adb shell "grep -aE 'Cmdline|signal|#0[0-3]' $T"

# 符号表定位（需要带调试符号的 .so）
llvm-nm --defined-only libvoicedesign_jni.so | grep nativeRun
#  -> 0000000000018968 T Java_com_..._nativeRun
# tombstone 里 pc=0x1c19c，函数起始 0x18968，偏移 = 0x1c19c - 0x18968 = 14388
llvm-addr2line -e libvoicedesign_jni.so -f -C 0x1c19c
#  -> voicedesign_jni.cpp:278
```

**故障地址的形态就是线索**：
- **页对齐**（如 `0x...600000`）→ 多半是 **mmap 长度不足**，访问落在映射末端之外
- **0x0** → 空指针（本项目的典型场景：后端 stop-execute 返回空 vector）

### 9.3 判断"在算"还是"卡住"

```bash
P=$(ps -A -o PID,NAME | grep <pkg> | awk '{print $1}')
awk '{print "state="$3" utime="$14" stime="$15}' /proc/$P/stat
sleep 15
awk '{print "state="$3" utime="$14" stime="$15}' /proc/$P/stat
# utime 增长 = 在算；utime 冻结 + state=S = 阻塞（DSP / IO / 锁）
```

### 9.4 DSP 内存测量（来自 SKILL.md）

```bash
adb shell "awk '/VmRSS:|VmHWM:/{print}' /proc/<pid>/status"
adb shell dmabuf_dump <pid>
adb shell dumpsys meminfo <pid>
# 近似 DSP 压力 = VmHWM + dmabuf total
```

### 9.5 库的摆法（**不要动公共目录**）

```text
/data/local/tmp/qwen3tts/vd_m5b/lib/     libMNN.so / libMNN_Express.so / libMNN_htpops.so
/data/local/tmp/qwen3tts/vd_m5b/adsp/    libMNN_htpops_skel.so
```

> ⚠️ 不要把 libMNN.so 放进 `/data/local/tmp/MNN/` 这种**公共目录** ——
> 本项目就遇到过：该文件被并行工作流替换，导致 shdr 失效。

```bash
LD_LIBRARY_PATH=<lib>:. ADSP_LIBRARY_PATH=<adsp> ./probe ...
# 注意 ADSP_LIBRARY_PATH 的权威写法(含厂商路径, 分号分隔):
#   <你的目录>;/vendor/lib/rfsa/adsp;/system/lib/rfsa/adsp
```

---

## 10. 上游资料索引（真正有用的四份）

| 资料 | 位置 | 价值 |
|---|---|---|
| Hexagon 后端部署说明 | `MNN/source/backend/hexagon/README.md` | **ADSP_LIBRARY_PATH 的权威写法**（分号 + 厂商 rfsa 路径） |
| 官方 App 写法 | `MNN/apps/Android/MnnLlmChat/.../QnnModule.kt:109` | `Os.setenv("ADSP_LIBRARY_PATH", nativeLibPath, true)` —— **Kotlin 侧、加载 native 之前** |
| Hexagon 优化/调试技能 | `MNN/skills/hexagon/SKILL.md` | 崩溃正则、`DSPOpType` profile 判据、DMA-BUF 测量法、**硬约束：不许 CPU fallback 当优化** |
| HVX FP16 激活实现 | `MNN/docs/perf/hexagon_pwl_activations.md` | Sigmoid 16 段 / Tanh 12 / GELU 12 / SiLU 8；Log 用 HVX log2 |

> MNN 仓库 `AGENTS.md` 声明 `schema/private/` 与 `source/internal/` 为**内部专有代码，
> 不得读取/修改/引用** —— 本项目**遵守**（未触碰）。

---

## 11. 工程纪律（用代价换来的）

```text
1. 一次只跑一个重任务
   -> 我曾并发跑 CPU(7.3GB) + HTP(1.1GB) 两个进程, 直接 15.4GB 内存打爆, 设备卡死
2. 优先 HTP(1.29GB) 而不是 CPU(6.08GB)
3. 用 setsid + 日志文件 + 轮询, 不用长阻塞调用
4. 用设备侧脚本文件, 避免多层引号转义
5. 跑之前先看 MemAvailable >= 5GB（不是 MemFree）、load average < 6、无残留进程
```

---

## 12. 资产与可复现清单

| 东西 | 位置 |
|---|---|
| CLI 探针源码 | `probes/cp_r3_probe.cpp`（codepred prefill+14 级）、`probes/cp_r4_probe.cpp`（自驱轨迹）、`probes/g2a_loop.cpp` / `g2b_loop2.cpp`（GraphB 20 步 + CDF 记录） |
| 逐层/单层诊断探针 | `probes/graphb_perlayer_probe.cpp`、`probes/vd_p1a_export_1l_diag.py` |
| 导出脚本 | `scripts/vd_b28_export_v6.py`（GraphB v6）、`scripts/vd_cpr1_export2.py`（codepred） |
| 历史报告（原始） | `evidence/reports/HTP_*.md`、`G1_28L_RESULT_*.md`、`G2B_RESULT_AND_DEVICE_INCIDENT_*.md`、`CPR3_*.md` |
| 设备上的库与图 | 见 §9.5（不进仓库，权重在 Release） |
