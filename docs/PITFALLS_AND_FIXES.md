# 踩坑与修复全集

按"症状 → 定位 → 根因 → 修法 → 判据"记录。**每一条都是真机实测踩出来的。**

---

## P1 · 真凶：`text_projection` 是 `bias=True` 的 MLP，漏了 bias

**症状**
设备端在线构造的 prompt 与 PC 参考**结构完全一致（都是 62 token）但内容全错**；
生成时 code0 反复重复、**永不吐 EOS**（跑到 maxFrames=300 才停）。

**定位**
把设备构造的 prompt dump 成文件，拉回 PC 逐元素比对：
```text
device shape (62,2048) == pc ref (62,2048)     <- 长度对
GLOBAL cos = 0.7824020                          <- 内容错
per-token cos: min=0.513669 max=0.864812
maxabsdiff = 7.1979e-01
```

于是把 projection 单独拿出来在 PC 上复现（用导出的 fp16 表）：
```text
PROJ cos = 0.8335207   maxabsdiff=0.71712
ref mean|.|=0.0405  mine mean|.|=0.0405      <- 幅度完全一致 -> 不是精度问题
```

再打印层的结构：
```text
(linear_fc1): Linear(in_features=2048, out_features=2048, bias=True)   <-- bias
(linear_fc2): Linear(in_features=2048, out_features=2048, bias=True)   <-- bias
fc1 stage:  x@f1.T vs torch  cos=0.9876823   <- 就是缺 bias 造成的
```

**根因**
`Qwen3TTSTalkerResizeMLP` 的构造函数默认 `bias=False`，**但实例化时传了 `bias=True`**。
我照着类定义写实现，没查实例，于是 fc1/fc2 都没加 bias。

**修法**
导出 bias 并在 matmul 后相加：
```python
b1 = tp.linear_fc1.bias.detach().float().numpy().astype(np.float32)
b2 = tp.linear_fc2.bias.detach().float().numpy().astype(np.float32)
b1.tofile('text_proj_fc1_bias.f32'); b2.tofile('text_proj_fc2_bias.f32')
```
```cpp
float acc = b1[o];                     // fc1
for (int i = 0; i < 2048; i++) acc += h2f(w[i]) * in[i];
mid[o] = acc / (1.0f + expf(-acc));    // SiLU

float acc2 = b2[o];                    // fc2
for (int i = 0; i < 2048; i++) acc2 += h2f(w[i]) * mid[i];
out[o] = acc2;
```

**判据**
```text
加 bias 后：PROJ cos = 1.0000000   maxabsdiff = 0.000000
端到端：DEVICE PROMPT vs PC REF  cos = 1.0000000  maxabsdiff = 1.997e-06
```

**教训**
> 对着 **类定义** 写实现会漏掉 **实例参数**。凡是有默认参数的层，必须 `print(module)` 看实例的真实结构。

---

## P2 · 真凶：`mmap` 长度把 float 个数当 bytes

**症状**
`nativeRun` 里 `SIGSEGV`，崩溃在读写 lm_head 表的循环里。
```text
signal 11 (SIGSEGV), code 1 (SEGV_MAPERR), fault addr 0x0000007237600000 (read)
```

**定位**
`addr2line` 精确定位到源码行：
```text
llvm-nm  ->  Java_com_voicedesign_app_VoiceDesignEngine_nativeRun = 0x18968
tombstone pc = 0x1c19c  ->  +14388
llvm-addr2line -e libvoicedesign_jni.so 0x1c19c
  -> voicedesign_jni.cpp:278
```

第 278 行是 `&s->lmW[(g+1)*2048*1024 + o2*1024]`。

**根因**
```cpp
s->lmW = (const float*)mmapFile(d+"lm_head_weight.f32",
                                (size_t)15*2048*1024,      // 这是 float 个数
                                &s->lmMap);                // 但 mmap 的 length 是字节数
```
只映射了应需的 1/4（31MB 而不是 126MB）。

**判据（关键特征）**
> **故障地址是页对齐的**（`0x...600000`）→ 说明访问落在了**映射区末端之外**，而不是随机的野指针。
> 这是"映射长度不足"的典型指纹。

**修法**
```cpp
mmapFile(d+"lm_head_weight.f32", (size_t)15*2048*1024*sizeof(float), &s->lmMap)
mmapFile(d+"codec_emb_weight.f32", (size_t)15*2048*2048*sizeof(float), &s->cpMap)
```

---

## P3 · 622MB 文本表整块读入 → load 阶段挂死

**症状**
加了文本表之后，App 卡在 `[1/3] load ...`，不出日志、不崩溃。
```text
state=S  utime=472 (冻结)   VmHWM=2081228 kB   VmRSS=560576 kB
```

**根因**
`std::vector<uint16_t> emb(151936*2048)` → 一次性 malloc + read 622MB，
叠加其它表后 RSS 峰值 2GB+，触发厂商内存管理/交换，进程进入不可调度状态。

**修法**
大表一律 mmap（页缓存可回收，不占堆）：
```cpp
static const void* mmapRO(const std::string& p, size_t bytes) {
    int fd = open(p.c_str(), O_RDONLY);
    if (fd < 0) return nullptr;
    void* q = mmap(nullptr, bytes, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    return (q == MAP_FAILED) ? nullptr : q;
}
```

**判据**
改完后 `[2/3] load OK`，load 稳定在 1.6-6.1 s。

---

## P4 · `ADSP_LIBRARY_PATH` 设置方式错误 → App 内 DSP 挂死

**症状**
App 内选 HTP 后端时，进程**卡死**（utime 冻结、state=S），logcat 里能看到 `ADSP_LOG_PRINT` 事件（DSP 被唤起但没回来）。
**同一台设备上 CLI 探针却能正常跑完。**

**定位**
读 MNN 仓库 `source/backend/hexagon/README.md`：
```bash
export ADSP_LIBRARY_PATH="/data/local/tmp/hexagon_libs;/vendor/lib/rfsa/adsp;/system/lib/rfsa/adsp"
export LD_LIBRARY_PATH="/data/local/tmp/hexagon_libs:$LD_LIBRARY_PATH"
```
再看官方 App `apps/Android/MnnLlmChat/.../QnnModule.kt:109`：
```kotlin
Os.setenv("ADSP_LIBRARY_PATH", nativeLibPath, true)
```

**根因（两处都错）**
1. 我把它**只设成 App 自己的 lib 目录** → 把厂商的 `/vendor/lib/rfsa/adsp` 覆盖掉了 → DSP 加载不到基础库
2. 我是在 **JNI 的 nativeLoad 里 `setenv`** → 那时 MNN 库已经 dlopen 完了，太晚

**修法**
```kotlin
companion object {
    @Volatile private var adspReady = false
    fun prepareAdsp(nativeLibDir: String) {
        if (adspReady) return
        val dirs = listOf(nativeLibDir, "/vendor/lib/rfsa/adsp", "/system/lib/rfsa/adsp")
            .filter { it.isNotBlank() }.joinToString(";")
        try { android.system.Os.setenv("ADSP_LIBRARY_PATH", dirs, true) } catch (_: Throwable) {}
        adspReady = true
    }
}
// 调用点必须在 new VoiceDesignEngine() 之前（构造函数里会 System.loadLibrary）
VoiceDesignEngine.prepareAdsp(applicationInfo.nativeLibraryDir)
val eng = VoiceDesignEngine()
```

**效果**
从"静默挂死"变成"可诊断的崩溃/报错"。**注意：这一修并没有让 HTP 在 App 内跑通**，
后续崩溃是空 VARP（见 P5）。HTP 在 App 内至今不可用，详见 docs/NPU_HEXAGON_NOTES.md。

---

## P5 · 后端 stop-execute 时 `onForward` 返回空 vector → 空指针崩溃

**症状**
HTP 跑到一半崩：
```text
signal 11 (SIGSEGV), code 1 (SEGV_MAPERR), fault addr 0x0000000000000000 (read)
#01 MNN::Express::VARP::operator->()
#02 nativeRun+17076
```

**根因**
MNN 的 `Module.cpp` 里有这么一段：
```cpp
if (hasNoExecution) {
    MNN_PRINT("Warning, Backend has stop execute, return empty output vector varps\n");
    // 返回空 vector
}
```
我的代码直接 `o[0]->readMap<float>()` → 对空 VARP 取箭头 → 崩。

**修法**
加统一防御，7 个 `onForward` 调用点全部包住：
```cpp
static bool outOk(const std::vector<VARP>& o, size_t need, const char* tag) {
    if (o.size() < need) { LOGI("BACKEND_STOP_EXECUTE at %s (out size=%zu need=%zu)", tag, o.size(), need); return false; }
    for (size_t i = 0; i < need; i++) { if (o[i].get() == nullptr) { LOGI("NULL_VARP at %s[%zu]", tag, i); return false; } }
    return true;
}
// 调用点
std::vector<VARP> o = s->mGB->onForward({...});
if (!outOk(o, 3, "graphb")) return std::vector<float>();
```

**注意** `VARP` 没有 `operator!`，要用 `o[i].get() == nullptr`。

**效果**
崩溃变成明确的 `ERR BACKEND_STOP_EXECUTE: <模块名>`，一眼看出是哪个子图挂了。

---

## P6 · tts special token 名字想当然

**症状**
`ERR prompt build failed`（`buildPrompt` 返回负数）。

**根因**
我写的是 `<|tts_bos|>` / `<|tts_eos|>` / `<|tts_pad|>`，
**真名不带竖线**：
```text
151671 = '<tts_pad>'
151672 = '<tts_text_bos>'
151673 = '<tts_text_eod>'
151674 = '<tts_text_bos_single>'
```

**修法**
`cpp
const int BOS = tk.idOf("<tts_text_bos>"), EOS = tk.idOf("<tts_text_eod>"), PAD = tk.idOf("<tts_pad>");
```

**教训**
> 特殊 token 的名字要从 `tokenizer_config.json` 的 `added_tokens_decoder` **读出来**，不能凭命名习惯猜。

---

## P7 · 缺 `repetition_penalty=1.05`

**症状**
生成的 code0 反复出现同一个值（`960,960 / 183,183 / 1894,1894`），不收敛、不吐 EOS。

**根因**
官方 `generate_config` 的默认采样参数里有 `repetition_penalty=1.05`，
我只实现了 `temperature / top_k / top_p`。

**修法**
按 HF `RepetitionPenaltyLogitsProcessor` 的语义实现（**只作用于历史中出现过的 id**）：
```cpp
static int sampleTopK(const float* lg, int vocab, double temp, int topk, bool suppress, double u,
                      double repPenalty = 1.0, const std::vector<int>* hist = nullptr) {
    std::vector<float> lg2;
    const float* src = lg;
    if (repPenalty != 1.0 && hist && !hist->empty()) {
        lg2.assign(lg, lg + vocab);
        for (int id : *hist) {
            if (id < 0 || id >= vocab) continue;
            lg2[id] = (lg2[id] < 0) ? (float)(lg2[id] * repPenalty) : (float)(lg2[id] / repPenalty);
        }
        src = lg2.data();
    }
    ...
}
// 调用：只作用于 talker 的 code0（与官方一致，sub-talker 不带惩罚）
int code0 = sampleTopK(lg.data(), 3072, 0.9, 50, true, nextU(s->seed), 1.05, &code0Hist);
```

---

## P8 · App 私有目录：adb 建的目录 App 读不了

**症状**
把模型 `cp` 到 `/sdcard/Android/data/<pkg>/files/voicedesign/` 后，App 报 load failed。
```text
# App 自己的视角
run-as com.readervoice.app ls /sdcard/Android/data/.../voicedesign/
  -> Permission denied
```

**根因**
用 adb(shell) 建的目录属主是 `shell:ext_data_rw`、模式 `drwxrws---`，App 的 uid 既不是 owner 也不在这个组。

**修法**
两个都要做：
1. 代码里用 **`filesDir`**（= `/data/user/0/<pkg>/files/`），不要用 `getExternalFilesDir()`
2. 部署走 **`run-as` 管道**：
```bash
cat /data/local/tmp/vd/x.bin | adb shell "run-as $PKG sh -c 'cat > files/voicedesign/x.bin'"
```

---

## P9 · 官方 App 内存策略会杀进程（前台服务也保不住）

**症状**
整链跑到一半进程消失，无 crash log。
```text
am_proc_died: [0,25797,com.readervoice.app,700,15]        reason = lmkd
am_kill: [...,com.voicedesign.app,200,
          iAwareF[LowMem(OVERSEA_APP:SIM_CHINA:MAIN_PROC:FG_SERVICE:)](fg-service), 6023356]
```
注意括号里的 **FG_SERVICE** —— 它是在**已经作为前台服务运行时**被杀的。

**根因**
设备（Honor）的低内存策略 + 4.7GB 模型工作集。

**缓解**
- 大表 mmap（见 P3）
- 用户侧白名单：设置 → 应用启动管理 → 手动管理（允许后台活动）
- 一次只跑一个重任务（我曾并发跑 CPU+HTP 两个进程，直接把设备干到卡死）

---

## P10 · 其它小坑

| 现象 | 根因 | 修法 |
|---|---|---|
| `Session` 类型名冲突 | `using namespace MNN` 与 `MNN::Session` 撞名 | 改名 `VdSession` |
| `VARP` 没有 `operator!` | 编译错误 | 用 `o[i].get() == nullptr` |
| `snprintf` 后进程死 | 诊断代码里循环上界写成 `pe.size()`（512*2048），实际有效只有 `L*2048` | 用 `ref.size()` |
| 安装后跑的还是旧版 | Honor 弹权限框被拒 → `INSTALL_FAILED_ABORTED: User rejected permissions` | 手机上点"允许"；装完用 `dumpsys package ... lastUpdateTime` 核验 |
| 设备有 `user 100:平行空间` | shell 被切到 user 100 | `pm install --user 0 -r`、`am start --user 0` |
| logcat 看不到自己的日志 | ring buffer 只有 256KB，被 HTP 的打印刷爆 | 日志写文件；或 `adb logcat -G 16M` |
| 十六进制 dump 全是 `89,90,91,92` | 某个 debug 环境变量的回读是垃圾数据 | 不要依赖它做判据 |
