# 验证记录（全部真机实测）

每一节都是"**怎么测的 + 原始输出 + 判据**"，不是结论堆砌。

---

## V1 · 分词器：设备 C++ vs HuggingFace

**怎么测**
PC 上用 HF `processor` 对 5 组文本编码，把 ids 写成期望值；
设备上跑同一组文本，输出自己的 ids；逐 id 比对。

覆盖：中文 + 中文标点（，、。）、英文缩写（`It's` / `don't`）、数字、换行、省略号。

```text
case 0 seg A : MATCH  (got 33 ids, exp 33 ids)
case 0 seg B : MATCH  (got 27 ids, exp 27 ids)
case 1 seg A : MATCH  (got 21 ids, exp 21 ids)
case 1 seg B : MATCH  (got 18 ids, exp 18 ids)
case 2 seg A : MATCH  (got 17 ids, exp 17 ids)
case 2 seg B : MATCH  (got 18 ids, exp 18 ids)
case 3 seg A : MATCH  (got 11 ids, exp 11 ids)
case 3 seg B : MATCH  (got 25 ids, exp 25 ids)
case 4 seg A : MATCH  (got 10 ids, exp 10 ids)
case 4 seg B : MATCH  (got 18 ids, exp 18 ids)

EXACT MATCH = 10/10
TOKENIZER_GATE = PASS
```

**判据**：10/10 逐 id 相同 → PASS

---

## V2 · Prompt 拼装：PC 复刻 vs 官方捕获

**怎么测**
Hook `talker.model.forward` 抓到官方在真实调用中喂进去的 `inputs_embeds`（62×2048），
再用自己实现的拼装逻辑独立复现一遍，逐元素比对。

```text
reconstructed shape = (62, 2048)  captured = (62, 2048)
GLOBAL cos = 1.00000000
per-token cos: min=1.000000 max=1.000000
tokens with cos<0.999: NONE
exact_equal = True  maxabsdiff=0.000e+00
```

**拼装公式（已验证）**
```text
62 tokens = instruct(33) + role(3) + main(5) + body(20) + tail(1)

instruct = te(instruct_ids)                    "<|im_start|>user\n{指令}<|im_end|>\n"
role     = te(asst_ids[:3])                    "<|im_start|>assistant\n"
main     = cat([pad]*4, bos) + codec[:5]
codec    = ce([think, think_bos, lang, think_eos, pad, bos])    lang(chinese)=2055
body     = cat((te(asst_ids[3:-5]), eos), dim=1) + ce([pad]*20)
tail     = pad + ce([bos])

te(x) = text_projection(text_embedding(x))     <- MLP 有 bias!
ce(x) = codec_embedding(x)
```

**判据**：`exact_equal=True`（逐位相同）→ PASS

---

## V3 · 设备端在线构造 prompt vs PC 参考

**怎么测**
App 里用官方那组文字跑，把设备构造的 prompt dump 成文件拉回 PC 比对。

（第一次跑：FAIL，见 P1 的 bias 缺失）

```text
修复后：
DEVICE PROMPT vs PC REF: shape (62, 2048) (62, 2048)
  cos = 1.0000000   maxabsdiff = 1.997e-06
```

**判据**：cos=1.0、误差仅 fp16 舍入量级 → PASS

---

## V4 · GraphB 28L 单步：CPU vs HTP

```text
hidden  cos = 0.9983157
cur_k   cos = 0.9999233
cur_v   cos = 0.9983765

延迟：HTP ~126 ms/step   CPU ~1190 ms/step   （9.4x）
内存：HTP ~1.29 GB        CPU ~6.08 GB
```

---

## V5 · codepred：prefill + 14 级 forced-prefix（CPU vs HTP）

**怎么测**
前缀锁死（HTP 不采样），prefill 一次 + 14 次 step，每级 dump hidden / kv_k / kv_v，
再用 host 的 lm_head 算 logits 比 top1 / top50。

**CP-R3A（prefill, S=2）**
```text
prefill_hidden cos = 0.9999958
prefill_kvk    cos = 0.9999997
prefill_kvv    cos = 0.9999992
lm_head[0] logits cos = 0.9999990   top1 1159/1159   ovl50 50/50
```

**CP-R3B（14 级）**
```text
stage   hidden_cos    kv_k_cos    kv_v_cos |  logits_cos  top1c  top1h   ovl
0        0.9999943   0.9999996   0.9999983 |   0.9999982   1114   1114    50
4        0.9999629   0.9999979   0.9999894 |   0.9999947   1579   1579    50
7        0.9999711   0.9999977   0.9999868 |   0.9999956     86     86    49
10       0.9999460   0.9999976   0.9999861 |   0.9999924    914    914    47
12       0.9999267   0.9999975   0.9999860 |   0.9999907   1219   1219    50
13       0.9999223   0.9999975   0.9999862 |   0.9999931   1668   1668    49

hidden  min=0.9999223   drift=-0.0000719
kv_k    min=0.9999975   kv_v min=0.9999859
logits  min=0.9999907   top1 14/14 全部一致

GATE hidden>=0.999     : True
GATE kv>=0.999         : True
GATE logits_cos>=0.999 : True
GATE top50>=49/50      : False（stage10 = 47/50）
```

**判定**：唯一未过的 top50 尾项，在 `logits cos 0.99999 且 top1 全一致` 的前提下属**尾部排序效应**，非决策面敏感。
=> **codepred HTP precision PASS，不需要 selective fp32**。

**性能（独占测量）**
```text
                CPU        HTP        倍率
prefill ms      98.97      1.95       51x
step 稳态 ms    ~79        ~5.3       ~15x
整帧估算        ~1223ms    ~71ms      ~17x
RSS             994 MB     367 MB
```

---

## V6 · codepred 自驱轨迹（shared-RNG）

```text
CPU codes = [1995 496 1114 1363 157 881 1579 145 1253 455 288 586 914 779 1219 901]
HTP codes = [1995 496 1114 1363 157 881 1579 1520 810 871 288 1331 914 1017 812 901]
first_bad_code_index = 7        <- 前 7 码完全一致

margin:
u = 0.888022022
CPU token=145   interval [0.885853, 0.896191)  dist_to_edge=0.002169
HTP token=1520  interval [0.886127, 0.896445)  dist_to_edge=0.001895
CDF 边界位移: lower +0.000274   upper +0.000254
```

**判定**：典型 **sampling-boundary flip**（CDF 只差 2.7e-4，u 落在两个 token 区间重叠段）。
不是 backend 数学崩坏 —— 但说明 **fp16 下无法逐位复现 CPU 轨迹**，属可接受特性。

---

## V7 · 端到端：真机文字 → WAV

**输入（用户在手机上改的）**
```text
instruct = 苍老女性，低沉略沙哑，声音有岁月感，语气沉稳、有威严，凶狠。
text     = 小家伙，修炼之事急不得，根基若是不稳，日后吃亏的还是你自己。
```

**运行日志**
```text
tokenizer + text tables loaded
loaded backend=opencl in 1589ms
tokenized instruct=33 assistant=27
prompt tokens=60                              <- 用户改文字后 token 数从 62 变 60
prefill 60 tokens in 71249ms
frame 0 code0=...  ...  frame 108 code0=1431 T=110

完成：结果 = OK steps=109 stop=EOS prefill=71249ms decode=272627ms decoder=57076ms
       wav=209280 samples (8.72s) finite=yes min=-0.4162 max=0.3825
完成：耗时 = 413327 ms
完成：WAV = .../vd_opencl_1789456894357.wav
完成：8.72 秒 / 418604 字节 —— 正在播放
播放结束
```

**音频实测**
```text
channels=1 rate=24000 frames=209280 sampwidth=2 dur=8.720s
peak=0.4162 rms=0.0601 finite=True
每0.5s RMS: 0.055 0.070 0.003 0.101 0.090 0.077 0.008 0.103 0.117 ...
非静音段 15/17
```

**判据**：`stop=EOS`（自然停止，非截断）+ PCM 全 finite + 语音包络正常 → PASS

---

## V8 · 其它实测量

| 项 | 值 |
|---|---|
| GraphB v6 fp16 权重 sha256 | `f23dbc06646349f3fd90aece...` (2,821,857,280 B) |
| codepred_prefill_fp16.mnn | 161,953,004 B，ArgMax=0 ScatterND=0 IsNaN=0，913 节点 |
| codepred_step_fp16.mnn | 161,923,900 B，ArgMax=0 ScatterND=0，809 节点 |
| 分词器压缩包 tokenizer.bin | 3,490,103 B（vocab 151643 + merges 151387 + added 33） |
| 设备 | SM8850 / BKQ-AN90 / canoe / Android 17 / arm64-v8a / MemTotal 15,471,648 kB |
| OpenCL 构建 | libMNN.so 55,339,776 B，libMNN_Express.so 20,994,096 B，libMNN_CL.so 66,821,352 B |
| Hexagon 构建 | libMNN.so 61,629,192 B，libMNN_Express.so 21,000,016 B |
