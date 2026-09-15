# 架构说明

## 1. 三段式：文字 → 离散码 → 波形

```text
[文本域]                      [离散码域]                        [波形域]
文字 --BPE--> token ids       16 码本 x N 帧 @ 12.5 Hz         24 kHz PCM
     --text_embedding-->     每帧: code0 (talker) + code1..15   1920 样点/帧
     --projection-->           (code_predictor)
     prompt_emb (L x 2048)
```

- **talker**：28 层，hidden 2048，16 头 / 8 KV 头，head_dim 128；输出 code0
- **code_predictor**：5 层，hidden 1024，由 code0 推出 code1..15
- **decoder**：16 码本 → 波形（本项目用静态 300 帧版本）

## 2. 图划分（为什么是这 7 张图）

| 图 | 输入 | 输出 | 用途 |
|---|---|---|---|
| `graphb28_v6_fp16` | inputs_embeds, kv_k, kv_v, rope_cos, rope_sin, attn_mask, slot_mask | hidden_states, cur_k, cur_v | talker 单步（**prefill 也用它**） |
| `codec_head_fp16` | hidden_states | logits(3072) | 采样 code0 |
| `codec_emb_fp16` | codes | emb | code0 → 嵌入 |
| `codepred_prefill_fp16` | past_hidden, code0_emb, rope, slot_mask0/1, attn_mask | hidden_last(1024), kv_k/v(5,1,8,16,128) | code_predictor prefill |
| `codepred_step_fp16` | token_emb, kv_k/v, rope, slot_mask, attn_mask | hidden_last, kv_k/v | code_predictor 单步 |
| `frame_emb_fp16` | codes(16) | emb(2048) | 16 码 → 帧嵌入 |
| `tokenizer_decoder_static_t300` | codes(1,16,300) | waveform | 码 → 波形 |

**留在 host 的**：15 个 `lm_head`、15 个 codec embedding 表、分词器、prompt 拼装、采样。
理由：这几步要么是纯查表/矩阵乘（host fp32 无精度损失），要么需要严格复刻官方采样语义。

## 3. KV cache 设计

### talker（GraphB）
- 容量 **768 slot**（位置 0..767）
- 写入方式：**one-hot slot_mask 广播加**，不是 `ScatterND`
  （因为 MNN Hexagon 的 `ScatterND` 不保留目标原值 —— Bug C）
- `attn_mask` 用 `0 / -1e4`（**不用 `-inf`**，fp16 下 `-inf` 会被钳到 65504 —— Bug B）

### code_predictor
- 容量只要 **16**（prefill 2 + 14 步）
- 原方案展开成 15 级大图 → O(n^2) attention + 每次重算全前缀；改成 prefill+step 后 attention 是 O(n)

## 4. RoPE

talker 用 **mRoPE（3 段交错）**，`mrope_section=[24,20,20]`。
MNN Hexagon 对 **stride-3 交错** 的 slice/scatter 降低错误（**Bug A**），
所以 **host 端预先交错好**再以普通 RoPE 的形式喂进去：

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

## 5. 采样（与官方对齐）

```text
do_sample        = True
temperature      = 0.9
top_k            = 50
top_p            = 1.0
repetition_penalty = 1.05     <- 只作用于 talker 的 code0
eos              = 2150 (codec_eos_token_id)
suppress         = [2048, 3072) 但排除 2150
sub-talker       = temperature 0.9 / top_k 50 / top_p 1.0（无 repetition penalty）
```

## 6. prompt 拼装的 token 布局

```text
偏移   内容                        来源
-----  --------------------------  ----------------------------------
0..32  instruct                    "<|im_start|>user\n{指令}<|im_end|>\n"
33..35 role                        "<|im_start|>assistant\n"
36..40 main                        cat([pad]*4, bos) + codec[:5]
41..60 body                        te(asst[3:-5]) + tts_eos, 再 + ce(pad)
61     tail                        pad + ce(bos)

codec = ce([think, think_bos, lang, think_eos, pad, bos])
       = ce([2154, 2156, 2055(中文), 2157, 2148, 2149])
```

> 这里的 L=62 是默认那组文字。换文字后 token 数与 body 长度都会变（实测换一段文字 → 60）。

## 7. 线程与生命周期

`text
Activity  ──start──▶  VoiceDesignService（前台服务 + 通知）
                          │  在独立 Thread 上跑整链
                          │  日志写 filesDir/voicedesign/vd_run.log
                          ▼
                      VoiceDesignEngine（native session）
                          │  prepareAdsp() 必须在构造之前
                          ▼
                      nativeRun() —— 持 mutex，一次一条

完成后：写 WAV → MediaPlayer 自动播放 → stopForeground → stopSelf
```

**为什么必须前台服务**：整链 7 分钟 + 4.7GB 工作集，普通后台进程会被厂商内存管理杀掉（见 PITFALLS P9）。
