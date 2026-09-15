# VD-M5B 真机整链状态（2026-08-26）

## 目标
SM8850 CPU 上跑通完整 VoiceDesign 自驱动链，产出第一条真机 reference.wav，并冻结为 DEVICE GOLDEN。

## 已完成
1. **PC 侧模型/图**
   - codepred_ar_fp16.mnn (506MB) —— 官方 AR 语义单图，MNN vs torch cos=0.9999988, codes 15/15 exact
   - frame_emb_fp16.mnn (138MB) —— codes(1,16) -> next inputs_embeds(1,1,2048)，MNN vs ONNX cos=0.99999998
   - codec_emb_fp16.mnn (12.6MB) —— code0 -> codec_emb(1,1,2048)
   - codec_head_fp16.mnn (12.6MB)
2. **prefill 状态（PC 算好，LP=62）**
   - prefill_hidden_last.raw (1,1,2048)
   - prefill_kv_k.raw / prefill_kv_v.raw (28,1,8,62,128)
   - 真实描述：text=小家伙，修炼之事急不得，根基若是不稳，日后吃亏的还是你自己。/ instruct=苍老男性，低沉略沙哑，声音有岁月感，语气沉稳、有威严，但并不凶狠。
3. **真机编排器** android/vd_m5b_chain.cpp -> artifacts/device-smoke/vd_m5b_chain
4. **全部制品已部署** /data/local/tmp/qwen3tts/vd_m5b/
   （GraphB 复用 ../vd_m5a/graphb_deploy_hand_fp16_20260826.mnn；decoder 复用 ../tokenizer_decoder_static_t300.mnn）

## 链定义
    hidden = prefill_last_hidden
    code0  = argmax(codec_head(hidden))   # suppress [2048,3072) 但保留 2150(EOS)
    loop n: cp = 62+n
      c0e    = codec_emb(code0)
      lg15   = codepred_ar(hidden, c0e)  -> codes15 = argmax 各行
      emb    = frame_emb([code0]+codes15)
      hidden, kv = GraphB(emb, pos=[cp,cp,cp], cp, kv768)
      code0  = argmax(codec_head(hidden))
      若 code0==2150 停（该帧不入列）；上限 300 帧
    decoder(codes(1,16,300) i32) -> waveform(1,1,576000,1)
    截断到 T*1920 样本 @24kHz，写 reference.wav

## 运行方式（关键）
    cd /data/local/tmp/qwen3tts/vd_m5b
    LD_LIBRARY_PATH=/data/local/tmp/MNN:. nohup ./vd_m5b_chain \
      ../vd_m5a/graphb_deploy_hand_fp16_20260826.mnn codec_head_fp16.mnn codec_emb_fp16.mnn \
      codepred_ar_fp16.mnn frame_emb_fp16.mnn ../tokenizer_decoder_static_t300.mnn \
      prefill_hidden_last.raw prefill_kv_k.raw prefill_kv_v.raw out 300 > out/chain.log 2>&1 &
  注意：真机 stdout 块缓冲，日志不实时；产物在结束时落盘；adb 断开会杀进程。

## 观测到
- 进程启动后 RSS 增长至 6.93GB（6 个模型加载完），utime 持续增长（20s 内 +1076 ticks），状态 R/S 交替 → 在推进。
- 尚未拿到 out/codes_i32.raw / waveform_f32.raw / reference.wav —— 传输中断，进程结局未知。

## 待办
1. 恢复设备连接后确认进程/产物；若被杀则重跑。
2. 拿到 reference.wav → 冻结 DEVICE GOLDEN。
3. GraphA t62 上机（其 .weight SHA=63b3b4da… 与设备已有 15c2ab4a… 不同，需推 2.8GB 或改导出使常量集一致）——目前 prefill 在 PC 侧算，GraphA 真机已单独验证 cos 0.999983。
4. HTP 支线（独立，不阻塞主线）：HTP-P0 证明真跑在 HTP → HTP-P1 device CPU vs HTP first-divergence。

## 已知硬件/环境
- 设备 SM8850 / BKQ-AN90 / Android 17；USB serial A3TE025B03003242（不稳定，频繁 offline）
- 无线调试端点频繁轮换且信任关系易失；SELinux Enforcing，shell 无 root
- GraphB 5D 输入必须走 Express Module API（Interpreter API 的 Tensor(CAFFE)+memcpy 会算错）
