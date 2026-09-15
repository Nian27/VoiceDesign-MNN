# MEMORY.md — 关键结论速查

> 只放**已冻结的事实**（有逐位证据）。推测写在 docs/ 里并标注。

## 模型事实

```text
模型          Qwen3-TTS-12Hz-1.7B-VoiceDesign
tts_model_type = voice_design        (不需要参考音频)
spk_id         = {}                  (无预设说话人)
talker         28 层, hidden 2048, 16 头 / 8 KV 头, head_dim 128
              mrope_section [24,20,20], interleaved=True
code_predictor 5 层, hidden 1024, 8 KV 头, head_dim 128, vocab 2048
帧率           12.5 Hz, 16 码本/帧
采样率         24 kHz
EOS           2150 (codec_eos_token_id)
特殊 id       pad 2148 / bos 2149 / think 2154 / nothink 2155 / think_bos 2156 / think_eos 2157
              language: chinese 2055, english 2050, ...
tts specials  151671 <tts_pad>  151672 <tts_text_bos>  151673 <tts_text_eod>   (不带竖线!)
```

## 采样参数（官方默认）

```text
do_sample True / temperature 0.9 / top_k 50 / top_p 1.0 / repetition_penalty 1.05
suppress [2048,3072) 但排除 2150
sub-talker: temperature 0.9 / top_k 50 / top_p 1.0（无 repetition penalty）
```

## 验证过的数字（可作回归基线）

```text
分词器      设备 C++ vs HF                 10/10 逐 id 相同
Prompt      PC 复刻 vs 官方捕获            exact_equal = True (diff 0.0)
Prompt      设备构造 vs PC 参考            cos = 1.0000000  maxabsdiff 1.997e-06
GraphB-HTP  28L 单步                       hidden 0.9983157  k 0.9999233  v 0.9983765
GraphB      延迟                           HTP 126 ms   CPU 1190 ms
codepred    prefill                        hidden 0.9999958  kv 0.9999997
codepred    14 级                          hidden min 0.9999223  logits min 0.9999907
                                           top1 14/14 一致
codepred    性能                           prefill 1.95ms / step 4.9ms / 整帧 71ms
端到端      OpenCL                         8.72s WAV, finite, stop=EOS
```

## 性能基线（Honor Magic8 Pro）

```text
                 OpenCL(App)      HTP(CLI)
load             1.6-6.1 s        -
prefill          71 s / 62 tok    31 s / 62 tok
decode           2.5 s/帧         0.126 s/帧 (GraphB) + 0.071 s/帧 (codepred)
decoder          57 s             -
合计             约 7 分钟/条      (若 HTP 可用: 估 ~1 分钟/条)
```

## 部署铁律

```text
1. App 读 filesDir = /data/user/0/<pkg>/files/ ，不是 /sdcard/Android/data/...
2. 部署必须 run-as 管道:  cat src | adb shell "run-as <pkg> sh -c 'cat > dst'"
3. 安装: pm install --user 0 -r  （设备有多用户/平行空间）
4. 装完必须核验: dumpsys package <pkg> | grep lastUpdateTime
5. 一次只跑一个重任务
```

## 还没搞清楚的

```text
- HTP 在 App 内为什么不可用（CLI 正常）。已排除 ADSP_LIBRARY_PATH。
  待查方向: untrusted_app 域的 FastRPC / dma-buf 限制
- OpenCL 上 2.5 s/帧 的具体瓶颈（怀疑是 2.8GB 权重的每步带宽）
- buildPrompt 对超长文本没有越界保护（maxTok=512）
```
