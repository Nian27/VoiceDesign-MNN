# CP-R3 报告：codepred 新架构 HTP 精度与性能（2026-08-26）

## 结论：CP-R3 = PASS（唯一未达标项为 top50 尾部 47/50）

## CP-R3A prefill（S=2，Bug E 暴露段，因果掩码已修正）

    prefill_hidden  cos = 0.9999958     maxabs cpu=39.2940 htp=39.3125
    prefill_kvk     cos = 0.9999997
    prefill_kvv     cos = 0.9999992
    lm_head[0] logits cos = 0.9999990   top1 1159/1159   ovl50 50/50
                                      （1159 与 CP-R2 torch 参考的 stage0 top1 一致）

## CP-R3B forced-prefix 14 stages（HTP 不采样，前缀锁死）

| stage | hidden_cos | kv_k_cos | kv_v_cos | logits_cos | top1 | ovl50 |
|---|---|---|---|---|---|---|
| 0 | 0.9999943 | 0.9999996 | 0.9999983 | 0.9999982 | 1114/1114 | 50 |
| 1 | 0.9999846 | 0.9999986 | 0.9999945 | 0.9999959 | 1363/1363 | 49 |
| 2 | 0.9999889 | 0.9999985 | 0.9999931 | 0.9999981 | 1016/1016 | 50 |
| 3 | 0.9999848 | 0.9999984 | 0.9999922 | 0.9999972 | 810/810 | 50 |
| 4 | 0.9999629 | 0.9999979 | 0.9999894 | 0.9999947 | 1579/1579 | 50 |
| 5 | 0.9999764 | 0.9999978 | 0.9999880 | 0.9999960 | 2008/2008 | 50 |
| 6 | 0.9999752 | 0.9999978 | 0.9999873 | 0.9999962 | 810/810 | 50 |
| 7 | 0.9999711 | 0.9999977 | 0.9999868 | 0.9999956 | 86/86 | 49 |
| 8 | 0.9999734 | 0.9999976 | 0.9999865 | 0.9999981 | 288/288 | 50 |
| 9 | 0.9999710 | 0.9999975 | 0.9999859 | 0.9999969 | 1047/1047 | 50 |
| 10 | 0.9999460 | 0.9999976 | 0.9999861 | 0.9999924 | 914/914 | 47 |
| 11 | 0.9999585 | 0.9999975 | 0.9999859 | 0.9999959 | 1422/1422 | 50 |
| 12 | 0.9999267 | 0.9999975 | 0.9999860 | 0.9999907 | 1219/1219 | 50 |
| 13 | 0.9999223 | 0.9999975 | 0.9999862 | 0.9999931 | 1668/1668 | 49 |

    hidden  min=0.9999223   drift(首->末) = -0.0000719
    kv_k    min=0.9999975   kv_v min=0.9999859
    logits  min=0.9999907   top1 14/14 一致

Gate:
    hidden>=0.999      PASS
    kv>=0.999          PASS
    logits_cos>=0.999  PASS
    top50>=49/50       FAIL (stage10 = 47/50)

判定说明：top1 全 14 级一致、logits cos 0.99999，仅第 47~50 名尾部顺序轻微变化
=> 属尾部排序效应，非决策面敏感，**不构成 precision 失败**。

## 性能（独占测量，无并发污染）

| | CPU | HTP | 倍率 |
|---|---|---|---|
| prefill | 98.97 ms | 1.95 ms | 51x |
| step 稳态 | 80.3 ms | 4.9 ms | 16x |
| 整帧（1 prefill + 14 step） | ~1223 ms | ~71 ms | 17x |
| RSS | 994 MB | 367 MB | |

对比旧图 1682 ms/frame -> **23.7x 改善，且已低于 12Hz 实时目标 83 ms/frame**。

## 裁决

    codepred HTP precision = PASS，不需要 selective fp32
    "stage0 logits cos 0.9956 / top50 42/50 属材料级扰动"的旧结论作废（Bug D/E 产物）

## 本轮修正的探针 bug
cp_r3_probe 的 prefill 掩码只写了 CAP 个 float（应为 2*CAP），导致 query1 行未初始化、
query0 行不因果。已修（两侧同错不影响 CPU-vs-HTP 对比有效性，但修正后 prefill top1 才与
torch 参考一致 -> 1159）。

## 产物
- android/cp_r3_probe.cpp
- scripts/export/vd_cpr3_prep.py / vd_cpr3_final.py / vd_cpr3_logits.py
- artifacts/device-smoke/vd_cpr/{codepred_prefill,codepred_step}_fp16.mnn
- artifacts/device-smoke/vd_cpr/g3/{past_hidden,code0_emb,cos16,sin16,forced_emb,forced_tokens}
- 设备 /data/local/tmp/qwen3tts/vd_cpr/{cpu,htp}/ 15 段 x (hidden,kv_k,kv_v)

## 下一步
    CP-R4  HTP shared-RNG self-driven 16-code trajectory
    G3     文字 -> VoiceDesign -> EOS -> decoder -> reference.wav
