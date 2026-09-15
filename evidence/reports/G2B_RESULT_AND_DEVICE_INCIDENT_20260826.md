# G2-B 结果（部分）+ 设备稳定性事故（2026-08-26）

## 一、HTP 侧 G2-B 20 步：跑通

    GEN steps=20 stop=MAXN
    MEM VmHWM 1952352 kB / VmRSS 1285896 kB      (1.29GB，安全)
    输出 81 个文件（20 步 x hidden/logits/c0e/emb + records.txt）

### 干净测量下的 HTP 每步延迟（重要修正）

| 模块 | 干净测量 | 之前被污染的值 | 说明 |
|---|---|---|---|
| GraphB 28L | ~126 ms | 139-142 | 一致 |
| codec_head | 4.2 ms | 118 ms | 之前"1.5x 慢"的结论错误 |
| codepred_ar | 1682 ms | 7474 ms | 新的真实瓶颈 |
| frame_emb | 5.3 ms | 908 ms | 之前"395x 慢"的结论错误 |

根因：先前那组数字是"CPU 与 HTP 两个 g2b_loop2 并行"时测的，互相争抢导致严重失真。
=> 凡是以往在并发条件下测得的 NPU 小图性能结论，一律作废，需在独占条件下重测。

## 二、16-code 轨迹对比（HTP vs device-CPU smoke）

| step | CPU code0 / codes15 | HTP code0 / codes15 | 判定 |
|---|---|---|---|
| 0 | 1995 / [496 933 22 ...] | 1995 / [496 933 22 ...] | 完全一致 |
| 1 | 2042 / [885 534 1645 ...] | 2042 / [2000 88 1876 ...] | code0 一致；codes15 分叉 |

=> first divergence = step 1，位于 codepred_ar 的 15 个采样之中（code0 路径仍一致）。

机制（与已确立证据一致，非新的 lowering 缺陷）：
- G2-A 已证 GraphB hidden 在 HTP 上 cos 约 0.9983（fp16 累积，无漂移）
- 每步共 16 次采样（1 次 code0 + 15 次 codepred），命中 CDF 边界的概率比单看 code0 高约 16 倍
- F1 显示：单次 code0 采样在 32 组固定随机数下 32/32 一致
- 到 step 1 时，15 个样本中有 1 个翻转

## 三、尚未完成：margin 判定（被设备故障阻塞）

用户判据：若极少数 step 因概率贴着 CDF 边界而分叉，先看 margin，不要立即宣布 HTP precision FAIL。

需要的证据：分叉点两侧的 codepred logits + 该采样的 CDF 边界距离。
已改好探针（记录全部 16 个采样的 CDF lo/hi + dump 每步 lg15），但运行反复失败：

    patch1 cdf-all    replaced=1
    patch2 lg15 dump  replaced=1

## 四、设备稳定性事故

### 已发生
1. 我并行启动 CPU+HTP 两个 7.3GB 进程 -> 设备内存耗尽卡死（严重操作失误，已致歉并定规则）。
2. 设备自行恢复（未重启，uptime 21:17），但状态很差：
       MemTotal 15471648 kB, MemFree 459028 kB, Cached 7660292 kB
       load average: 16.31, 18.52, 26.32   (8 核)
3. 之后每次跑 3.5GB 模型加载，adbd 被 OOM-kill，无线 adb 每几分钟断一次；
   setsid 后台进程在断连时被杀（日志空，死在加载阶段）。

### 结论
- 设备当前不适合再起重进程；需要重启清页缓存与 App 常驻，或静置沉降。
- 无线调试链路本身也不稳（本轮多次端口漂移 / OOM 掉线）。

## 五、下一步（恢复后按序）

1. 设备重启或静置至 MemFree 大于 3GB 且 load 小于 6。
2. 独占跑 g2b_loop2 ... htp 3 62（已带 CDF 全量 + lg15 dump），拿到分叉点 margin。
3. 独占跑 CPU 侧短程（N=3，7.3GB）——只在内存确有余量时；否则用 PC torch 复现替代。
4. 依据 margin 判定：
   - margin 极小（采样贴着边界）-> 判为采样敏感性，不是 backend 数学崩坏，可继续 G3。
   - margin 很大 -> 转 selective fp32（优先 codec_head/codepred 路径），不动 GraphB 整体精度。

## 六、产物
- artifacts/device-smoke/vd_p2/g2/g2bhtp2/records.txt （HTP 20 步完整记录）
- android/g2b_loop2.cpp （已加 CDF 全量 + lg15 dump）
- scripts/export/patch_g2b.py
