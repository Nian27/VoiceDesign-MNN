# AGENTS.md — VoiceDesign-MNN

给在本仓库工作的 agent 的上下文约定。**先读本文件，再读 README 与 docs/。**

## 项目一句话

把 Qwen3-TTS-12Hz-1.7B-VoiceDesign 端侧化：**纯文字**设计音色 → 真机生成 WAV。
不需要参考音频（那是 Voice Clone，另一条线）。

## 不变量（不可破坏）

```text
1.  原文/用户输入不可变 —— 只做 tokenize，不改写
2.  host 采样语义必须与官方一致（do_sample/top_k/top_p/temperature/repetition_penalty）
3.  图里不放 ArgMax / ScatterND（前者破坏采样语义，后者在 Hexagon 上不保留目标原值）
4.  大表一律 mmap，禁止整块 read() 进 std::vector
5.  mmap 的长度参数是「字节」，不是「元素个数」
6.  ADSP_LIBRARY_PATH 必须含厂商 rfsa 路径，且在任何 MNN 库 dlopen 之前设置
7.  后端 stop-execute 时 onForward 会返回空 vector —— 解引用前必须检查
8.  模型不进 APK、不进 git；用户数据不可破坏
9.  secrets 绝不写入仓库（发现凭据只记录类别与位置，不打印值）
10. 一次只跑一个重任务（并发跑会把设备内存打爆，实测卡死过）
```

## 工作流

```text
改代码 -> ./gradlew :app:assembleDebug
       -> adb push apk + pm install --user 0 -r
       -> dumpsys package ... lastUpdateTime  核验真的装上了 (必做!)
       -> am start --user 0 ... --ez autorun true
       -> run-as <pkg> cat files/voicedesign/vd_run.log   看进度
```

## 验证纪律

- 结论必须有**逐位/逐 id** 的证据（cos / exact_equal / 逐 id 比对），不能"看起来对"
- 数值改动后重跑对应 Gate，把数据写进 `docs/VALIDATION.md`
- 失败路线与成功路线同等记录（写进 `docs/DEVELOPMENT_STORY.md` 的"复盘"）
- 关键词搜索 = discovery，不是 proof。冻结事实要结构化验证

## 排查手册（优先用这些）

| 症状 | 先做什么 |
|---|---|
| 原生崩溃 | `llvm-addr2line -e libvoicedesign_jni.so -f -C 0x<pc>`；看故障地址形态（页对齐=映射边界，0x0=空指针）|
| 卡住不出日志 | 看 `/proc/<pid>/stat` 的 utime 是否冻结；冻结=阻塞（DSP/IO），不冻结=在算 |
| 行为没变化 | 先核验 APK 真的装上了（`lastUpdateTime`）|
| 跑到一半死 | `logcat -b events \| grep am_proc_died` 看是不是被杀 |
| 装不上 | `pm install --user 0`；设备有多用户（平行空间）|

## 文档约定

- 根 `README.md` 保持"能让人跑起来"的完整性
- 一次性细节写 `docs/`，不塞 README
- 每条坑必须给：**症状 / 根因 / 修法 / 判据**（见 PITFALLS 的格式）

## 禁止

- ❌ `git reset --hard` 清问题
- ❌ 删除用户数据 / 模型证据 / runs
- ❌ 为了让测试过而降 Gate
- ❌ 把未知结果写成确定事实
- ❌ 把 PAT / token / 密钥写进任何文件
