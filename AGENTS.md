# AGENTS.md

This file provides guidance to Qoder (qoder.com) when working with code in this repository.

## Project Overview

WatchHand Android — 复现论文 WatchHand (arXiv:2602.21610, ACM CHI 2026)，用智能手表/手机的扬声器+麦克风发射 18-20kHz FMCW 超声信号，通过回声轮廓+深度学习做连续手部感知。当前阶段：手势分类闭环已跑通（采集→提取→训练→测试→ONNX→服务端实时预测），跨会话 window-acc ~0.56，持续采集中；远期目标 20 关节 3D 追踪。完整上下文见 CONTEXT.md。

## Build & Run

```bash
# 构建（需要 JDK 17，推荐 Oracle/Corretto 17）
export JAVA_HOME=/Library/Java/JavaVirtualMachines/jdk-17.jdk/Contents/Home
gradle :app:assembleDebug --no-daemon

# 安装到设备
adb install -r app/build/outputs/apk/debug/app-debug.apk

# 截图
adb exec-out screencap -p > screenshot.png
```

- **技术栈**: AGP 8.7.3, Kotlin 2.1.0, Compose BOM 2024.06.00, Gradle 8.7
- **SDK**: compileSdk=34, minSdk=26, targetSdk=34
- **权限**: `RECORD_AUDIO`, `MODIFY_AUDIO_SETTINGS`
- **无测试框架配置**，项目当前无单元测试

## Architecture

整个 app 是一个单 Activity 的 Compose 应用，核心是实时音频信号处理管线。四个类形成清晰的数据流：

```
FmcwGenerator → AudioManager → EchoProfileProcessor → MainActivity (Compose UI)
 (信号生成)      (音频调度)      (回声轮廓计算)         (可视化)
```

### FmcwGenerator
- 生成 18-20kHz FMCW 扫频 chirp（588 samples = 13.333ms @ 统一 44.1kHz，75fps）
- **汉宁窗**: 对 chirp 加 Hann window 消除起止瞬态失真和频谱泄漏，这是消除可听蜂鸣声的关键
- `generatePlaybackBuffer()` 生成循环播放的长缓冲区

### EchoProfileProcessor — 核心算法 (论文 Algorithm 1)
- **带通滤波**: 3 级 HP(18kHz) + 2 级 LP(20kHz) biquad 级联 ≈ 5 阶 Butterworth，滤波器状态跨 `feed()` 调用保持
- **对齐约定（三端一致）**：相关约定 lag = startOffset + d（Σ seg[k+j]·tx[j]，非卷积 k−j）；startOffset = p_start（b0 逐 chirp 平均 |corr| 最大 bin，候选 (b0−5)%L/(5−b0)%L + 直达峰验证，DIRECT_BIN=5）
- **锁定生命周期**：SNR ≥ 9 且连续 8 个 chirp 边界通过才锁；服务端流间断 >500ms 重建处理器
- **流式实现**：逐 chirp 分段相关（前 2 chirp 作 overlap，与整段互相关切片等价），每满一个 chirp 产出一帧，96 帧滚动窗口
- **差分轮廓**: `|P[f]| - |P[f-1]|` 帧间幅度差
- **输出布局**: 列优先 `[distanceBins × frameCount]`，即 `profile[d * timeWindowFrames + f]`，原始轮廓保留带符号值

### AudioManager
- 同时管理 AudioTrack（播放 FMCW）和 AudioRecord（录音），使用 `UNPROCESSED` 音频源，失败时回退到 `DEFAULT`
- **采样率回退**: 优先使用设备原生采样率（手机 48kHz，手表 44.1kHz），避免重采样导致频率偏移
- 录音线程中实时读取 → 喂给 `EchoProfileProcessor.feed()` → 回调 UI
- 30 秒播放缓冲区循环写入

### MainActivity (Compose)
- `EchoProfileHeatmap` composable 渲染热力图
- 原始和差分轮廓共用蓝色系 sequential 色图（蓝→青→洋红→黄），匹配论文 Figure 4/5
- 差分轮廓用对称 clipping（±98th percentile of |values|），确保 0 映射到蓝色背景
- 可视化用 2%/98% percentile clipping（自适应不同硬件），训练时不做 clipping

## Critical Signal Processing Constraints

这些是从调试中验证的关键约束，修改信号处理代码时必须遵守：

1. **流式与批处理同网格**: 流式逐 chirp 分段相关与 extract.py 整段 FFT 互相关必须用同一 p_start 对齐约定（lag = startOffset + d）；卷积 k−j 或论文式 (bestIdx+L/2) 公式会把直达声放到窗口外，表现为 Pred 恒定/幅度缩 45 倍。
2. **流式处理用环形缓冲区**: 增量 feed → 每满一个 chirp 产出一帧 → 滚动窗口。不要累积后每次处理整个窗口。
3. **AudioTrack/AudioRecord 停止顺序**: stop 前必须检查 `playState`/`recordingState`，否则 `IllegalStateException`。全部包 try-catch。
4. **输出数组是列优先**: `profile[distanceBin * frameCount + frameIndex]`，不是行优先。
5. **统一采样率 44.1kHz**: 所有设备一律请求 44100Hz（手表原生；48kHz 手机经系统 SRC，2026-08-19 实验验证不闪）。统一后 chirp/帧率/距离网格跨设备天然一致。
6. **chirp 必须加汉宁窗**: 矩形窗 chirp 起止处有瞬态突变，产生频谱泄漏和可听谐波。汉宁窗平滑起止，消除可听蜂鸣声。
7. **跨硬件统一参数规则**: 采样率 44.1kHz、频段 18-20kHz、chirp 588 samples；客户端 `AudioManager.CHIRP_LENGTH` 与服务端 `WatchHandServer.CHIRP_LENGTH` 必须保持同步；`extract.py` 的 `REF_FS=44100`（旧 48kHz 会话经插值对齐）。

## Key Parameters

| 参数 | 值 | 来源 |
|------|-----|------|
| 采样率 | 44.1 kHz（所有设备统一） | 手表原生；手机经 SRC（实验验证可用） |
| FMCW 频段 | 18-20 kHz | Nyquist 余量 |
| Chirp 长度 | 588 samples (13.333ms) | 帧率 75fps |
| 距离分辨率 | 3.89 mm/pixel | C/(2×44100)，跨设备天然一致 |
| 距离 bins | 60 | 每帧取前 60 个样本（23.3cm） |
| 时间窗口 | 96 frames (1.28s @ 75fps) | 滚动窗口 |
| 播放音量 | 20% | 手机扬声器功率大，20% 足够 |

## 服务端与训练管线

- **Java 服务端**（server/WatchHandServer.java，唯一处理端）：流式轮廓 + Swing 热力图 + 引导式采集 UI + ONNX Runtime 实时预测（Pred label，每 2 帧推理取末步 argmax）。编译 `javac -cp "lib/*" WatchHandServer.java`，启动 `./start_server.sh`。
- **训练管线**（server/train/）：extract.py（整段 FFT 互相关 + p_start 对齐 + 双通道 + 滑窗 stride 22 + 会话级缓存，无设备校准）→ train.py（帧 CNN + 因果 Transformer，默认 3.74M，10 类 × 12 密集步，last.pt 原子保存自动续训）→ test.py（--split）→ export_onnx.py；`run_pipeline.sh` 一键串联。
- **采集协议**：train/ 放混合会话（classes "0,1,2"），test/ 放单类会话；边界不插 0；无 mask。

## Pending Work (from CONTEXT.md)

- 更多混合会话数据（跨时段/摆位），手势扩到 10 类
- 数据足够后试更大容量 / FastViT 基线
- Ground Truth 采集（MediaPipe Hands）→ 3D 回归
