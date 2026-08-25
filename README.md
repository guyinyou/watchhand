# WatchHand

复现论文 **WatchHand**（arXiv:2602.21610, ACM CHI 2026）：仅用商用设备的扬声器 + 麦克风发射 18–20kHz FMCW 超声，通过回声轮廓 + 深度学习做连续手势识别/手部追踪。

当前阶段：**手势分类闭环已跑通**（采集 → 提取 → 训练 → 测试 → ONNX 导出 → 服务端实时预测），跨会话 window-acc ~0.56，持续采集中。

## 架构

```
┌─────────────┐  raw PCM (TCP)   ┌──────────────────────────────┐
│ Android APK │ ───────────────► │ Java 服务端 (WatchHandServer) │
│ 播放+录音    │                  │  流式回声轮廓 + 热力图         │
│ 本地热力图   │                  │  引导式采集 UI + 标签          │
└─────────────┘                  │  ONNX 实时预测 (Pred label)   │
                                 └──────────────────────────────┘
                                            ▲ last.onnx
                                 ┌──────────┴─────────┐
                                 │ 训练管线 (server/train)│
                                 │ extract→train→test  │
                                 │ →export_onnx        │
                                 └────────────────────┘
```

- **APK**（`app/`）：FMCW 播放 + 录音 + TCP 流式上传 + 本地热力图。连接与流媒体状态分离，连接后自动开始流。
- **Java 服务端**（`server/WatchHandServer.java`）：逐 chirp 分段相关的流式回声轮廓、Swing 热力图、引导式循环采集（按 trial 自动切类写标签；trial 间静默、结束才 beep，避免录入提示音）、ONNX Runtime 实时预测。
- **训练管线**（`server/train/`）：见下。

## 快速开始

```bash
# 1. 服务端（JDK 17；ONNX 依赖在 server/lib/）
cd server && ./start_server.sh            # 默认端口 9999

# 2. APK
./build_apk.sh --install <device-serial>

# 3. 采集：服务端 UI 勾选 test set 决定存 train/ 或 test/；
#    classes 填 "0,1,2" = 混合会话（进 train/，0=手放平的 rest trial），
#    单类会话进 test/；2s 准备期标签=第一类，类别变化才写标签事件

# 4. 迭代训练（python 3.10）
cd server/train
./run_pipeline.sh                          # extract → train(续训) → test → export onnx
python3 test.py --device mps               # 只看 test
python3 test.py --device mps --split train # 只看 train
```

## 关键参数（全设备统一）

| 参数 | 值 |
|------|-----|
| 采样率 | 44.1kHz（48k 硬件经 SRC，已验证可用） |
| 频段 | 18–20kHz |
| Chirp | 588 samples = 13.333ms，帧率 75fps |
| 距离网格 | 60 bins × 3.89mm |
| 模型窗口 | 96 帧 = 1.28s，stride 22 滑窗 |
| 直达声锚定 | bin 5（DIRECT_BIN，三端一致） |
| 模型 | 帧 CNN + 因果 Transformer，默认 3.74M（4 层 d256），输出 10 类 × 12 密集步 |

## 文档

- [CONTEXT.md](CONTEXT.md)：完整项目上下文（决策、教训、管线细节、当前状态）
- [AGENTS.md](AGENTS.md)：给 AI 代理的工程约束
- [server/README.md](server/README.md)：服务端说明
- [docs/paper-full-text.txt](docs/paper-full-text.txt)：论文全文

## 仓库说明

`dataset.npz` / `profile_cache/` 等 >100MB 或可再生产物被 .gitignore 排除（GitHub 单文件限制），克隆后跑 `python3 extract.py` 即可从 `collected_data/` 重建。
