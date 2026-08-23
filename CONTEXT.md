# WatchHand 项目上下文

> 记录项目所有关键上下文与决策，用于跨会话无缝继续开发。最后更新：2026-08-23。

---

## 1. 项目概述

复现论文 WatchHand（arXiv:2602.21610, CHI 2026）：商用设备扬声器+麦克风发射 FMCW 超声，回声轮廓 + 深度学习做连续手部感知。

**当前阶段**：手势分类闭环跑通（采集→提取→训练→测试→ONNX→服务端实时预测）。跨会话 window-acc ~0.56（混合训练+单类测试协议），瓶颈是数据量，持续采集中。远期目标仍是论文的 20 关节 3D 追踪（需 MediaPipe ground truth）。

---

## 2. 架构与代码结构

```
app/                          Android 客户端（Kotlin/Compose）
  FmcwGenerator.kt            chirp 生成（汉宁窗，与服务端模板一致）
  EchoProfileProcessor.kt     本地流式轮廓（逐 chirp 分段相关）
  AudioManager.kt             播放+录音；连接时只发 raw，跳过本地处理
  TcpAudioClient.kt           单例 TCP；header 在首包数据时发；串行发送防乱序
  MainActivity.kt             UI + 本地热力图 + 连接控制
server/
  WatchHandServer.java        Java 服务端（唯一处理端；JDK17 + server/lib/onnxruntime jar）
  start_server.sh             启动（-cp ".:lib/*"，默认端口 9999）
  train/
    extract.py                raw→双通道窗口（带会话级缓存 profile_cache/）
    train.py                  训练（last.pt 每 epoch 原子覆盖、自动续训）
    test.py                   评估（--split test/train/all）
    export_onnx.py            last.pt → last.onnx
    run_pipeline.sh           一键 extract→train→test→export
    train_window.py           FastViT-T12 基线（历史对照）
```

---

## 3. 统一信号参数（三端一致，改动需同步）

| 参数 | 值 | 同步点 |
|------|-----|--------|
| 采样率 | 44.1kHz（48k 硬件经 SRC，已验证不闪） | 客户端请求值 |
| 频段 | 18–20kHz | `fMaxFor(sampleRate)` 客户端/服务端同步 |
| Chirp | 588 samples @75fps | 客户端 `CHIRP_LENGTH` ↔ 服务端 `CHIRP_LENGTH` |
| 距离 | 60 bins × 3.89mm | extract / 服务端 / APK |
| 窗口 | 96 帧（1.28s），stride 22 | extract ↔ train |
| DIRECT_BIN | 5 | extract / 服务端 / APK |

---

## 4. 对齐约定与锁定生命周期（核心教训）

- **相关约定**：bin d 取 lag = `startOffset + d`（Σ seg[k+j]·tx[j]），**不是卷积 k−j**；`startOffset = p_start`（b0 逐 chirp 平均 |corr| 最大 bin，候选 (b0−5)%L / (5−b0)%L + 直达峰验证）。旧卷积+论文式 (bestIdx+L/2) 公式把直达声放到 bin≈295（窗口外），表现为 Pred 恒定、幅度缩 45 倍。
- **锁定生命周期**：SNR = max|corr|/mean|corr| ≥ 9（信号 13.5–14.2，静默 ≈5.2，带零能量保护）且**连续 8 个 chirp 边界**通过才锁；服务端额外做**流间断 >500ms 重建处理器**（客户端 stop→start 断流 ⇒ 等价本地"处理器随音频会话生灭"）。
- socket 有序连续，首包即会话 t=0，与本地同起点；不需要传输层静默建模。

---

## 5. 采集与划分协议

- 引导式循环采集：classes 列表（**train 用 "0,1,2" 混合**，0=rest trial）、trial 2.5s、rounds 可配、test set 复选框决定目录。
- **边界不插零长度 0 事件**（用户要求；训练不需要 mask）；仅启动后 2s 准备期为 0。
- **划分**：train/ = 混合会话，test/ = 单类会话。
- 数据三件套 .raw/.labels/.meta；历史带边界标记的 .labels 已清洗。

---

## 6. 训练管线

- **extract.py**：批处理整段互相关（FFT，`irfft(spec, size)`）→ p_start 对齐 → 双通道（原始带符号 + 差分 |P[f]|−|P[f−1]|）→ 滑窗。**无设备校准**（决策：稳定伪影/漂移交给模型学；校准曾实现又删，PROCESS_VERSION=3 使旧缓存失效）。会话级缓存按 (版本, raw/meta size+mtime) 失效。
- **train.py**：帧 CNN（距离轴 2× 下采样×3，时间不降采样）→ 因果 Transformer（默认 4 层 d256 = 3.74M；`--layers/--dmodel` 调容量）→ 密集头：后半窗 12 步（48::4，~54ms）× **10 类**（`--classes` 默认 10，空类留位）。损失 = 12 步普通交叉熵。增广：位移±2 + 幅度抖动 + **输入级 drop**（`--drop` 默认 0.3，随机抹时间/距离段）+ dropout 0.2。
- **续训**：last.pt 每 epoch 原子保存（.tmp + os.replace）；只拦截影响形状的配置（dmodel/layers/classes），dropout/drop 改了可无缝续训。损坏检查点改名归档不删。
- **容量策略**：容量跟数据量匹配——小噪数据用小模型（2 层 d128），干净混合数据用默认大模型；数据涨一档试一次更大容量，val 定取舍。
- **head-only 适配**：扩类时旧头权重前 N 行移植到新头、冻结 backbone 先训头 5–10 epoch，phase check 后决定是否解冻。

---

## 7. 实时推理

- `export_onnx.py` 导出（torch vs onnx 测试集 100% 一致）；服务端 ONNX Runtime 每 2 帧推理一次，取末步 argmax 显示 `Pred: N`（0 灰/1 绿/2 红/其余蓝）。
- 服务端输入构造与 train 完全一致：edge pad 60→64、差分通道、逐通道窗口归一化。

---

## 8. 关键决策与教训（精选）

1. **逐 chirp 分段相关**取代整窗 FFT 重算（0.05ms/帧 vs 100ms/帧），帧网格锚定 chirp 整数倍 ⇒ 不闪。
2. **闪烁真因**是帧网格漂移+算力积压，非 SRC；统一 44.1kHz 可行。
3. **原始轮廓保留带符号值**（论文约定），abs 只进差分。
4. **不做设备校准**；跨会话泛化瓶颈是数据量与摆位漂移，混合训练解耦类别与会话漂移。
5. **产物不删只改名归档**；**只执行用户指定步骤**（不擅自训练/截图）。
6. 训练日志 train acc 在 drop 增广样本上计算，低于 test.py --split train 的干净评估，属口径差异。

---

## 9. 当前状态与下一步

- 已完成：闭环全链路、对齐/锁定生命周期、采集协议、10 类头、缓存、原子保存、文档沉淀、全部入 git。
- 指标：train 0.98+；test window-acc ~0.5–0.56（数据受限）。
- 下一步：① 更多混合会话数据（不同时段/摆位）；② 手势扩到 10 类；③ 数据足够后试更大容量/FastViT；④ 远期 MediaPipe ground truth + 3D 回归。

---

## 10. 仓库与构建

- 远程：git@github.com:guyinyou/watchhand.git
- APK：`./build_apk.sh [--install <serial>]`（JDK17 + Gradle 9.7 + AGP 8.7.3）
- 服务端：`javac -cp "lib/*" WatchHandServer.java && ./start_server.sh`
- 训练：python 3.10（/Library/Frameworks/Python.framework/Versions/3.10）+ torch 2.5.1（MPS 稳定）
- .gitignore 排除：dataset*.npz / exp*.npz（>100MB GitHub 限制）、profile_cache/、__pycache__/、*.log；克隆后 `extract.py` 可重建数据集。
