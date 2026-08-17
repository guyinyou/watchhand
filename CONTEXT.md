# WatchHand 项目上下文

> 本文档记录项目所有关键上下文，用于跨机器/跨会话无缝继续开发。

---

## 1. 项目概述

**目标**：复现论文 WatchHand（arXiv:2602.21610, ACM CHI 2026），实现基于智能手表/手机的连续 3D 手部姿态追踪。

**论文核心**：仅用商用智能手表的内置扬声器 + 麦克风，发射 18-21kHz FMCW 超声信号，通过回声轮廓 + 深度学习实现 20 个手指关节的 3D 姿态追踪。

**当前阶段**：数据预处理 + 可视化管线已验证通过，待进入数据采集和模型训练阶段。

---

## 2. 代码结构

```
watchhand-android/
├── build.gradle.kts              # AGP 8.7.3 + Kotlin 2.1.0 + Compose Plugin 2.1.0
├── settings.gradle.kts
├── gradle.properties
├── local.properties              # SDK 路径（需根据机器修改）
├── gradle/wrapper/
│   └── gradle-wrapper.properties # Gradle 8.7
└── app/
    ├── build.gradle.kts          # minSdk=26, targetSdk=34, Compose BOM 2024.06.00
    └── src/main/
        ├── AndroidManifest.xml   # RECORD_AUDIO + MODIFY_AUDIO_SETTINGS 权限
        └── java/com/watchhand/app/
            ├── FmcwGenerator.kt          # FMCW 扫频信号生成（18-21kHz, 600 samples/chirp）
            ├── EchoProfileProcessor.kt   # 核心：回声轮廓处理（滤波→互相关→reshape→差分）
            ├── AudioManager.kt           # 音频采集管理（同时播放+录音+实时处理）
            └── MainActivity.kt           # UI + 热力图可视化
```

**Git 仓库**：https://github.com/guyinyou/watchhand

---

## 3. 构建环境

### 必需工具
- **JDK 17**（Microsoft OpenJDK 21 有 bug，必须用 Oracle/Corretto 17）
- **Gradle 9.7**（通过 Homebrew 安装：`brew install gradle`）
- **Android SDK**：
  - Platform 34（compileSdk）
  - Build-Tools（最新版）
  - Command-line Tools
- **AGP 8.7.3**（兼容 Gradle 9.x）

### 构建命令
```bash
export JAVA_HOME=/Library/Java/JavaVirtualMachines/jdk-17.jdk/Contents/Home
cd watchhand-android
gradle :app:assembleDebug --no-daemon
```

### 安装到设备
```bash
adb install -r app/build/outputs/apk/debug/app-debug.apk
```

### 截图调试
```bash
adb exec-out screencap -p > screenshot.png
```

---

## 4. 信号处理管线（核心）

### 参数
| 参数 | 值 | 说明 |
|------|-----|------|
| 采样率 | 48 kHz | Android AudioRecord/AudioTrack |
| FMCW 频段 | 18-21 kHz | 不可闻超声波 |
| Chirp 长度 | 600 samples | 12.5ms |
| Chirp 速率 | 80 Hz | 48000/600 |
| 距离分辨率 | 3.57 mm/pixel | C/(2×Fs) = 343/96000 |
| 距离范围 | 21.42 cm | 60 bins × 3.57mm |
| 时间窗口 | 96 frames | 1.2 秒 |

### Algorithm 1 实现步骤

1. **带通滤波**：3 级 HP (18kHz) + 2 级 LP (21kHz) biquad 级联 ≈ 5 阶 Butterworth
   - 系数用 Audio EQ Cookbook 公式计算
   - 滤波器状态跨 feed 调用保持（流式处理）

2. **起始位置检测**：收集前 4 个 chirp 的数据，与 tx chirp 做互相关找直达声峰值
   - 论文公式：`p_start = (p_start + L - ⌊L/2⌋) mod L`

3. **整段互相关**（关键！）：
   - **不是**逐 chirp 互相关（会导致边界错位）
   - **而是**对累积的整段滤波后数据做一次大互相关
   - 然后 reshape 成帧（每帧 600 samples）
   - 每帧取前 60 个样本作为距离 bin

4. **差分轮廓**：`|P[f]| - |P[f-1]|`（帧间幅度差）

5. **输出布局**：列优先 `[distanceBins × frameCount]`

---

## 5. 可视化

### 色图
- **原始回声轮廓**：viridis（紫→绿→黄）
- **差分回声轮廓**：diverging（蓝→黑→黄）

### Clipping 策略
- **论文**：固定 ±10¹⁰（针对手表硬件调的）
- **我们**：2%/98% percentile clipping（自适应不同硬件）
- **训练时**：不做 clipping，只用窗口归一化（论文要求）

### 背景色
- Canvas 先画 viridis(0) 作为底色（深紫蓝），匹配论文 Figure 4/5

### UI 显示
- 参数信息卡片显示可视化范围（2%-98% percentile 值）

---

## 6. 关键决策与教训

### 6.1 逐 chirp 互相关 ≠ 整段互相关
**错误做法**：把录音切成单个 chirp → 每个 chirp 单独做互相关
**正确做法**：累积整段数据 → 一次大互相关 → reshape

原因：startOffset 的存在导致逐 chirp 切分会跨边界混合数据，互相关结果完全错误。

### 6.2 流式处理 vs 批处理
**错误做法**：累积数据 → 每次处理整个窗口 → 帧数跳动
**正确做法**：环形缓冲区 → 增量 feed → 每满一个 chirp 产出一帧 → 滚动窗口

### 6.3 停止闪退
AudioTrack.stop() / AudioRecord.stop() 在状态不对时抛 IllegalStateException。
**解决**：先检查 playState/recordingState，再 stop，全部包 try-catch。

### 6.4 FMCW 频率选择：SNR vs 可闻性
- 18-21kHz 理论上不可闻，但年轻人能听到 18kHz
- 扬声器非线性失真会产生可听谐波（互调失真）
- 手机扬声器功率大 → 震动感明显
- 手表扬声器功率小（61-81 dBA）→ 可闻性低
- **这是 trade-off，无法完全消除**

### 6.5 设备差异
- 不同手表的扬声器/麦克风位置不同 → 回声轮廓时空模式不同
- 数据集不能跨设备直接使用
- 论文有通用校准算法（峰值错位校正 + 周期性漂移校准），我们尚未实现

---

## 7. 设备兼容性

### 已验证
| 设备 | 系统 | 采样率 | 状态 |
|------|------|--------|------|
| nubia NX769J (Z70 Ultra) | Android 16 | 48 kHz | ✅ 管线通，SNR 低，可闻性高 |

### 论文验证设备
| 设备 | 扬声器功率 (10cm) | 状态 |
|------|------------------|------|
| Samsung Galaxy Watch 7 | 61.6 dBA | ✅ 论文验证 |
| Xiaomi Watch 2 Pro | 66.9 dBA | ✅ 论文验证 |
| Google Pixel Watch 3 | 80.8 dBA | ✅ 论文验证 |

### 硬件要求
- 扬声器 + 麦克风（至少各一个）
- 支持 48kHz 16-bit PCM 同时播放和录制
- Android API 26+（AudioRecord.Builder / AudioTrack.Builder / UNPROCESSED 音频源）

---

## 8. 当前状态

### 已完成
- [x] FMCW 信号生成器
- [x] 带通滤波（biquad 级联）
- [x] 起始位置检测
- [x] 整段互相关 + reshape
- [x] 差分回声轮廓
- [x] 流式处理（环形缓冲区）
- [x] 可视化（viridis + diverging 色图 + percentile clipping）
- [x] 停止闪退修复
- [x] 帧数稳定（96 帧滚动窗口）
- [x] APK 构建 + 安装

### 待完成
- [ ] 窗口归一化（训练用，论文 Section 3.3.2）
- [ ] 设备校准算法（峰值错位校正 + 周期性漂移校准，论文 Section 3.3.3）
- [ ] 数据采集方案设计（手势协议、session 设计）
- [ ] Ground Truth 采集（MediaPipe Hands + webcam）
- [ ] 模型训练（FastViT-T12 回归 20 关节 3D 坐标）
- [ ] 三种训练协议（within-session / cross-session / cross-user）

---

## 9. 下一步行动

### 立即可做
1. **继续用手机调试**：提高音量找到 SNR sweet spot，验证手部移动时能看到曲线模式
2. **设计数据采集方案**：参考论文 Study 1 的 18 种手势协议

### 中期
3. **获取/采集数据集**：论文已开源 35.6 小时数据集（40 人）
4. **实现归一化 + 校准**
5. **搭建训练环境**（Python + PyTorch + FastViT-T12）

### 长期
6. **上手表设备**：购买 Galaxy Watch 7 或类似 WearOS 手表
7. **端侧部署**：模型量化到 26.7MB，在手表上实时推理（0.115s 延迟）

---

## 10. 参考资源

- 论文：https://arxiv.org/abs/2602.21610
- 论文 HTML：https://arxiv.org/html/2602.21610v1
- 论文 PDF（ACM）：https://dl.acm.org/doi/10.1145/3772318.3790932
- 数据集：论文提到已开源，需查找具体发布位置
- FastViT-T12：https://github.com/apple/ml-fastvit
- MediaPipe Hands：https://google.github.io/mediapipe/solutions/hands

---

## 11. 常见问题

### Q: 为什么热力图偏黄？
A: 直达声值远大于其他区域，min-max 归一化后大部分值落在高范围。用 percentile clipping 解决。

### Q: 为什么有震动感/可闻性？
A: 18kHz 年轻人能听到 + 扬声器非线性失真产生可听谐波。手机扬声器功率大，问题更明显。

### Q: 数据集拿到能直接用吗？
A: 不能。不同设备的扬声器/麦克风位置不同，回声轮廓模式不同。需要用自己的设备采集数据。

### Q: 训练时需要 clipping 吗？
A: 不需要。论文说 "to preserve maximum information, we use the raw unclipped echo profiles during training"。只用窗口归一化。

### Q: 可视化范围和论文不一样？
A: 正常。±10¹⁰ 是论文针对手表硬件调的。不同硬件的值范围不同，percentile clipping 自动适应。

---

*最后更新：2026-08-17*
