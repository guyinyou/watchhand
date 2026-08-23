# WatchHand 分体式架构

本项目支持分体式架构：Android 设备作为采集端，PC 作为处理端。

## 架构

```
┌─────────────────┐         TCP          ┌─────────────────┐
│   Android APK   │  ──────────────────> │  Python Server  │
│   (采集端)       │   原始 PCM 音频流      │   (处理端)       │
│                 │                      │                 │
│ - 播放 FMCW     │                      │ - 接收 PCM      │
│ - 录制音频      │                      │ - 信号处理      │
│ - 发送原始数据  │                      │ - 实时可视化    │
└─────────────────┘                      └─────────────────
```

## 使用步骤

### 1. 启动 Python 服务端

```bash
cd server
pip install -r requirements.txt
python3 watchhand_server.py --host 0.0.0.0 --port 9999
```

服务端会在 `0.0.0.0:9999` 监听连接。

### 2. 配置 Android APK

1. 打开 WatchHand APK
2. 在 "TCP 数据流" 卡片中：
   - 打开开关启用 TCP 流
   - 输入服务端的 IP 地址（如 `192.168.1.100`）
   - 输入端口（默认 `9999`）
3. 点击 "开始采集"

### 3. 查看可视化

服务端会自动显示两个热力图：
- **Original Echo Profile**: 原始回声轮廓
- **Differential Echo Profile**: 差分回声轮廓

## TCP 协议

### 连接握手

1. 客户端发送 `"WATCHHAND\n"` (10 bytes)
2. 客户端发送采样率 (4 bytes, little-endian int32)
3. 客户端发送通道数 (1 byte) 和位深 (1 byte)

### 数据传输

- 原始 PCM 音频数据，16-bit little-endian
- 持续流式传输，无包边界

## 参数说明

| 参数 | 值 | 说明 |
|------|-----|------|
| 采样率 | 48 kHz (手机) / 44.1 kHz (手表) | 设备原生采样率 |
| FMCW 频段 | 18-21 kHz | 论文参数 |
| Chirp 长度 | 600 samples | 12.5ms @ 48kHz |
| 距离分辨率 | 3.57 mm/pixel | C/(2×Fs) |
| 距离 bins | 60 | 21.4 cm 范围 |
| 时间窗口 | 96 frames | 1.2s 滚动窗口 |

## 故障排除

### 连接失败

- 确保手机和 PC 在同一网络
- 检查防火墙设置
- 确认端口未被占用

### 无数据显示

- 检查 APK 是否显示 "已连接"
- 确认服务端日志显示 "Connected"
- 检查采样率是否匹配

## 开发说明

### APK 端关键文件

- `TcpAudioClient.kt`: TCP 客户端实现
- `AudioManager.kt`: 音频管理，集成 TCP 发送
- `MainActivity.kt`: UI，TCP 配置

### 服务端关键类

- `WatchHandServer`: TCP 服务器主类
- `EchoProfileProcessor`: 信号处理管线
- `BiquadFilter`: 双二阶滤波器状态
