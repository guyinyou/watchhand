package com.watchhand.app

import android.Manifest
import android.content.Context
import android.content.SharedPreferences
import android.content.pm.PackageManager
import android.os.Bundle
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat
import kotlinx.coroutines.delay

/** 本地录制状态机：IDLE -> RECORDING -> CONFIRM_SAVE -> IDLE */
enum class RecState { IDLE, RECORDING, CONFIRM_SAVE }

class MainActivity : ComponentActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        // 页面常亮：采集/录制期间避免熄屏（前台有效，无需权限）
        window.addFlags(android.view.WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        setContent {
            MaterialTheme {
                WatchHandScreen()
            }
        }
    }
}

@Composable
fun WatchHandScreen() {
    val context = LocalContext.current
    
    // Load saved server config from SharedPreferences
    val prefs = context.getSharedPreferences("watchhand_config", Context.MODE_PRIVATE)
    val savedHost = prefs.getString("server_host", "30.221.108.126") ?: "30.221.108.126"
    val savedPort = prefs.getInt("server_port", 9999)
    
    var isRecording by remember { mutableStateOf(false) }
    var statusText by remember { mutableStateOf("等待启动") }
    var originalProfile by remember { mutableStateOf<FloatArray?>(null) }
    var differentialProfile by remember { mutableStateOf<FloatArray?>(null) }
    var distBins by remember { mutableIntStateOf(AudioManager.DISTANCE_BINS) }
    var frameCount by remember { mutableIntStateOf(0) }
    var audioManager by remember { mutableStateOf<AudioManager?>(null) }
    // 统一采样率 44100（所有设备一致；TCP 连接可能早于录音启动，header 必须携带正确采样率）
    var actualRate by remember { mutableIntStateOf(44100) }

    // TCP streaming configuration - initialized from saved preferences
    var serverHost by remember { mutableStateOf(savedHost) }
    var serverPort by remember { mutableIntStateOf(savedPort) }
    var tcpClient by remember { mutableStateOf<TcpAudioClient?>(null) }
    var tcpStatus by remember { mutableStateOf("未连接") }
    var isTcpConnecting by remember { mutableStateOf(false) }

    // 本地训练数据录制状态
    var recState by remember { mutableStateOf(RecState.IDLE) }
    var recLabel by remember { mutableIntStateOf(0) }
    var recElapsedSec by remember { mutableIntStateOf(0) }
    var recResult by remember { mutableStateOf<LocalRecorder.Result?>(null) }
    var pendingRecordLabel by remember { mutableIntStateOf(-1) }  // 等待录音权限授予后要开始的录制 label

    // Percentile clipping range for visualization
    var visRange by remember { mutableStateOf("-") }

    // Debug log displayed on UI
    var debugLog by remember { mutableStateOf("") }
    val appendLog: (String) -> Unit = { msg ->
        val timestamp = java.text.SimpleDateFormat("HH:mm:ss.SSS", java.util.Locale.getDefault()).format(java.util.Date())
        debugLog = "[$timestamp] $msg\n$debugLog"
        // Keep last 50 lines
        val lines = debugLog.split("\n")
        if (lines.size > 50) {
            debugLog = lines.take(50).joinToString("\n")
        }
    }

    val onProfileUpdate: (FloatArray, FloatArray, Int, Int) -> Unit = { orig, diff, bins, frames ->
        originalProfile = orig.copyOf()
        differentialProfile = diff.copyOf()
        distBins = bins
        frameCount = frames
        statusText = "采集中... $frames frames"

        // Calculate percentile range for display
        val sorted = orig.sortedArray()
        val lo = sorted[(sorted.size * 0.02).toInt()]
        val hi = sorted[(sorted.size * 0.98).toInt()]
        visRange = "${String.format("%.2e", lo)} ~ ${String.format("%.2e", hi)}"
    }

    val onStatus: (String) -> Unit = { msg ->
        statusText = msg
        appendLog(msg)
        // Parse actual sample rate from status message like "录音中... (44100Hz)"
        val regex = Regex("""\((\d+)Hz\)""")
        regex.find(msg)?.groupValues?.get(1)?.toIntOrNull()?.let { rate ->
            actualRate = rate
        }
    }

    // TCP connect/disconnect function
    val onTcpConnectClick: () -> Unit = {
        val current = TcpAudioClient.instance
        if (current?.isConnected() == true) {
            // Disconnect
            current.disconnect()
            tcpClient = null
            tcpStatus = "未连接"
            isTcpConnecting = false
            appendLog("TCP 已断开")
        } else {
            // Connect
            isTcpConnecting = true
            tcpStatus = "连接中..."
            appendLog("正在连接 $serverHost:$serverPort...")
            val client = TcpAudioClient(serverHost, serverPort, actualRate, { status ->
                tcpStatus = status
                isTcpConnecting = false
                appendLog("TCP: $status")
                // Save to SharedPreferences when connected successfully
                if (status.contains("已连接")) {
                    prefs.edit()
                        .putString("server_host", serverHost)
                        .putInt("server_port", serverPort)
                        .apply()
                    appendLog("已保存服务器配置")
                }
            }, appendLog)
            tcpClient = client
            client.connect()
        }
    }

    // Update actual rate when audioManager changes
    LaunchedEffect(audioManager) {
        audioManager?.let { actualRate = it.actualSampleRate }
    }

    // 启动音频管线（FMCW 播放 + 录音），若已在运行则跳过
    val startAudioPipeline: () -> Unit = {
        if (audioManager == null) {
            isRecording = true
            statusText = "正在初始化..."

            appendLog("Creating AudioManager (TCP singleton: ${TcpAudioClient.instance != null})")

            val manager = AudioManager(
                context = context,
                onEchoProfileUpdate = onProfileUpdate,
                onStatusUpdate = onStatus,
                onSampleRateUpdate = { rate ->
                    actualRate = rate
                    // 录制会话开始后首个样本到达前修正实际采样率（保证时长上限与 meta 一致）
                    LocalRecorder.instance?.updateSampleRate(rate)
                    appendLog("实际采样率: ${rate}Hz")
                }
            )
            audioManager = manager
            manager.start()
        }
    }

    // 停止音频管线
    val stopAudioPipeline: () -> Unit = {
        audioManager?.stop()
        audioManager = null
        isRecording = false
        statusText = "已停止"
    }

    // 结束本次录制（手动停止 / 到时自动停止共用）：震动提醒 + 进入保存确认态
    val stopRecordingFlow: () -> Unit = {
        val result = LocalRecorder.instance?.stop()
        recResult = result
        recState = RecState.CONFIRM_SAVE
        // 录制结束震动提醒（双脉冲）
        try {
            val vibrator = context.getSystemService(android.os.Vibrator::class.java)
            vibrator?.vibrate(
                android.os.VibrationEffect.createWaveform(longArrayOf(0, 200, 100, 200), -1)
            )
        } catch (e: Exception) {
            appendLog("震动提醒失败: ${e.message}")
        }
        if (result != null) {
            appendLog("录制结束: label=${result.label}, ${result.numSamples} samples (${String.format("%.1f", result.durationS)}s)")
        }
    }

    // 点击 label 按钮开始本地录制
    val doStartRecording: (Int) -> Unit = { label ->
        // 录制与 TCP 互斥：自动断开已连接的 TCP
        TcpAudioClient.instance?.takeIf { it.isConnected() }?.let {
            it.disconnect()
            tcpClient = null
            tcpStatus = "未连接"
            isTcpConnecting = false
            appendLog("TCP 已自动断开（开始本地录制）")
        }
        val dir = context.getExternalFilesDir("recordings")
        if (dir == null) {
            appendLog("存储目录不可用，无法录制")
        } else {
            val recorder = LocalRecorder(dir, onAutoStop = stopRecordingFlow)
            val rate = audioManager?.actualSampleRate ?: actualRate
            if (recorder.start(label, rate)) {
                recLabel = label
                recElapsedSec = 0
                recState = RecState.RECORDING
                statusText = "录制中 · Label $label"
                appendLog("本地录制开始: label=$label, 最长 ${LocalRecorder.MAX_DURATION_S}s")
                startAudioPipeline()
            } else {
                appendLog("录制启动失败（上一会话未结束）")
            }
        }
    }

    // 录制中每秒刷新倒计时显示（实际停止由样本计数触发，见 LocalRecorder.feed）
    LaunchedEffect(recState) {
        if (recState == RecState.RECORDING) {
            recElapsedSec = 0
            while (recElapsedSec < LocalRecorder.MAX_DURATION_S) {
                delay(1000)
                recElapsedSec += 1
            }
        }
    }

    // Permission launcher
    val permissionLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        if (granted) {
            if (pendingRecordLabel >= 0) {
                // 权限是为开始录制而申请的
                val label = pendingRecordLabel
                pendingRecordLabel = -1
                doStartRecording(label)
            } else {
                startAudioPipeline()
            }
        } else {
            pendingRecordLabel = -1
            statusText = "需要录音权限"
            Toast.makeText(context, "需要录音权限才能使用", Toast.LENGTH_LONG).show()
        }
    }

    // label 按钮点击入口：检查录音权限
    val onLabelClick: (Int) -> Unit = { label ->
        val hasPermission = ContextCompat.checkSelfPermission(
            context, Manifest.permission.RECORD_AUDIO
        ) == PackageManager.PERMISSION_GRANTED
        if (hasPermission) {
            doStartRecording(label)
        } else {
            pendingRecordLabel = label
            permissionLauncher.launch(Manifest.permission.RECORD_AUDIO)
        }
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(MaterialTheme.colorScheme.background)
            .verticalScroll(rememberScrollState())
            .padding(16.dp),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Text(
            text = "WatchHand 回声轮廓可视化",
            style = MaterialTheme.typography.headlineSmall,
            modifier = Modifier.padding(bottom = 8.dp)
        )

        Text(
            text = statusText,
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            modifier = Modifier.padding(bottom = 16.dp)
        )

        Button(
            onClick = {
                if (isRecording) {
                    stopAudioPipeline()
                } else {
                    val hasPermission = ContextCompat.checkSelfPermission(
                        context, Manifest.permission.RECORD_AUDIO
                    ) == PackageManager.PERMISSION_GRANTED

                    if (hasPermission) {
                        startAudioPipeline()
                    } else {
                        permissionLauncher.launch(Manifest.permission.RECORD_AUDIO)
                    }
                }
            },
            enabled = recState == RecState.IDLE,  // 本地录制期间与可视化采集互斥
            colors = ButtonDefaults.buttonColors(
                containerColor = if (isRecording) MaterialTheme.colorScheme.error
                else MaterialTheme.colorScheme.primary
            ),
            modifier = Modifier.fillMaxWidth().height(48.dp)
        ) {
            Text(if (isRecording) "停止采集" else "开始采集")
        }

        Spacer(modifier = Modifier.height(16.dp))

        // 训练数据本地录制卡片
        Card(
            modifier = Modifier.fillMaxWidth(),
            colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant)
        ) {
            Column(modifier = Modifier.padding(12.dp)) {
                Text("训练数据录制", style = MaterialTheme.typography.titleSmall)
                Spacer(modifier = Modifier.height(8.dp))
                if (recState == RecState.RECORDING) {
                    Text(
                        text = "录制中 · Label $recLabel",
                        style = MaterialTheme.typography.titleMedium
                    )
                    Text(
                        text = "${String.format("%d:%02d", recElapsedSec / 60, recElapsedSec % 60)} / ${LocalRecorder.MAX_DURATION_S / 60}:00",
                        style = MaterialTheme.typography.bodyMedium,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                    Spacer(modifier = Modifier.height(8.dp))
                    Button(
                        onClick = stopRecordingFlow,
                        colors = ButtonDefaults.buttonColors(containerColor = MaterialTheme.colorScheme.error),
                        modifier = Modifier.fillMaxWidth().height(48.dp)
                    ) {
                        Text("停止录制")
                    }
                } else {
                    Text(
                        text = "点击标签开始录制（最长 ${LocalRecorder.MAX_DURATION_S / 60} 分钟）",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                    Spacer(modifier = Modifier.height(8.dp))
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.spacedBy(8.dp)
                    ) {
                        for (label in 0..2) {
                            Button(
                                onClick = { onLabelClick(label) },
                                enabled = recState == RecState.IDLE,
                                modifier = Modifier.weight(1f).height(56.dp)
                            ) {
                                Text("$label", style = MaterialTheme.typography.titleLarge)
                            }
                        }
                    }
                }
            }
        }

        Spacer(modifier = Modifier.height(16.dp))

        // TCP Streaming Configuration Card
        Card(
            modifier = Modifier.fillMaxWidth(),
            colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant)
        ) {
            Column(modifier = Modifier.padding(12.dp)) {
                Text("TCP 数据流", style = MaterialTheme.typography.titleSmall)
                Spacer(modifier = Modifier.height(8.dp))
                Text("状态: $tcpStatus", style = MaterialTheme.typography.bodySmall)
                Spacer(modifier = Modifier.height(8.dp))
                OutlinedTextField(
                    value = serverHost,
                    onValueChange = { serverHost = it },
                    label = { Text("服务器地址") },
                    modifier = Modifier.fillMaxWidth(),
                    enabled = !isRecording && tcpClient?.isConnected() != true,
                    singleLine = true
                )
                Spacer(modifier = Modifier.height(8.dp))
                OutlinedTextField(
                    value = serverPort.toString(),
                    onValueChange = { serverPort = it.toIntOrNull() ?: 9999 },
                    label = { Text("端口") },
                    modifier = Modifier.fillMaxWidth(),
                    enabled = !isRecording && tcpClient?.isConnected() != true,
                    singleLine = true,
                    keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number)
                )
                Spacer(modifier = Modifier.height(12.dp))
                Button(
                    onClick = onTcpConnectClick,
                    modifier = Modifier.fillMaxWidth().height(40.dp),
                    enabled = !isTcpConnecting && recState == RecState.IDLE,  // 本地录制期间禁用 TCP
                    colors = ButtonDefaults.buttonColors(
                        containerColor = if (tcpClient?.isConnected() == true)
                            MaterialTheme.colorScheme.error
                        else MaterialTheme.colorScheme.primary
                    )
                ) {
                    Text(
                        if (isTcpConnecting) "连接中..."
                        else if (tcpClient?.isConnected() == true) "断开连接"
                        else "连接"
                    )
                }
            }
        }

        Spacer(modifier = Modifier.height(16.dp))

        Card(
            modifier = Modifier.fillMaxWidth(),
            colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant)
        ) {
            Column(modifier = Modifier.padding(12.dp)) {
                Text("参数信息", style = MaterialTheme.typography.titleSmall)
                Spacer(modifier = Modifier.height(4.dp))
                val distResMm = 343.0 / (2.0 * actualRate) * 1000.0
                val distRangeCm = AudioManager.DISTANCE_BINS * distResMm / 10.0
                val chirpLen = AudioManager.CHIRP_LENGTH
                Text("采样率: ${actualRate / 1000}.${actualRate % 1000 / 100} kHz", style = MaterialTheme.typography.bodySmall)
                Text("FMCW 频段: ${AudioManager.F_MIN.toInt() / 1000}-${AudioManager.F_MAX.toInt() / 1000} kHz", style = MaterialTheme.typography.bodySmall)
                Text("Chirp 长度: $chirpLen samples (${String.format("%.1f", chirpLen.toDouble() / actualRate * 1000)}ms)", style = MaterialTheme.typography.bodySmall)
                Text("距离分辨率: ${String.format("%.2f", distResMm)} mm/pixel", style = MaterialTheme.typography.bodySmall)
                Text("距离范围: ${String.format("%.1f", distRangeCm)} cm", style = MaterialTheme.typography.bodySmall)
                Text("时间窗口: ${AudioManager.TIME_WINDOW_FRAMES} frames (${String.format("%.1f", AudioManager.TIME_WINDOW_FRAMES * chirpLen.toDouble() / actualRate * 1000)}ms)", style = MaterialTheme.typography.bodySmall)
                if (frameCount > 0) {
                    Text("当前帧数: $frameCount", style = MaterialTheme.typography.bodySmall)
                    Text("可视化范围 (2%-98%): $visRange", style = MaterialTheme.typography.bodySmall)
                }
            }
        }

        Spacer(modifier = Modifier.height(16.dp))

        Text("原始回声轮廓 (Original Echo Profile)", style = MaterialTheme.typography.titleSmall)
        EchoProfileHeatmap(
            profile = originalProfile,
            distBins = distBins,
            frameCount = frameCount,
            modifier = Modifier.fillMaxWidth().height(200.dp)
        )

        Spacer(modifier = Modifier.height(16.dp))

        Text("差分回声轮廓 (Differential Echo Profile)", style = MaterialTheme.typography.titleSmall)
        EchoProfileHeatmap(
            profile = differentialProfile,
            distBins = distBins,
            frameCount = maxOf(0, frameCount - 1),
            isDifferential = true,
            modifier = Modifier.fillMaxWidth().height(200.dp)
        )

        Spacer(modifier = Modifier.height(32.dp))

        // Debug Log Card
        Card(
            modifier = Modifier.fillMaxWidth(),
            colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant)
        ) {
            Column(modifier = Modifier.padding(12.dp)) {
                Text("调试日志", style = MaterialTheme.typography.titleSmall)
                Spacer(modifier = Modifier.height(4.dp))
                Text(
                    text = debugLog.ifEmpty { "无日志" },
                    style = MaterialTheme.typography.bodySmall,
                    fontFamily = androidx.compose.ui.text.font.FontFamily.Monospace,
                    modifier = Modifier.height(150.dp)
                )
            }
        }

        Spacer(modifier = Modifier.height(16.dp))
    }

    // 录制结束：是否保存确认弹窗
    if (recState == RecState.CONFIRM_SAVE) {
        val result = recResult
        AlertDialog(
            onDismissRequest = {},  // 必须显式选择保存或丢弃
            title = { Text("保存本次录制？") },
            text = {
                if (result != null) {
                    Text("Label ${result.label} · 时长 ${String.format("%.1f", result.durationS)} s")
                }
            },
            confirmButton = {
                TextButton(onClick = {
                    val name = LocalRecorder.instance?.save()
                    if (name != null) {
                        appendLog("已保存: $name")
                        Toast.makeText(context, "已保存 $name", Toast.LENGTH_LONG).show()
                    } else {
                        appendLog("保存失败")
                        Toast.makeText(context, "保存失败", Toast.LENGTH_LONG).show()
                    }
                    stopAudioPipeline()
                    recState = RecState.IDLE
                    recResult = null
                }) { Text("保存") }
            },
            dismissButton = {
                TextButton(onClick = {
                    LocalRecorder.instance?.discard()
                    appendLog("已丢弃录制数据")
                    stopAudioPipeline()
                    recState = RecState.IDLE
                    recResult = null
                }) { Text("丢弃") }
            }
        )
    }
}

@Composable
fun EchoProfileHeatmap(
    profile: FloatArray?,
    distBins: Int,
    frameCount: Int,
    isDifferential: Boolean = false,
    modifier: Modifier = Modifier
) {
    androidx.compose.foundation.Canvas(modifier = modifier) {
        // Draw blue background (matches paper's blue-based colormap)
        drawRect(echoProfileColor(0f), size = size)

        if (profile == null || frameCount == 0) {
            return@Canvas
        }

        val cellWidth = size.width / frameCount
        val cellHeight = size.height / distBins

        // Compute normalization parameters
        // Original profile: percentile clipping (2%/98%) for adaptive range
        // Differential profile: symmetric clipping around 0 to preserve zero-centering
        val lo: Float
        val hi: Float
        if (isDifferential) {
            val absArray = FloatArray(profile.size) { kotlin.math.abs(profile[it]) }
            val absSorted = absArray.sortedArray()
            val maxAbs = absSorted[(absSorted.size * 0.98).toInt()]
            lo = -maxAbs
            hi = maxAbs
        } else {
            val sorted = profile.sortedArray()
            lo = sorted[(sorted.size * 0.02).toInt()]
            hi = sorted[(sorted.size * 0.98).toInt()]
        }
        val range = hi - lo

        if (range < 1e-10f) {
            drawRect(echoProfileColor(0.5f), size = size)
            return@Canvas
        }

        // Min-max normalize with clipping
        for (f in 0 until frameCount) {
            for (d in 0 until distBins) {
                val rawValue = profile[d * frameCount + f]
                val clipped = rawValue.coerceIn(lo, hi)
                val normalized = (clipped - lo) / range

                drawRect(
                    color = echoProfileColor(normalized),
                    topLeft = androidx.compose.ui.geometry.Offset(f * cellWidth, d * cellHeight),
                    size = androidx.compose.ui.geometry.Size(cellWidth + 0.5f, cellHeight + 0.5f)
                )
            }
        }
    }
}

/**
 * Blue-based sequential colormap matching the paper's echo-profile visualization.
 * Blue → Cyan → Magenta → Yellow
 * Both original and differential profiles use this same colormap.
 */
private fun echoProfileColor(t: Float): androidx.compose.ui.graphics.Color {
    val clamped = t.coerceIn(0f, 1f)
    return when {
        clamped < 0.333f -> {
            // Blue → Cyan
            val s = clamped / 0.333f
            androidx.compose.ui.graphics.Color(0f, 0.7f * s, 0.8f + 0.2f * s)
        }
        clamped < 0.667f -> {
            // Cyan → Magenta
            val s = (clamped - 0.333f) / 0.334f
            androidx.compose.ui.graphics.Color(0.7f * s, 0.7f * (1f - s), 1f - 0.2f * s)
        }
        else -> {
            // Magenta → Yellow
            val s = (clamped - 0.667f) / 0.333f
            androidx.compose.ui.graphics.Color(0.7f + 0.3f * s, 0.7f * s, 0.8f * (1f - s))
        }
    }
}
