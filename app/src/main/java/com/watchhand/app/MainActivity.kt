package com.watchhand.app

import android.Manifest
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
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat

class MainActivity : ComponentActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
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
    var isRecording by remember { mutableStateOf(false) }
    var statusText by remember { mutableStateOf("等待启动") }
    var originalProfile by remember { mutableStateOf<FloatArray?>(null) }
    var differentialProfile by remember { mutableStateOf<FloatArray?>(null) }
    var distBins by remember { mutableIntStateOf(AudioManager.DISTANCE_BINS) }
    var frameCount by remember { mutableIntStateOf(0) }
    var audioManager by remember { mutableStateOf<AudioManager?>(null) }

    // Percentile clipping range for visualization
    var visRange by remember { mutableStateOf("-") }

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

    val onStatus: (String) -> Unit = { statusText = it }

    // Permission launcher
    val permissionLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        if (granted) {
            isRecording = true
            statusText = "正在初始化..."
            val manager = AudioManager(onEchoProfileUpdate = onProfileUpdate, onStatusUpdate = onStatus)
            audioManager = manager
            manager.start()
        } else {
            statusText = "需要录音权限"
            Toast.makeText(context, "需要录音权限才能使用", Toast.LENGTH_LONG).show()
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
                    audioManager?.stop()
                    audioManager = null
                    isRecording = false
                    statusText = "已停止"
                } else {
                    val hasPermission = ContextCompat.checkSelfPermission(
                        context, Manifest.permission.RECORD_AUDIO
                    ) == PackageManager.PERMISSION_GRANTED

                    if (hasPermission) {
                        isRecording = true
                        statusText = "正在初始化..."
                        val manager = AudioManager(onEchoProfileUpdate = onProfileUpdate, onStatusUpdate = onStatus)
                        audioManager = manager
                        manager.start()
                    } else {
                        permissionLauncher.launch(Manifest.permission.RECORD_AUDIO)
                    }
                }
            },
            colors = ButtonDefaults.buttonColors(
                containerColor = if (isRecording) MaterialTheme.colorScheme.error
                else MaterialTheme.colorScheme.primary
            ),
            modifier = Modifier.fillMaxWidth().height(48.dp)
        ) {
            Text(if (isRecording) "停止采集" else "开始采集")
        }

        Spacer(modifier = Modifier.height(16.dp))

        Card(
            modifier = Modifier.fillMaxWidth(),
            colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant)
        ) {
            Column(modifier = Modifier.padding(12.dp)) {
                Text("参数信息", style = MaterialTheme.typography.titleSmall)
                Spacer(modifier = Modifier.height(4.dp))
                Text("采样率: 48 kHz", style = MaterialTheme.typography.bodySmall)
                Text("FMCW 频段: 18-21 kHz", style = MaterialTheme.typography.bodySmall)
                Text("Chirp 长度: 600 samples (12.5ms)", style = MaterialTheme.typography.bodySmall)
                Text("距离分辨率: 3.57 mm/pixel", style = MaterialTheme.typography.bodySmall)
                Text("距离范围: ${AudioManager.DISTANCE_BINS * 3.57 / 10} cm", style = MaterialTheme.typography.bodySmall)
                Text("时间窗口: ${AudioManager.TIME_WINDOW_FRAMES} frames (1.2s)", style = MaterialTheme.typography.bodySmall)
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
        // Draw viridis low-end color as background (dark purple-blue, matches paper)
        drawRect(viridisColor(0f), size = size)

        if (profile == null || frameCount == 0) {
            return@Canvas
        }

        val cellWidth = size.width / frameCount
        val cellHeight = size.height / distBins

        // Percentile-based clipping for visualization (adapts to different hardware)
        // Paper uses fixed ±10^10, but that's tuned for watch hardware.
        // For visualization across devices, use 2nd/98th percentile.
        val sorted = profile.sortedArray()
        val lo = sorted[(sorted.size * 0.02).toInt()]
        val hi = sorted[(sorted.size * 0.98).toInt()]
        val range = hi - lo

        if (range < 1e-10f) {
            drawRect(viridisColor(0.5f), size = size)
            return@Canvas
        }

        // Min-max normalize with percentile clipping
        for (f in 0 until frameCount) {
            for (d in 0 until distBins) {
                val rawValue = profile[d * frameCount + f]
                val clipped = rawValue.coerceIn(lo, hi)
                val normalized = (clipped - lo) / range

                val color = if (isDifferential) {
                    divergingColor(normalized)
                } else {
                    viridisColor(normalized)
                }

                drawRect(
                    color = color,
                    topLeft = androidx.compose.ui.geometry.Offset(f * cellWidth, d * cellHeight),
                    size = androidx.compose.ui.geometry.Size(cellWidth + 0.5f, cellHeight + 0.5f)
                )
            }
        }
    }
}

private fun viridisColor(t: Float): androidx.compose.ui.graphics.Color {
    val clamped = t.coerceIn(0f, 1f)
    val r = (0.267f + clamped * (0.993f - 0.267f)).coerceIn(0f, 1f)
    val g = (0.004f + clamped * (0.906f - 0.004f)).coerceIn(0f, 1f)
    val b = (0.329f + (1f - clamped) * (0.741f - 0.329f)).coerceIn(0f, 1f)
    return androidx.compose.ui.graphics.Color(r, g, b)
}

private fun divergingColor(t: Float): androidx.compose.ui.graphics.Color {
    val clamped = t.coerceIn(0f, 1f)
    return if (clamped < 0.5f) {
        val s = clamped * 2f
        androidx.compose.ui.graphics.Color(0.1f * (1f - s), 0.1f * (1f - s), 0.8f * (1f - s) + 0.1f)
    } else {
        val s = (clamped - 0.5f) * 2f
        androidx.compose.ui.graphics.Color(0.9f * s + 0.05f, 0.9f * s + 0.05f, 0.1f * (1f - s))
    }
}
