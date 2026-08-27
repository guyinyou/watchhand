package com.watchhand.app

import android.os.Handler
import android.os.Looper
import android.util.Log
import java.io.BufferedOutputStream
import java.io.File
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/**
 * 本地训练数据录制器：与 TCP 发送同一条数据通路（原始 PCM 直接写盘，不做本地可视化处理）。
 * 产物 .raw/.labels/.meta 三件套与服务端 WatchHandServer 采集格式完全一致，
 * adb pull 后可直接被 extract.py 消费。
 */
class LocalRecorder(
    private val saveDir: File,
    private val onAutoStop: (() -> Unit)? = null
) {
    companion object {
        private const val TAG = "WatchHand-Recorder"

        /** 单次录制最长时间（秒） */
        const val MAX_DURATION_S = 120

        /** meta 中 time_window_frames 字段：与当前模型/服务端会话保持一致 */
        private const val META_TIME_WINDOW_FRAMES = 16

        /** 全局单例：录音线程可直接判空访问，避免 Compose 状态捕获问题 */
        @Volatile
        var instance: LocalRecorder? = null
            private set
    }

    /** 录制结束后的会话结果（未保存前仅存在于临时文件） */
    data class Result(
        val label: Int,
        val numSamples: Long,
        val sampleRate: Int,
        val durationS: Double
    )

    private var out: BufferedOutputStream? = null
    private var tmpFile: File? = null
    private var label = 0
    private var sampleRate = 44100
    private var maxSamples = MAX_DURATION_S.toLong() * sampleRate
    private var sampleCount = 0L
    private var startedAt = 0L
    private var autoStopTriggered = false
    private val mainHandler = Handler(Looper.getMainLooper())

    /** 是否正在录制（录音线程分流判断用） */
    val isRecording: Boolean
        get() = out != null

    /**
     * 开始录制：创建临时 raw 文件，等待 feed() 写入。
     * @param label 本次录制的标签（单标签会话）
     * @param sampleRate 初始采样率（若 AudioManager 随后上报实际采样率会经 updateSampleRate 修正）
     */
    fun start(label: Int, sampleRate: Int): Boolean {
        if (isRecording) return false
        try {
            if (!saveDir.exists()) saveDir.mkdirs()
            val tmp = File(saveDir, "recording_tmp.raw")
            if (tmp.exists()) tmp.delete()
            out = BufferedOutputStream(FileOutputStream(tmp), 256 * 1024)
            tmpFile = tmp
            this.label = label
            this.sampleRate = sampleRate
            this.maxSamples = MAX_DURATION_S.toLong() * sampleRate
            this.sampleCount = 0
            this.startedAt = System.currentTimeMillis()
            this.autoStopTriggered = false
            instance = this
            Log.i(TAG, "Recording started: label=$label, rate=${sampleRate}Hz, max=${MAX_DURATION_S}s")
            return true
        } catch (e: Exception) {
            Log.e(TAG, "Failed to start recording: ${e.message}")
            out = null
            tmpFile = null
            return false
        }
    }

    /**
     * 录音实际采样率上报时修正（仅在尚未写入任何样本时生效，
     * 保证 maxSamples 上限与 meta 中 sample_rate 严格一致）。
     */
    fun updateSampleRate(rate: Int) {
        if (sampleCount == 0L && rate > 0 && rate != sampleRate) {
            sampleRate = rate
            maxSamples = MAX_DURATION_S.toLong() * rate
            Log.i(TAG, "Sample rate corrected to ${rate}Hz before first sample")
        }
    }

    /**
     * 写入 PCM 样本（16-bit 小端）。达到最长时间后停止写入并回调 onAutoStop（主线程）。
     */
    fun feed(samples: ShortArray) {
        val o = out ?: return
        if (sampleCount >= maxSamples) return
        var toWrite = samples.size
        if (sampleCount + toWrite > maxSamples) {
            toWrite = (maxSamples - sampleCount).toInt()
        }
        try {
            val buf = ByteBuffer.allocate(toWrite * 2).order(ByteOrder.LITTLE_ENDIAN)
            for (i in 0 until toWrite) buf.putShort(samples[i])
            o.write(buf.array())
            sampleCount += toWrite
        } catch (e: Exception) {
            Log.e(TAG, "Write failed: ${e.message}")
        }
        if (sampleCount >= maxSamples && !autoStopTriggered) {
            autoStopTriggered = true
            Log.i(TAG, "Max duration reached (${MAX_DURATION_S}s), auto stop")
            mainHandler.post { onAutoStop?.invoke() }
        }
    }

    /**
     * 结束录制：flush 并关闭文件，返回会话结果。临时文件保留，等待 save()/discard()。
     */
    fun stop(): Result? {
        val o = out ?: return null
        try {
            o.flush()
            o.close()
        } catch (e: Exception) {
            Log.w(TAG, "Error closing raw stream: ${e.message}")
        }
        out = null
        val result = Result(label, sampleCount, sampleRate, sampleCount / sampleRate.toDouble())
        Log.i(TAG, "Recording stopped: ${result.numSamples} samples (${String.format(Locale.US, "%.1f", result.durationS)}s)")
        return result
    }

    /**
     * 保存会话：临时文件重命名为 gesture_data_<时间戳>.raw，并写出 .labels/.meta。
     * @return 保存的会话基础名（不含扩展名），失败返回 null
     */
    fun save(): String? {
        val tmp = tmpFile ?: return null
        if (out != null) stop()  // 兜底：未 stop 直接 save
        try {
            val ts = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date(startedAt))
            val baseName = "gesture_data_$ts"
            val rawFile = File(saveDir, "$baseName.raw")
            if (!tmp.renameTo(rawFile)) {
                tmp.copyTo(rawFile, overwrite = true)
                tmp.delete()
            }

            // 单标签会话：文件偏移 0 处即为本次 label（与服务端 labelEvents 格式一致）
            File(saveDir, "$baseName.labels").writeText("0 $label\n")

            // meta 字段与 WatchHandServer.stopCollection 保持一致，extract.py 直接可读
            val numSamples = sampleCount
            val duration = numSamples / sampleRate.toDouble()
            val meta = buildString {
                append("sample_rate=$sampleRate\n")
                append("channels=1\n")
                append("bits=16\n")
                append("num_raw_samples=$numSamples\n")
                append("duration_s=${String.format(Locale.US, "%.2f", duration)}\n")
                append("f_min=${AudioManager.F_MIN.toInt()}\n")
                append("f_max=${AudioManager.F_MAX.toInt()}\n")
                append("chirp_length=${AudioManager.CHIRP_LENGTH}\n")
                append("chirp_duration_ms=${String.format(Locale.US, "%.3f", AudioManager.CHIRP_LENGTH * 1000.0 / sampleRate)}\n")
                append("distance_bins=${AudioManager.DISTANCE_BINS}\n")
                append("time_window_frames=$META_TIME_WINDOW_FRAMES\n")
                append("started_at=${SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.US).format(Date(startedAt))}\n")
            }
            File(saveDir, "$baseName.meta").writeText(meta)

            Log.i(TAG, "Saved session $baseName -> ${saveDir.path}")
            instance = null
            return baseName
        } catch (e: Exception) {
            Log.e(TAG, "Save failed: ${e.message}")
            return null
        }
    }

    /**
     * 丢弃会话：删除临时文件。
     */
    fun discard() {
        if (out != null) stop()
        try {
            tmpFile?.delete()
        } catch (e: Exception) {
            Log.w(TAG, "Error deleting tmp file: ${e.message}")
        }
        tmpFile = null
        instance = null
        Log.i(TAG, "Recording discarded")
    }
}
