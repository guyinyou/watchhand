package com.watchhand.app

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.AudioTrack
import android.media.MediaRecorder
import android.util.Log
import androidx.core.app.ActivityCompat
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.launch
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit

/**
 * Manages simultaneous FMCW audio playback and microphone recording,
 * with real-time echo profile processing.
 */
class AudioManager(
    private val context: Context,
    private val sampleRate: Int = 44100,
    private val onEchoProfileUpdate: ((original: FloatArray, differential: FloatArray, distBins: Int, frameCount: Int) -> Unit)? = null,
    private val onStatusUpdate: ((String) -> Unit)? = null,
    private val onSampleRateUpdate: ((Int) -> Unit)? = null  // Dedicated callback for actual sample rate
) {
    companion object {
        private const val TAG = "WatchHand-Audio"
        const val F_MIN = 18000.0
        const val F_MAX = 20000.0  // 18-20kHz：44.1kHz 硬件假设下的统一频段（保证 Nyquist 余量）
        const val DISTANCE_BINS = 60
        const val TIME_WINDOW_FRAMES = 96

        /** Chirp 长度：588 samples = 13.333ms @ 统一 44.1kHz，帧率 75fps（服务端 WatchHandServer.CHIRP_LENGTH 保持同步） */
        const val CHIRP_LENGTH = 588
    }

    /** 统一采样率：所有设备一律请求 44100Hz。
     * 手表原生 44.1kHz；48kHz 手机经系统 SRC，已实验验证可用（2026-08-19，不闪）。
     * 采样率统一后 chirp/帧率/距离网格跨设备天然一致，混合训练零额外处理。 */
    private fun targetSampleRate(): Int = 44100

    private var audioTrack: AudioTrack? = null
    private var audioRecord: AudioRecord? = null
    private var scope: CoroutineScope? = null
    private var isRunning = false

    // 录音实际采样率协调：播放 chirp 必须按录音实际采样率生成，
    // 否则 tx chirp 与 rx 采样率不匹配，互相关结果完全错误
    @Volatile
    private var recordRateDetermined = 0
    private var recordRateLatch = CountDownLatch(1)

    // Processor created lazily after we know the actual recording sample rate
    private var processor: EchoProfileProcessor? = null

    /** Actual sample rate used (may differ from requested) */
    var actualSampleRate: Int = sampleRate
        private set

    /**
     * Start FMCW playback and recording.
     */
    fun start() {
        if (isRunning) return
        isRunning = true

        scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)

        // 统一采样率 44.1kHz：跨设备参数完全一致（手机经 SRC，已验证可用）
        val nativeRate = targetSampleRate()
        onStatusUpdate?.invoke("统一采样率: ${nativeRate}Hz")

        // 重置采样率协调状态（支持重复 start/stop）
        recordRateDetermined = 0
        recordRateLatch = CountDownLatch(1)

        // Start recording + processing (先启动，确定实际采样率)
        scope?.launch {
            recordAndProcess(nativeRate)
        }

        // Start playback (等待录音确定实际采样率后再生成 chirp)
        scope?.launch {
            playFmcw(nativeRate)
        }

        onStatusUpdate?.invoke("采集已启动")
        Log.i(TAG, "AudioManager started")
    }

    /**
     * Stop playback and recording.
     */
    fun stop() {
        isRunning = false
        try {
            audioTrack?.let {
                if (it.playState == AudioTrack.PLAYSTATE_PLAYING) it.stop()
                it.release()
            }
        } catch (e: Exception) {
            Log.w(TAG, "Error stopping AudioTrack: ${e.message}")
        }
        audioTrack = null

        try {
            audioRecord?.let {
                if (it.recordingState == AudioRecord.RECORDSTATE_RECORDING) it.stop()
                it.release()
            }
        } catch (e: Exception) {
            Log.w(TAG, "Error stopping AudioRecord: ${e.message}")
        }
        audioRecord = null

        try {
            scope?.cancel()
        } catch (e: Exception) {
            Log.w(TAG, "Error cancelling scope: ${e.message}")
        }
        scope = null
        onStatusUpdate?.invoke("采集已停止")
        Log.i(TAG, "AudioManager stopped")
    }

    /**
     * Check if currently running.
     */
    fun isRunning(): Boolean = isRunning

    private fun playFmcw(nativeRate: Int) {
        // 等待录音线程确定实际采样率（播放 chirp 必须与录音采样率一致）
        if (!recordRateLatch.await(3, TimeUnit.SECONDS)) {
            Log.w(TAG, "Timeout waiting for record rate, falling back to native rate")
        }
        val targetRate = if (recordRateDetermined > 0) recordRateDetermined else nativeRate
        Log.i(TAG, "Playback target rate: ${targetRate}Hz (record-determined)")

        // 请求与录音一致的采样率，避免系统重采样（会把 18-20kHz 移入可听范围）
        val candidateRates = (listOf(targetRate, nativeRate, sampleRate)).distinct()
        var track: AudioTrack? = null
        var usedRate = targetRate

        for (rate in candidateRates) {
            val minBuf = AudioTrack.getMinBufferSize(
                rate, AudioFormat.CHANNEL_OUT_MONO, AudioFormat.ENCODING_PCM_16BIT
            )
            if (minBuf <= 0) {
                Log.w(TAG, "Sample rate ${rate}Hz not supported for playback")
                continue
            }
            try {
                track = AudioTrack.Builder()
                    .setAudioAttributes(
                        AudioAttributes.Builder()
                            .setUsage(AudioAttributes.USAGE_MEDIA)
                            .setContentType(AudioAttributes.CONTENT_TYPE_SONIFICATION)
                            .build()
                    )
                    .setAudioFormat(
                        AudioFormat.Builder()
                            .setSampleRate(rate)
                            .setChannelMask(AudioFormat.CHANNEL_OUT_MONO)
                            .setEncoding(AudioFormat.ENCODING_PCM_16BIT)
                            .build()
                    )
                    .setBufferSizeInBytes(minBuf * 4)
                    .setTransferMode(AudioTrack.MODE_STREAM)
                    .build()
                // Use the track's actual sample rate (may differ from requested)
                usedRate = track.sampleRate
                Log.i(TAG, "AudioTrack created: requested=${rate}Hz actual=${usedRate}Hz (minBuf=$minBuf)")
                break
            } catch (e: UnsupportedOperationException) {
                Log.w(TAG, "AudioTrack build failed at ${rate}Hz: ${e.message}")
            }
        }

        if (track == null) {
            onStatusUpdate?.invoke("播放初始化失败：设备不支持任何采样率")
            Log.e(TAG, "Cannot create AudioTrack at any sample rate")
            return
        }

        audioTrack = track
        actualSampleRate = usedRate

        // Always regenerate chirp at the actual playback sample rate to avoid resampling artifacts
        val playbackGen = FmcwGenerator(usedRate, F_MIN, F_MAX, CHIRP_LENGTH)
        val playbackBuffer = playbackGen.generatePlaybackBuffer(30.0)
        Log.i(TAG, "Playback chirp: ${usedRate}Hz, ${playbackBuffer.size} samples, freq ${F_MIN}-${F_MAX}Hz")

        track.play()

        // Write buffer in chunks (100ms, 按实际采样率动态计算)
        val chunkSize = usedRate / 100
        var offset = 0
        while (isRunning && offset < playbackBuffer.size) {
            val remaining = playbackBuffer.size - offset
            val toWrite = minOf(chunkSize, remaining)
            val written = track.write(playbackBuffer, offset, toWrite)
            if (written < 0) {
                Log.e(TAG, "AudioTrack write error: $written")
                break
            }
            offset += written
        }

        // Loop playback
        while (isRunning) {
            offset = 0
            while (isRunning && offset < playbackBuffer.size) {
                val remaining = playbackBuffer.size - offset
                val toWrite = minOf(chunkSize, remaining)
                val written = track.write(playbackBuffer, offset, toWrite)
                if (written < 0) break
                offset += written
            }
        }
    }

    private suspend fun recordAndProcess(nativeRate: Int) {
        // 请求设备原生采样率并与播放保持一致，否则互相关结果错误
        val candidateRates = (listOf(nativeRate, sampleRate)).distinct()
        var record: AudioRecord? = null
        var actualRate = nativeRate

        for (rate in candidateRates) {
            val minBuf = AudioRecord.getMinBufferSize(
                rate, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT
            )
            onStatusUpdate?.invoke("尝试录音采样率: ${rate}Hz, minBuf=$minBuf")
            if (minBuf <= 0) {
                onStatusUpdate?.invoke("  -> ${rate}Hz 不支持")
                continue
            }
            val rec = AudioRecord.Builder()
                .setAudioSource(MediaRecorder.AudioSource.UNPROCESSED)
                .setAudioFormat(
                    AudioFormat.Builder()
                        .setSampleRate(rate)
                        .setChannelMask(AudioFormat.CHANNEL_IN_MONO)
                        .setEncoding(AudioFormat.ENCODING_PCM_16BIT)
                        .build()
                )
                .setBufferSizeInBytes(maxOf(minBuf * 4, rate * 2 * 2))
                .build()

            if (rec.state == AudioRecord.STATE_INITIALIZED) {
                record = rec
                actualRate = rec.sampleRate
                onStatusUpdate?.invoke("  -> UNPROCESSED 成功! requested=${rate}Hz actual=${actualRate}Hz")
                break
            } else {
                rec.release()
                onStatusUpdate?.invoke("  -> UNPROCESSED 初始化失败 at ${rate}Hz")
            }
        }

        // Fallback to DEFAULT audio source
        if (record == null) {
            onStatusUpdate?.invoke("UNPROCESSED 全部失败，尝试 DEFAULT 音频源")
            for (rate in candidateRates) {
                val minBuf = AudioRecord.getMinBufferSize(
                    rate, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT
                )
                if (minBuf <= 0) {
                    onStatusUpdate?.invoke("  -> ${rate}Hz 不支持")
                    continue
                }
                val rec = AudioRecord.Builder()
                    .setAudioSource(MediaRecorder.AudioSource.DEFAULT)
                    .setAudioFormat(
                        AudioFormat.Builder()
                            .setSampleRate(rate)
                            .setChannelMask(AudioFormat.CHANNEL_IN_MONO)
                            .setEncoding(AudioFormat.ENCODING_PCM_16BIT)
                            .build()
                    )
                    .setBufferSizeInBytes(maxOf(minBuf * 4, rate * 2 * 2))
                    .build()
                if (rec.state == AudioRecord.STATE_INITIALIZED) {
                    record = rec
                    actualRate = rec.sampleRate
                    onStatusUpdate?.invoke("  -> DEFAULT 成功! requested=${rate}Hz actual=${actualRate}Hz")
                    break
                } else {
                    rec.release()
                    onStatusUpdate?.invoke("  -> DEFAULT 初始化失败 at ${rate}Hz")
                }
            }
        }

        if (record == null) {
            onStatusUpdate?.invoke("录音初始化失败")
            Log.e(TAG, "AudioRecord initialization failed at all rates")
            return
        }

        audioRecord = record
        actualSampleRate = actualRate
        // 通知播放线程录音实际采样率（播放 chirp 按此生成）
        recordRateDetermined = actualRate
        recordRateLatch.countDown()
        onSampleRateUpdate?.invoke(actualRate)  // Notify UI of actual sample rate

        // Create processor with actual recording sample rate
        processor = EchoProfileProcessor(
            sampleRate = actualRate,
            fMin = F_MIN,
            fMax = F_MAX,
            chirpLength = CHIRP_LENGTH,
            distanceBins = DISTANCE_BINS,
            timeWindowFrames = TIME_WINDOW_FRAMES
        )
        Log.i(TAG, "EchoProfileProcessor created at ${actualRate}Hz, distanceRes=${String.format("%.2f", 343.0 / (2.0 * actualRate) * 1000.0)}mm/pixel")

        record.startRecording()
        Log.i(TAG, "Recording started at ${actualRate}Hz")
        onStatusUpdate?.invoke("录音中... (${actualRate}Hz)")

        // 读取/处理分离：读取循环只负责取数入队，FFT 等重计算在独立协程做。
        // 计算或 GC 停顿不会阻塞读取，避免 AudioRecord 溢出丢样本
        // （丢样本 = tx/rx 相位跳变 = 热力图突然变竖条纹的根因）
        val readBuffer = ShortArray(actualRate / 100) // 10ms read buffer, 按实际采样率动态计算
        val pendingSamples = kotlinx.coroutines.channels.Channel<ShortArray>(600) // ~6s 缓冲垫

        scope?.launch {
            for (samples in pendingSamples) {
                val result = processor?.feed(samples)
                if (result != null) {
                    val (original, differential, nFrames) = result
                    onEchoProfileUpdate?.invoke(original, differential, DISTANCE_BINS, nFrames)
                }
            }
        }

        while (isRunning) {
            val read = record.read(readBuffer, 0, readBuffer.size)
            if (read > 0) {
                val samples = readBuffer.copyOf(read)

                val recorder = LocalRecorder.instance
                if (recorder?.isRecording == true) {
                    // 本地录制优先：与 TCP 发送同一条数据通路，只写盘，跳过本地处理与 TCP
                    recorder.feed(samples)
                } else if (TcpAudioClient.instance?.isConnected() == true) {
                    // TCP connected: only send raw data, skip local processing
                    TcpAudioClient.instance?.sendAudio(samples)
                } else {
                    // No TCP connection: 入队由处理协程消费（缓冲满时背压等待，不丢样本）
                    pendingSamples.send(samples)
                }
            } else if (read < 0) {
                Log.e(TAG, "AudioRecord read error: $read")
                break
            }
        }
        pendingSamples.close()

        try {
            if (record.recordingState == AudioRecord.RECORDSTATE_RECORDING) {
                record.stop()
            }
        } catch (e: Exception) {
            Log.w(TAG, "Error stopping record: ${e.message}")
        }
    }

    private fun shortToBytes(s: Short): ByteArray {
        return byteArrayOf(
            (s.toInt() and 0xFF).toByte(),
            ((s.toInt() shr 8) and 0xFF).toByte()
        )
    }

    private fun bytesToShortArray(bytes: ByteArray): ShortArray {
        val samples = ShortArray(bytes.size / 2)
        for (i in samples.indices) {
            samples[i] = (bytes[i * 2].toInt() and 0xFF or (bytes[i * 2 + 1].toInt() shl 8)).toShort()
        }
        return samples
    }
}
