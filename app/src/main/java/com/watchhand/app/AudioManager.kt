package com.watchhand.app

import android.Manifest
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

/**
 * Manages simultaneous FMCW audio playback and microphone recording,
 * with real-time echo profile processing.
 */
class AudioManager(
    private val sampleRate: Int = 48000,
    private val onEchoProfileUpdate: ((original: FloatArray, differential: FloatArray, distBins: Int, frameCount: Int) -> Unit)? = null,
    private val onStatusUpdate: ((String) -> Unit)? = null
) {
    companion object {
        private const val TAG = "WatchHand-Audio"
        const val CHIRP_LENGTH = 600
        const val F_MIN = 18000.0
        const val F_MAX = 21000.0
        const val DISTANCE_BINS = 60
        const val TIME_WINDOW_FRAMES = 96
    }

    private var audioTrack: AudioTrack? = null
    private var audioRecord: AudioRecord? = null
    private var scope: CoroutineScope? = null
    private var isRunning = false

    private val fmcwGenerator = FmcwGenerator(sampleRate, F_MIN, F_MAX, CHIRP_LENGTH)
    private val processor = EchoProfileProcessor(
        sampleRate = sampleRate,
        fMin = F_MIN,
        fMax = F_MAX,
        chirpLength = CHIRP_LENGTH,
        distanceBins = DISTANCE_BINS,
        timeWindowFrames = TIME_WINDOW_FRAMES
    )

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

        // Generate playback buffer (30 seconds of FMCW chirps)
        val playbackBuffer = fmcwGenerator.generatePlaybackBuffer(30.0)
        onStatusUpdate?.invoke("FMCW 信号已生成 (${playbackBuffer.size} samples)")

        // Start playback
        scope?.launch {
            playFmcw(playbackBuffer)
        }

        // Start recording + processing
        scope?.launch {
            recordAndProcess()
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

    private fun playFmcw(buffer: ShortArray) {
        val minBufferSize = AudioTrack.getMinBufferSize(
            sampleRate,
            AudioFormat.CHANNEL_OUT_MONO,
            AudioFormat.ENCODING_PCM_16BIT
        )

        val track = AudioTrack.Builder()
            .setAudioAttributes(
                AudioAttributes.Builder()
                    .setUsage(AudioAttributes.USAGE_MEDIA)
                    .setContentType(AudioAttributes.CONTENT_TYPE_SONIFICATION)
                    .build()
            )
            .setAudioFormat(
                AudioFormat.Builder()
                    .setSampleRate(sampleRate)
                    .setChannelMask(AudioFormat.CHANNEL_OUT_MONO)
                    .setEncoding(AudioFormat.ENCODING_PCM_16BIT)
                    .build()
            )
            .setBufferSizeInBytes(maxOf(minBufferSize, buffer.size * 2))
            .setTransferMode(AudioTrack.MODE_STREAM)
            .build()

        audioTrack = track
        actualSampleRate = track.sampleRate

        track.play()

        // Write buffer in chunks
        val chunkSize = 4800 // 100ms chunks
        var offset = 0
        while (isRunning && offset < buffer.size) {
            val remaining = buffer.size - offset
            val toWrite = minOf(chunkSize, remaining)
            val written = track.write(buffer, offset, toWrite)
            if (written < 0) {
                Log.e(TAG, "AudioTrack write error: $written")
                break
            }
            offset += written
        }

        // Loop playback
        while (isRunning) {
            offset = 0
            while (isRunning && offset < buffer.size) {
                val remaining = buffer.size - offset
                val toWrite = minOf(chunkSize, remaining)
                val written = track.write(buffer, offset, toWrite)
                if (written < 0) break
                offset += written
            }
        }
    }

    private fun recordAndProcess() {
        val minBufferSize = AudioRecord.getMinBufferSize(
            sampleRate,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT
        )

        val record = AudioRecord.Builder()
            .setAudioSource(MediaRecorder.AudioSource.UNPROCESSED)
            .setAudioFormat(
                AudioFormat.Builder()
                    .setSampleRate(sampleRate)
                    .setChannelMask(AudioFormat.CHANNEL_IN_MONO)
                    .setEncoding(AudioFormat.ENCODING_PCM_16BIT)
                    .build()
            )
            .setBufferSizeInBytes(maxOf(minBufferSize * 4, sampleRate * 2 * 2)) // 2 seconds buffer
            .build()

        audioRecord = record

        if (record.state != AudioRecord.STATE_INITIALIZED) {
            onStatusUpdate?.invoke("AudioRecord 初始化失败，尝试使用 DEFAULT 音频源")
            Log.w(TAG, "UNPROCESSED source failed, trying DEFAULT")

            val fallbackRecord = AudioRecord.Builder()
                .setAudioSource(MediaRecorder.AudioSource.DEFAULT)
                .setAudioFormat(
                    AudioFormat.Builder()
                        .setSampleRate(sampleRate)
                        .setChannelMask(AudioFormat.CHANNEL_IN_MONO)
                        .setEncoding(AudioFormat.ENCODING_PCM_16BIT)
                        .build()
                )
                .setBufferSizeInBytes(maxOf(minBufferSize * 4, sampleRate * 2 * 2))
                .build()

            if (fallbackRecord.state != AudioRecord.STATE_INITIALIZED) {
                onStatusUpdate?.invoke("AudioRecord 初始化失败")
                Log.e(TAG, "AudioRecord initialization failed")
                return
            }
            audioRecord = fallbackRecord
        }

        actualSampleRate = record.sampleRate
        record.startRecording()

        Log.i(TAG, "Recording started at ${actualSampleRate}Hz")
        onStatusUpdate?.invoke("录音中... (${actualSampleRate}Hz)")

        // Streaming: feed samples directly to processor incrementally
        val readBuffer = ShortArray(4800) // 100ms read buffer

        while (isRunning) {
            val read = record.read(readBuffer, 0, readBuffer.size)
            if (read > 0) {
                // Feed directly to streaming processor
                val result = processor.feed(readBuffer.copyOf(read))

                if (result != null) {
                    val (original, differential, nFrames) = result
                    onEchoProfileUpdate?.invoke(original, differential, DISTANCE_BINS, nFrames)
                }
            } else if (read < 0) {
                Log.e(TAG, "AudioRecord read error: $read")
                break
            }
        }

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
