package com.watchhand.app

import android.os.Handler
import android.os.Looper
import android.util.Log
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.launch
import java.io.OutputStream
import java.net.Socket
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * TCP client for streaming raw PCM audio data to a remote server.
 *
 * Protocol:
 * - Connection header: "WATCHHAND\n" + sampleRate (4 bytes, little-endian) + channels (1 byte) + bitsPerSample (1 byte)
 * - Data packets: PCM samples as raw bytes (little-endian 16-bit)
 */
class TcpAudioClient(
    private val serverHost: String,
    private val serverPort: Int,
    private val sampleRate: Int,
    private val onConnectionStatus: ((String) -> Unit)? = null,
    private val onLog: ((String) -> Unit)? = null
) {
    companion object {
        private const val TAG = "WatchHand-TcpClient"
        private const val CONNECTION_HEADER = "WATCHHAND\n"
        /** Global singleton instance - always accessible, no Compose state capture issues */
        @Volatile
        var instance: TcpAudioClient? = null
            private set
    }

    private var socket: Socket? = null
    private var outputStream: OutputStream? = null
    private var scope: CoroutineScope? = null
    private var isConnected = false
    private var headerSent = false  // Header sent on first data, not on connect
    private val mainHandler = Handler(Looper.getMainLooper())

    // Send buffer: accumulate samples and flush when full (reduces TCP small packet overhead)
    private val sendBuffer = ShortArray(sampleRate / 100) // 100ms, 按实际采样率动态计算
    private var sendBufferPos = 0
    // 单线程串行发送队列：避免多个发送协程并发写 OutputStream 导致包序交换
    // （包序交换 = 服务端数据流局部错位 = 热力图周期性断层）
    private val sendQueue = Channel<ShortArray>(300) // ~30s 缓冲

    /**
     * Connect to the server and send connection header.
     */
    fun connect() {
        if (isConnected) return
    
        val self = this
        scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
        scope?.launch {
            try {
                onLog?.invoke("Connecting to $serverHost:$serverPort...")
                socket = Socket(serverHost, serverPort)
                outputStream = socket?.getOutputStream()
    
                // Send connection header immediately
                val headerBytes = CONNECTION_HEADER.toByteArray()
                val sampleRateBytes = ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN)
                    .putInt(sampleRate).array()
                val configBytes = byteArrayOf(1, 16)
    
                outputStream?.write(headerBytes)
                outputStream?.write(sampleRateBytes)
                outputStream?.write(configBytes)
                outputStream?.flush()
    
                isConnected = true
                instance = self
                onLog?.invoke("Header sent: WATCHHAND, ${sampleRate}Hz, mono, 16-bit")

                // 单一发送协程：严格按入队顺序写 socket，保证字节流连续不错位
                val out = outputStream!!
                // 清空上次连接残留
                while (sendQueue.tryReceive().isSuccess) {}
                scope?.launch {
                    for (chunk in sendQueue) {
                        try {
                            val byteBuffer = ByteBuffer.allocate(chunk.size * 2).order(ByteOrder.LITTLE_ENDIAN)
                            for (sample in chunk) byteBuffer.putShort(sample)
                            out.write(byteBuffer.array())
                            out.flush()
                        } catch (e: Exception) {
                            onLog?.invoke("Send failed: ${e.message}")
                            isConnected = false
                            instance = null
                            mainHandler.post {
                                onConnectionStatus?.invoke("发送失败：${e.message}")
                            }
                            break
                        }
                    }
                }
                mainHandler.post {
                    onConnectionStatus?.invoke("已连接：$serverHost:$serverPort")
                }
                onLog?.invoke("Connected!")
            } catch (e: Exception) {
                isConnected = false
                onLog?.invoke("Connection failed: ${e.message}")
                mainHandler.post {
                    onConnectionStatus?.invoke("连接失败：${e.message}")
                }
            }
        }
    }

    /**
     * Send PCM audio samples. Accumulates in buffer, flushes when full.
     */
    fun sendAudio(samples: ShortArray) {
        val out = outputStream ?: return

        var offset = 0
        var remaining = samples.size

        while (remaining > 0) {
            val space = sendBuffer.size - sendBufferPos
            val toCopy = minOf(space, remaining)
            System.arraycopy(samples, offset, sendBuffer, sendBufferPos, toCopy)
            sendBufferPos += toCopy
            offset += toCopy
            remaining -= toCopy

            if (sendBufferPos == sendBuffer.size) {
                val dataToSend = sendBuffer.copyOf()
                sendBufferPos = 0
                // 入队由发送协程串行写出；队列满（网络持续故障）时丢弃并记录
                if (sendQueue.trySend(dataToSend).isFailure) {
                    onLog?.invoke("Send queue full, dropping ${dataToSend.size} samples")
                }
            }
        }
    }

    /**
     * Disconnect from the server.
     */
    fun disconnect() {
        isConnected = false
        instance = null
        try {
            outputStream?.close()
            socket?.close()
        } catch (e: Exception) {
            onLog?.invoke("Error closing: ${e.message}")
        }
        outputStream = null
        socket = null
        scope?.cancel()
        scope = null
        onLog?.invoke("Disconnected")
        onConnectionStatus?.invoke("已断开")
    }

    /**
     * Check if currently connected.
     */
    fun isConnected(): Boolean = isConnected
}
