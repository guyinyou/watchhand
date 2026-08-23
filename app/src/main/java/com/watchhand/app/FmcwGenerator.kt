package com.watchhand.app

import kotlin.math.PI
import kotlin.math.cos

/**
 * Generates FMCW (Frequency-Modulated Continuous Wave) chirp signals.
 *
 * 跨硬件统一参数（适配 48 kHz / 44.1 kHz 原生采样率）：
 * - Frequency range: 18-20 kHz（两种硬件共同的安全频段，
 *   44.1 kHz 下保证 Nyquist 余量；论文为 18-21 kHz @ 48 kHz）
 * - Chirp duration: 13.333 ms（= 4/300 s，两种采样率下均为整数样本：
 *   640 @ 48kHz / 588 @ 44.1kHz，帧率统一 75fps，
 *   由 AudioManager.chirpLengthFor(rate) 计算）
 * - Sample rate: 设备原生采样率（必须，避免系统 SRC 重采样）
 */
class FmcwGenerator(
    val sampleRate: Int = 44100,
    val fMin: Double = 18000.0,
    val fMax: Double = 20000.0,
    val chirpLength: Int = 588
) {
    /** Duration of one chirp in seconds */
    val chirpDuration: Double = chirpLength.toDouble() / sampleRate

    /** Pre-generated single chirp signal (normalized to Short.MAX_VALUE range) */
    val chirpSignal: ShortArray by lazy { generateChirp() }

    /**
     * Generate a single FMCW chirp.
     *
     * Instantaneous frequency sweeps linearly from fMin to fMax:
     *   f(t) = f0 + (f1 - f0) * t / T
     *
     * Phase is the integral of frequency:
     *   phase(t) = 2π * (f0 * t + (f1 - f0) * t² / (2T))
     */
    fun generateChirp(): ShortArray {
        val signal = ShortArray(chirpLength)
        val f0 = fMin
        val f1 = fMax
        val T = chirpDuration

        for (n in 0 until chirpLength) {
            val t = n.toDouble() / sampleRate
            val phase = 2.0 * PI * (f0 * t + (f1 - f0) * t * t / (2.0 * T))
            // Hann window to reduce spectral leakage and transient distortion
            val window = 0.5 * (1.0 - cos(2.0 * PI * n / (chirpLength - 1)))
            val sample = cos(phase) * window
            signal[n] = (sample * Short.MAX_VALUE).toInt().toShort()
        }
        return signal
    }

    /**
     * Generate a long playback buffer by repeating the chirp.
     * @param durationSeconds Total playback duration
     */
    fun generatePlaybackBuffer(durationSeconds: Double): ShortArray {
        val totalSamples = (durationSeconds * sampleRate).toInt()
        val buffer = ShortArray(totalSamples)
        val chirp = chirpSignal

        var pos = 0
        while (pos + chirpLength <= totalSamples) {
            System.arraycopy(chirp, 0, buffer, pos, chirpLength)
            pos += chirpLength
        }
        return buffer
    }

    /**
     * Get the frequency at a given sample index within a chirp.
     */
    fun getFrequencyAtSample(n: Int): Double {
        val t = n.toDouble() / sampleRate
        return fMin + (fMax - fMin) * t / chirpDuration
    }
}
