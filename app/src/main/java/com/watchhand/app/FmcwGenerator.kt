package com.watchhand.app

import kotlin.math.PI
import kotlin.math.cos

/**
 * Generates FMCW (Frequency-Modulated Continuous Wave) chirp signals.
 *
 * Parameters match the WatchHand paper:
 * - Frequency range: 18-21 kHz (inaudible)
 * - Chirp length: 600 samples
 * - Sample rate: 48 kHz
 * - Chirp duration: 12.5 ms
 */
class FmcwGenerator(
    val sampleRate: Int = 48000,
    val fMin: Double = 18000.0,
    val fMax: Double = 21000.0,
    val chirpLength: Int = 600
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
            val sample = cos(phase)
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
