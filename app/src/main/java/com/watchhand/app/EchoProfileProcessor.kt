package com.watchhand.app

import kotlin.math.*

/**
 * Echo Profile Processor - matches paper's Algorithm 1 exactly.
 *
 * Pipeline:
 * 1. Bandpass filter (3 HP + 2 LP biquad cascade ≈ 5th-order Butterworth)
 * 2. Accumulate filtered samples in a ring buffer
 * 3. When enough data: full cross-correlation → reshape → crop to 60×96
 * 4. Differential profile: |P[f]| - |P[f-1]|
 */
class EchoProfileProcessor(
    val sampleRate: Int = 48000,
    val fMin: Double = 18000.0,
    val fMax: Double = 21000.0,
    val chirpLength: Int = 600,
    val distanceBins: Int = 60,
    val timeWindowFrames: Int = 96
) {
    val distanceResolutionMm: Double = 343.0 / (2.0 * sampleRate) * 1000.0

    // --- Internal state ---
    private val txChirpFloat: FloatArray

    // Biquad filter states (persist across feed calls)
    private var hpState1 = BiquadState()
    private var hpState2 = BiquadState()
    private var hpState3 = BiquadState()
    private var lpState1 = BiquadState()
    private var lpState2 = BiquadState()
    private val hpCoeffs: FloatArray
    private val lpCoeffs: FloatArray

    // Ring buffer for filtered audio: need enough for correlation + margin
    // Correlation output length = rxLen + chirpLen - 1
    // We need at least timeWindowFrames chirps of valid correlation data
    // So rx buffer needs: (timeWindowFrames + margin) * chirpLength samples
    private val processChirps = timeWindowFrames + 30 // extra for alignment margin
    private val rxBufferSize = processChirps * chirpLength
    private val rxRing = FloatArray(rxBufferSize)
    private var rxHead = 0       // next write position in ring
    private var rxFilled = 0     // how many samples filled so far

    // Start alignment offset (found once at beginning)
    private var startOffset = -1
    private val alignBuf = FloatArray(chirpLength * 4)
    private var alignPos = 0
    private var startFound = false

    init {
        txChirpFloat = generateChirpFloat()
        hpCoeffs = computeHighpassCoeffs(fMin, sampleRate)
        lpCoeffs = computeLowpassCoeffs(fMax, sampleRate)
    }

    /**
     * Feed raw audio samples. Returns processed profiles when enough data accumulated.
     */
    fun feed(samples: ShortArray): Triple<FloatArray, FloatArray, Int>? {
        // Filter and accumulate
        for (s in samples) {
            var x = s.toFloat()
            x = biquadStep(x, hpCoeffs, hpState1)
            x = biquadStep(x, hpCoeffs, hpState2)
            x = biquadStep(x, hpCoeffs, hpState3)
            x = biquadStep(x, lpCoeffs, lpState1)
            x = biquadStep(x, lpCoeffs, lpState2)

            if (!startFound) {
                alignBuf[alignPos % alignBuf.size] = x
                alignPos++
                if (alignPos >= alignBuf.size) {
                    startOffset = findStartOffset(alignBuf)
                    startFound = true
                    rxHead = 0
                    rxFilled = 0
                }
                continue
            }

            rxRing[rxHead % rxRing.size] = x
            rxHead++
            if (rxFilled < rxRing.size) rxFilled++
        }

        // Process when we have enough new data since last processing
        // Need at least chirpLength new samples to produce one new frame
        val minNewSamples = chirpLength
        if (rxFilled < rxBufferSize || !startFound) return null

        // Extract contiguous window from ring buffer for processing
        val extractSize = rxBufferSize
        val startIdx = ((rxHead - extractSize) % rxRing.size + rxRing.size) % rxRing.size
        val rxData = FloatArray(extractSize)
        for (i in 0 until extractSize) {
            rxData[i] = rxRing[(startIdx + i) % rxRing.size]
        }

        // Apply start alignment: skip startOffset samples
        val alignedRx = rxData.copyOfRange(startOffset.coerceIn(0, rxData.size - 1), rxData.size)

        // Full cross-correlation
        val corr = crossCorrelateFull(alignedRx, txChirpFloat)

        // Reshape into frames: each frame is chirpLength samples
        val nFrames = corr.size / chirpLength
        if (nFrames < timeWindowFrames) return null

        // Build original profile: take first distanceBins of each frame
        // Keep only the most recent timeWindowFrames
        val frameStart = nFrames - timeWindowFrames
        val originalProfile = FloatArray(distanceBins * timeWindowFrames)
        for (fi in 0 until timeWindowFrames) {
            val frameIdx = frameStart + fi
            val offset = frameIdx * chirpLength
            for (d in 0 until distanceBins) {
                originalProfile[d * timeWindowFrames + fi] = corr[offset + d]
            }
        }

        // Differential profile
        val diffProfile = FloatArray(distanceBins * (timeWindowFrames - 1))
        for (fi in 1 until timeWindowFrames) {
            for (d in 0 until distanceBins) {
                val curr = abs(originalProfile[d * timeWindowFrames + fi])
                val prev = abs(originalProfile[d * timeWindowFrames + fi - 1])
                diffProfile[d * (timeWindowFrames - 1) + fi - 1] = curr - prev
            }
        }

        return Triple(originalProfile, diffProfile, timeWindowFrames)
    }

    fun reset() {
        hpState1 = BiquadState()
        hpState2 = BiquadState()
        hpState3 = BiquadState()
        lpState1 = BiquadState()
        lpState2 = BiquadState()
        rxHead = 0
        rxFilled = 0
        startOffset = -1
        alignPos = 0
        startFound = false
    }

    // ===================== Internal methods =====================

    private fun generateChirpFloat(): FloatArray {
        val signal = FloatArray(chirpLength)
        val f0 = fMin; val f1 = fMax
        val T = chirpLength.toDouble() / sampleRate
        for (n in 0 until chirpLength) {
            val t = n.toDouble() / sampleRate
            val phase = 2.0 * PI * (f0 * t + (f1 - f0) * t * t / (2.0 * T))
            signal[n] = cos(phase).toFloat()
        }
        return signal
    }

    private fun computeHighpassCoeffs(fc: Double, fs: Int): FloatArray {
        val w0 = 2.0 * PI * fc / fs
        val cosW0 = cos(w0); val sinW0 = sin(w0)
        val alpha = sinW0 / (2.0 * 0.707)
        val a0 = 1.0 + alpha
        return floatArrayOf(
            ((1.0 + cosW0) / 2.0 / a0).toFloat(),
            (-(1.0 + cosW0) / a0).toFloat(),
            ((1.0 + cosW0) / 2.0 / a0).toFloat(),
            (-2.0 * cosW0 / a0).toFloat(),
            ((1.0 - alpha) / a0).toFloat()
        )
    }

    private fun computeLowpassCoeffs(fc: Double, fs: Int): FloatArray {
        val w0 = 2.0 * PI * fc / fs
        val cosW0 = cos(w0); val sinW0 = sin(w0)
        val alpha = sinW0 / (2.0 * 0.707)
        val a0 = 1.0 + alpha
        return floatArrayOf(
            ((1.0 - cosW0) / 2.0 / a0).toFloat(),
            ((1.0 - cosW0) / a0).toFloat(),
            ((1.0 - cosW0) / 2.0 / a0).toFloat(),
            (-2.0 * cosW0 / a0).toFloat(),
            ((1.0 - alpha) / a0).toFloat()
        )
    }

    private fun biquadStep(x: Float, coeffs: FloatArray, state: BiquadState): Float {
        val b0 = coeffs[0]; val b1 = coeffs[1]; val b2 = coeffs[2]
        val a1 = coeffs[3]; val a2 = coeffs[4]
        val y = b0 * x + state.z1
        state.z1 = b1 * x - a1 * y + state.z2
        state.z2 = b2 * x - a2 * y
        return y
    }

    private class BiquadState(var z1: Float = 0f, var z2: Float = 0f)

    private fun findStartOffset(buf: FloatArray): Int {
        // Cross-correlate with tx chirp, find earliest strong peak
        val searchLen = minOf(buf.size + chirpLength - 1, chirpLength * 3)
        var bestIdx = 0; var bestVal = 0f
        for (k in 0 until searchLen) {
            var sum = 0f
            val jMin = max(0, k - buf.size + 1)
            val jMax = min(chirpLength - 1, k)
            for (j in jMin..jMax) sum += buf[k - j] * txChirpFloat[j]
            val v = abs(sum)
            if (v > bestVal) { bestVal = v; bestIdx = k }
        }
        // Paper's adjustment: align to frame center
        val halfL = chirpLength / 2
        return ((bestIdx + chirpLength - halfL) % chirpLength).coerceIn(0, chirpLength - 1)
    }

    /**
     * Full cross-correlation: output[k] = sum_j rx[k-j] * tx[j]
     * This matches the paper's Rx  Tx operation.
     */
    private fun crossCorrelateFull(rx: FloatArray, tx: FloatArray): FloatArray {
        val n = rx.size + tx.size - 1
        val result = FloatArray(n)
        // Only compute up to what we need: nFrames * chirpLength where nFrames = timeWindowFrames + margin
        val needed = (processChirps) * chirpLength
        val computeLen = minOf(n, needed)
        for (k in 0 until computeLen) {
            var sum = 0f
            val jMin = max(0, k - rx.size + 1)
            val jMax = min(tx.size - 1, k)
            for (j in jMin..jMax) sum += rx[k - j] * tx[j]
            result[k] = sum
        }
        return result
    }
}
