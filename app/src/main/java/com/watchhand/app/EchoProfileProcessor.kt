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
    val sampleRate: Int = 44100,
    val fMin: Double = 18000.0,
    val fMax: Double = 20000.0,
    val chirpLength: Int = 588,  // 统一 13.333ms chirp（见 AudioManager.chirpLengthFor）
    val distanceBins: Int = 60,
    val timeWindowFrames: Int = 96
) {
    val distanceResolutionMm: Double = 343.0 / (2.0 * sampleRate) * 1000.0

    // 直达声锚定 bin，与服务端/extract.py 的 DIRECT_BIN 保持一致
    private val directBin = 5

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

    // 不足一个 chirp 的零头先暂存，凑满整 chirp 才做分段相关：
    // 段边界永远是 chirp 整数倍 → 帧网格相位恒定，热力图不闪
    private val pendingBuf = FloatArray(chirpLength)
    private var pendingPos = 0

    // Start alignment offset (found once at beginning)
    private var startOffset = -1
    private val alignBuf = FloatArray(chirpLength * 4)
    private var alignPos = 0
    private var startFound = false
    private var alignPass = 0  // SNR 连续通过计数：满 8 个 chirp 边界才锁定，保证对齐缓冲全是信号

    // 逐 chirp 分段相关（取代整窗 FFT 重算）：每帧仅 60 bins × 600 ≈ 3.6 万次乘加（≈0.05ms），
    // 相对整窗 FFT（≈100ms/帧，超预算 8 倍）快 2000 倍，长时间运行/热降频也不会落后丢样本。
    // 数学上与全流互相关切片等价：seg = 前两个 chirp + 当前 chirp，重叠保证边界完整。
    private val prevChunks = FloatArray(chirpLength * 2)
    private val segBuf = FloatArray(chirpLength * 3)
    private val rollingOrig = FloatArray(distanceBins * timeWindowFrames)
    private val rollingDiff = FloatArray(distanceBins * (timeWindowFrames - 1))
    private val newCol = FloatArray(distanceBins)
    private var producedFrames = 0

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
                // 缓冲满后按 chirp 边界重试对齐：信号未到（SNR 不足）就继续等，
                // 避免“先连接后开音”时锁定在纯噪声上
                if (alignPos >= alignBuf.size && alignPos % chirpLength == 0) {
                    val off = findStartOffset(alignBuf)
                    if (off >= 0) {
                        // 信号刚到时缓冲只有一部分是信号，需连续 8 次通过才锁定
                        alignPass++
                        if (alignPass >= 8) {
                            startOffset = off
                            startFound = true
                        }
                    } else {
                        alignPass = 0
                    }
                }
                continue
            }

            pendingBuf[pendingPos++] = x
            if (pendingPos < chirpLength) continue  // 零头先留着，凑满一个 chirp 再做分段相关
            processChunk()
            pendingPos = 0
        }

        if (producedFrames < timeWindowFrames) return null
        return Triple(rollingOrig.copyOf(), rollingDiff.copyOf(), timeWindowFrames)
    }

    /**
     * 每凑满一个 chirp 产出一帧：对（前 2 个 chirp + 当前 chirp）做分段互相关，
     * 取 60 个距离 bin 作为新列，滚动更新 96 帧窗口。
     * 与全流互相关切片等价，但计算量仅 60×600 乘加/帧。
     */
    private fun processChunk() {
        val L = chirpLength
        // seg = 前两个 chirp + 当前 chirp
        for (i in 0 until 2 * L) segBuf[i] = prevChunks[i]
        for (i in 0 until L) segBuf[2 * L + i] = pendingBuf[i]
        // 滚动 overlap 窗口
        for (i in 0 until L) prevChunks[i] = prevChunks[i + L]
        for (i in 0 until L) prevChunks[L + i] = pendingBuf[i]

        // 取 bin：相关约定 lag = startOffset + d（与 extract.py 互相关一致，
        // 直达声锚定 directBin，流式/批处理/训练逐帧同网格）
        for (d in 0 until distanceBins) {
            val k = startOffset + d
            var sum = 0f
            for (j in 0 until L) sum += segBuf[k + j] * txChirpFloat[j]
            newCol[d] = sum
        }

        // 96 帧窗口左移一列，新列追加到尾部
        for (d in 0 until distanceBins) {
            val base = d * timeWindowFrames
            for (fi in 0 until timeWindowFrames - 1) rollingOrig[base + fi] = rollingOrig[base + fi + 1]
            rollingOrig[base + timeWindowFrames - 1] = newCol[d]
        }
        // 差分轮廓同样滚动：diff[f] = |P[f]| - |P[f-1]|（论文约定，abs 只在这里用）
        for (d in 0 until distanceBins) {
            val base = d * (timeWindowFrames - 1)
            for (fi in 0 until timeWindowFrames - 2) rollingDiff[base + fi] = rollingDiff[base + fi + 1]
            val curr = abs(rollingOrig[d * timeWindowFrames + timeWindowFrames - 1])
            val prev = abs(rollingOrig[d * timeWindowFrames + timeWindowFrames - 2])
            rollingDiff[base + timeWindowFrames - 2] = curr - prev
        }
        producedFrames++
    }

    fun reset() {
        hpState1 = BiquadState()
        hpState2 = BiquadState()
        hpState3 = BiquadState()
        lpState1 = BiquadState()
        lpState2 = BiquadState()
        startOffset = -1
        alignPos = 0
        startFound = false
        alignPass = 0
        pendingPos = 0
        producedFrames = 0
        prevChunks.fill(0f)
        rollingOrig.fill(0f)
        rollingDiff.fill(0f)
    }

    // ===================== Internal methods =====================

    private fun generateChirpFloat(): FloatArray {
        val signal = FloatArray(chirpLength)
        val f0 = fMin; val f1 = fMax
        val T = chirpLength.toDouble() / sampleRate
        for (n in 0 until chirpLength) {
            val t = n.toDouble() / sampleRate
            val phase = 2.0 * PI * (f0 * t + (f1 - f0) * t * t / (2.0 * T))
            // 汉宁窗：与发射 chirp（FmcwGenerator）保持一致，保证匹配滤波一致性
            val window = 0.5 * (1.0 - cos(2.0 * PI * n / (chirpLength - 1)))
            signal[n] = (cos(phase) * window).toFloat()
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
        // 与 extract.py batch_profile 的 p_start 约定一致：相关约定 corr[k]=Σ buf[k+j]·tx[j]，
        // b0 = 逐 chirp 平均幅度最大的 bin，候选 + 直达峰验证，返回 p_start
        val L = chirpLength
        val searchLen = minOf(buf.size + L - 1, L * 3)
        val corr = FloatArray(searchLen)
        var sumAbs = 0f; var maxAbs = 0f
        for (k in 0 until searchLen) {
            var sum = 0f
            val jMax = min(L - 1, buf.size - 1 - k)
            for (j in 0..jMax) sum += buf[k + j] * txChirpFloat[j]
            corr[k] = sum
            val a = abs(sum)
            sumAbs += a
            if (a > maxAbs) maxAbs = a
        }
        // SNR 门槛：直达声峰值需显著高于相关底噪（实测信号 ≈14，静默 ≈5）
        if (sumAbs <= 0f || maxAbs < 9 * sumAbs / searchLen) return -1
        val n0 = (searchLen / L) * L
        var b0 = 0; var best = -1f
        for (d in 0 until L) {
            var s = 0f
            var k = d
            while (k < n0) { s += abs(corr[k]); k += L }
            if (s > best) { best = s; b0 = d }
        }
        var pStart = (b0 - directBin % L + L) % L
        for (cand in intArrayOf((b0 - directBin + L) % L, (directBin - b0 + L) % L)) {
            val nf = (searchLen - cand) / L - 2
            if (nf <= 0) continue
            var am = 0; var amv = -1f
            for (d in 0 until distanceBins) {
                var s = 0f
                for (f2 in 0 until nf) s += abs(corr[cand + f2 * L + d])
                if (s > amv) { amv = s; am = d }
            }
            if (am - directBin in 0..2) { pStart = cand; break }
        }
        // 帧相位对齐：bin d 取 lag = startOffset + d，需 startOffset ≡ pStart (mod L)
        return pStart % L
    }

}
