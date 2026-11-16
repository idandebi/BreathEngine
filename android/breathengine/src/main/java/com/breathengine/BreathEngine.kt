package com.breathengine

import android.graphics.Bitmap
import android.graphics.Rect
import java.nio.ByteBuffer

/**
 * Low-level engine: one call per camera frame -> [BreathResult].
 * Use this when you already have frames and shoulder positions. For a
 * ready-made camera pipeline use [BreathSession] or [BreathAnalyzer].
 *
 * Not thread-safe: call from a single thread (e.g. the camera analyzer thread).
 */
class BreathEngine(val config: BreathConfig = BreathConfig()) : AutoCloseable {

    /** Pixel formats for [process] with raw buffers. Codes match the C API. */
    enum class PixelFormat(val code: Int) { RGB(0), BGR(1), RGBA(2), BGRA(3), NV21(4), NV12(5) }

    private var handle: Long = NativeBridge.nativeCreate(config, config.classifier.code, config.signalSource.code)
    private val out = FloatArray(NativeBridge.RESULT_SIZE)

    init {
        if (handle == 0L) throw BreathEngineException("Failed to create native engine")
    }

    /**
     * @param bitmap upright ARGB_8888 frame (the same image the shoulders were detected on).
     * @param shoulders shoulder pixels in [bitmap], or null if no person was found.
     */
    fun process(
        bitmap: Bitmap,
        shoulders: Shoulders?,
        guide: GuideSegment? = null,
        timestampMs: Long = System.currentTimeMillis(),
    ): BreathResult {
        val st = NativeBridge.nativeProcessBitmap(
            checkHandle(), bitmap, shoulders != null,
            shoulders?.leftX ?: 0f, shoulders?.leftY ?: 0f,
            shoulders?.rightX ?: 0f, shoulders?.rightY ?: 0f,
            guide?.code ?: -1, timestampMs.toDouble(), out,
        )
        if (st != 0) throw BreathEngineException("process failed ($st)")
        return toResult(shoulders, bitmap.width, bitmap.height, timestampMs)
    }

    /**
     * Raw buffer variant (direct ByteBuffers). [rotation] is the clockwise
     * rotation that makes the frame upright; shoulders are in upright pixels.
     */
    fun process(
        buffer: ByteBuffer, width: Int, height: Int, rowStride: Int, format: PixelFormat,
        rotation: Int, shoulders: Shoulders?, guide: GuideSegment? = null,
        uvBuffer: ByteBuffer? = null, uvRowStride: Int = 0,
        timestampMs: Long = System.currentTimeMillis(),
    ): BreathResult {
        require(buffer.isDirect) { "buffer must be a direct ByteBuffer" }
        val st = NativeBridge.nativeProcessBuffer(
            checkHandle(), buffer, width, height, rowStride, format.code, rotation,
            uvBuffer, uvRowStride, shoulders != null,
            shoulders?.leftX ?: 0f, shoulders?.leftY ?: 0f,
            shoulders?.rightX ?: 0f, shoulders?.rightY ?: 0f,
            guide?.code ?: -1, timestampMs.toDouble(), out,
        )
        if (st != 0) throw BreathEngineException("process failed ($st)")
        val upright = rotation % 180 == 0
        return toResult(shoulders, if (upright) width else height, if (upright) height else width,
            timestampMs)
    }

    /**
     * Image motion caused by the phone since the previous frame (e.g. from
     * [GyroTracker]); applies to the next [process] call only.
     */
    fun setCameraMotion(m: CameraMotion) =
        NativeBridge.nativeSetCameraMotion(checkHandle(), m.dx, m.dy, m.roll)

    /** Clears the vote window and the tracked region (e.g. when a new session starts). */
    fun reset() = NativeBridge.nativeReset(checkHandle())

    override fun close() {
        if (handle != 0L) {
            NativeBridge.nativeDestroy(handle)
            handle = 0L
        }
    }

    private fun checkHandle(): Long {
        check(handle != 0L) { "BreathEngine is closed" }
        return handle
    }

    private fun toResult(s: Shoulders?, w: Int, h: Int, ts: Long): BreathResult {
        val roiW = out[7].toInt()
        val roi = if (roiW > 0) {
            val x = out[5].toInt()
            val y = out[6].toInt()
            Rect(x, y, x + roiW, y + out[8].toInt())
        } else null
        return BreathResult(
            state = BreathState.fromCode(out[0].toInt()),
            rawState = BreathState.fromCode(out[1].toInt()),
            dyFlow = out[2],
            votesInhale = out[3].toInt(),
            votesExhale = out[4].toInt(),
            roi = roi,
            reanchored = out[9] != 0f,
            flowValid = out[10] != 0f,
            engineMs = out[11],
            shoulders = s,
            frameWidth = w,
            frameHeight = h,
            timestampMs = ts,
            stateVotes = BreathState.fromCode(out[12].toInt()),
            stateSmooth = BreathState.fromCode(out[13].toInt()),
            level = out[14],
            confidence = out[15],
            breathPosition = out[16],
            breathCount = out[17].toInt(),
            breathRateBpm = out[18],
            signal = out[19],
            signalValid = out[20] != 0f,
            dyFeatures = out[21],
            featureCount = out[22].toInt(),
            featureDx = out[23],
            dyChest = out[24],
            dyBelly = out[25],
            camDx = out[26],
            camDy = out[27],
            shaking = out[28] != 0f,
            points = FloatArray(2 * out[29].toInt().coerceIn(0, NativeBridge.MAX_POINTS)) { out[30 + it] },
        )
    }

    companion object {
        @JvmStatic
        val version: String get() = NativeBridge.nativeVersion()
    }
}
