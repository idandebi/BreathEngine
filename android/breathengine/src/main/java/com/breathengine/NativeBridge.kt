package com.breathengine

import android.graphics.Bitmap
import java.nio.ByteBuffer

internal object NativeBridge {
    const val MAX_POINTS = 256
    const val RESULT_SIZE = 30 + 2 * MAX_POINTS

    init {
        System.loadLibrary("breathengine_jni")
    }

    @JvmStatic external fun nativeVersion(): String
    @JvmStatic external fun nativeCreate(config: BreathConfig?, classifier: Int, signalSource: Int): Long
    @JvmStatic external fun nativeDestroy(handle: Long)
    @JvmStatic external fun nativeReset(handle: Long)
    @JvmStatic external fun nativeSetCameraMotion(handle: Long, dx: Float, dy: Float, roll: Float)

    @JvmStatic external fun nativeProcessBitmap(
        handle: Long, bitmap: Bitmap, hasShoulders: Boolean,
        lx: Float, ly: Float, rx: Float, ry: Float, guide: Int, timestampMs: Double, out: FloatArray,
    ): Int

    @JvmStatic external fun nativeProcessBuffer(
        handle: Long, buffer: ByteBuffer, width: Int, height: Int, stride: Int,
        format: Int, rotation: Int, uv: ByteBuffer?, uvStride: Int, hasShoulders: Boolean,
        lx: Float, ly: Float, rx: Float, ry: Float, guide: Int, timestampMs: Double, out: FloatArray,
    ): Int
}
