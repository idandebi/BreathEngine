package com.breathengine

import android.content.Context
import android.graphics.Bitmap
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.core.Delegate
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarker

/**
 * Finds the shoulders with MediaPipe PoseLandmarker (pose_landmarker_full,
 * bundled from models/). Synchronous, VIDEO running mode.
 */
class PoseDetector(
    context: Context,
    modelAsset: String = DEFAULT_MODEL,
    useGpu: Boolean = false,
    minDetectionConfidence: Float = 0.5f,
    minTrackingConfidence: Float = 0.5f,
) : AutoCloseable {

    private val landmarker: PoseLandmarker
    private var lastTs = -1L

    init {
        val base = BaseOptions.builder()
            .setModelAssetPath(modelAsset)
            .setDelegate(if (useGpu) Delegate.GPU else Delegate.CPU)
            .build()
        val opts = PoseLandmarker.PoseLandmarkerOptions.builder()
            .setBaseOptions(base)
            .setRunningMode(RunningMode.VIDEO)
            .setNumPoses(1)
            .setMinPoseDetectionConfidence(minDetectionConfidence)
            .setMinPosePresenceConfidence(0.5f)
            .setMinTrackingConfidence(minTrackingConfidence)
            .build()
        landmarker = PoseLandmarker.createFromOptions(context, opts)
    }

    /** @param bitmap upright frame. @return shoulders in its pixels, or null. */
    fun detect(bitmap: Bitmap, timestampMs: Long): Shoulders? {
        // VIDEO mode requires strictly increasing timestamps.
        val ts = maxOf(timestampMs, lastTs + 1)
        lastTs = ts
        val result = landmarker.detectForVideo(BitmapImageBuilder(bitmap).build(), ts)
        val pose = result.landmarks().firstOrNull() ?: return null
        if (pose.size <= RIGHT_SHOULDER) return null
        val l = pose[LEFT_SHOULDER]
        val r = pose[RIGHT_SHOULDER]
        val w = bitmap.width.toFloat()
        val h = bitmap.height.toFloat()
        return Shoulders(l.x() * w, l.y() * h, r.x() * w, r.y() * h)
    }

    override fun close() = landmarker.close()

    companion object {
        const val DEFAULT_MODEL = "pose_landmarker_full.task"
        const val LEFT_SHOULDER = 11
        const val RIGHT_SHOULDER = 12
    }
}
