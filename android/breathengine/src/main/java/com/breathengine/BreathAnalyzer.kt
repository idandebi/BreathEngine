package com.breathengine

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Matrix
import android.os.SystemClock
import android.util.Log
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean

/**
 * CameraX analyzer: frame -> upright bitmap -> shoulders -> [BreathEngine].
 * Plug it into your own ImageAnalysis use case (RGBA_8888 output), or let
 * [BreathSession] do everything.
 *
 * Shoulder detection runs asynchronously on its own thread and the engine
 * uses the latest shoulders, so the engine sees every camera frame (the vote
 * window is counted in frames). This mirrors the Unity build, which read
 * MediaPipe's `latestResult`. The ROI is anchored between re-anchors, so a
 * shoulder result that is a frame or two old is fine.
 *
 * [onResult] is called on the analyzer thread.
 */
class BreathAnalyzer(
    context: Context,
    config: BreathConfig = BreathConfig(),
    useGpuForPose: Boolean = false,
    /** false = detect shoulders on every frame before the engine (slower). */
    private val asyncPose: Boolean = true,
    /**
     * Shoulders older than this (from the frame they were detected on) are
     * treated as "no person". Pose can take 300-400 ms on a busy GPU, so this
     * must be well above one pose run, or most frames lose their shoulders.
     * Shoulders move slowly; the grid tracks the breathing motion itself.
     */
    private val maxPoseAgeMs: Long = 1500,
    private val onResult: (BreathResult) -> Unit,
) : ImageAnalysis.Analyzer, AutoCloseable {

    private val engine = BreathEngine(config)
    private val pose = PoseDetector(context.applicationContext, useGpu = useGpuForPose)
    private val poseExecutor: ExecutorService = Executors.newSingleThreadExecutor()
    private val poseBusy = AtomicBoolean(false)
    private var firstFrameNs = -1L
    private var lastTs = -1L
    private val gyro = GyroTracker(context.applicationContext)
    private var prevFrameRtNs = -1L

    /** Set by [BreathSession] once the camera is chosen; null = no motion compensation. */
    @Volatile var cameraGeometry: CameraGeometry? = null

    /** Display rotation the frames are upright for (Surface.ROTATION_*). */
    @Volatile var displayRotation: Int = android.view.Surface.ROTATION_0

    /** Scale for the gyroscope -> pixel conversion (1 = from lens data). */
    @Volatile var gyroGain: Double = 1.0

    private class PoseSample(val shoulders: Shoulders?, val atMs: Long, val w: Int, val h: Int)

    @Volatile private var latestPose: PoseSample? = null
    @Volatile private var lastPoseMs = 0f

    /** Current phase of your on-screen guide, or null for unguided detection. */
    @Volatile var guideSegment: GuideSegment? = null

    @Volatile private var resetRequested = false
    @Volatile private var closed = false

    /** Restart detection (call when a new session starts). */
    fun reset() {
        resetRequested = true
    }

    /** Starts/stops reading the gyroscope (phone-motion compensation). */
    fun startSensors() = gyro.start()
    fun stopSensors() = gyro.stop()

    override fun analyze(image: ImageProxy) {
        try {
            if (closed) return
            if (resetRequested) {
                engine.reset()
                latestPose = null
                firstFrameNs = -1L
                lastTs = -1L
                prevFrameRtNs = -1L
                resetRequested = false
            }
            val t0 = SystemClock.elapsedRealtimeNanos()
            val frame = toUprightBitmap(image)
            val t1 = SystemClock.elapsedRealtimeNanos()
            // Capture time of the frame (ms since the first frame): the smooth
            // classifier uses real frame spacing, not processing time.
            val camNs = image.imageInfo.timestamp
            if (firstFrameNs < 0 || camNs < firstFrameNs) firstFrameNs = camNs
            val ts = maxOf((camNs - firstFrameNs) / 1_000_000L, lastTs + 1)
            lastTs = ts

            if (asyncPose) {
                if (poseBusy.compareAndSet(false, true)) {
                    poseExecutor.execute { runPose(frame, ts) }
                }
            } else {
                runPose(frame, ts)
            }

            val p = latestPose
            val shoulders = if (p != null && ts - p.atMs <= maxPoseAgeMs &&
                p.w == frame.width && p.h == frame.height) p.shoulders else null
            applyCameraMotion(camNs, frame.width, frame.height)
            val t2 = SystemClock.elapsedRealtimeNanos()
            val r = engine.process(frame, shoulders, guideSegment, ts)
            val t3 = SystemClock.elapsedRealtimeNanos()
            val syncPoseMs = if (asyncPose) 0f else lastPoseMs
            onResult(r.copy(
                totalMs = (t3 - t0) / 1e6f - (if (asyncPose) (t2 - t1) / 1e6f else 0f),
                convertMs = (t1 - t0) / 1e6f,
                poseMs = if (asyncPose) lastPoseMs else syncPoseMs,
            ))
        } catch (t: Throwable) {
            Log.e(TAG, "analyze failed", t)
        } finally {
            image.close()
        }
    }

    /** Phone rotation since the previous frame -> image motion for the engine. */
    private fun applyCameraMotion(camNs: Long, w: Int, h: Int) {
        val geom = cameraGeometry ?: return
        // Gyroscope events use the elapsedRealtime clock; some cameras use the
        // monotonic clock (no deep-sleep time), so convert those.
        val rtNs = if (geom.realtimeTimestamps) camNs
        else camNs + (SystemClock.elapsedRealtimeNanos() - System.nanoTime())
        val prev = prevFrameRtNs
        prevFrameRtNs = rtNs
        if (prev < 0) return
        val rot = gyro.integrate(prev, rtNs) ?: return
        engine.setCameraMotion(GyroTracker.toImageMotion(rot, geom, displayRotation, w, h, gyroGain))
    }

    private fun runPose(frame: Bitmap, ts: Long) {
        try {
            if (closed) return
            val t = SystemClock.elapsedRealtimeNanos()
            val s = pose.detect(frame, ts)
            lastPoseMs = (SystemClock.elapsedRealtimeNanos() - t) / 1e6f
            latestPose = PoseSample(s, ts, frame.width, frame.height)
        } catch (t: Throwable) {
            Log.e(TAG, "pose failed", t)
        } finally {
            poseBusy.set(false)
        }
    }

    override fun close() {
        closed = true
        gyro.close()
        poseExecutor.execute { pose.close() }
        poseExecutor.shutdown()
        engine.close()
    }

    private fun toUprightBitmap(image: ImageProxy): Bitmap {
        val bmp = image.toBitmap()
        val rot = image.imageInfo.rotationDegrees
        if (rot == 0) return bmp
        val m = Matrix().apply { postRotate(rot.toFloat()) }
        return Bitmap.createBitmap(bmp, 0, 0, bmp.width, bmp.height, m, false)
    }

    companion object {
        private const val TAG = "BreathAnalyzer"
    }
}
