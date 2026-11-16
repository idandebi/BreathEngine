package com.breathengine

import android.annotation.SuppressLint
import android.content.Context
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CaptureRequest
import android.hardware.display.DisplayManager
import android.os.Handler
import android.os.Looper
import android.util.Log
import android.util.Range
import android.util.Size
import android.view.Display
import android.view.Surface
import androidx.annotation.OptIn
import androidx.camera.camera2.interop.Camera2CameraInfo
import androidx.camera.camera2.interop.Camera2Interop
import androidx.camera.camera2.interop.ExperimentalCamera2Interop
import androidx.camera.core.CameraInfo
import androidx.camera.core.CameraSelector
import androidx.camera.core.ExtendableBuilder
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import kotlin.math.atan
import kotlin.math.max

/**
 * High-level API: opens the camera and streams breathing results.
 *
 * ```
 * val session = BreathSession(context, lifecycleOwner, previewView) { result ->
 *     label.text = result.state.name
 * }
 * session.start()          // requires CAMERA permission
 * ...
 * session.close()
 * ```
 * Results are delivered on the main thread.
 */
class BreathSession(
    private val context: Context,
    private val lifecycleOwner: LifecycleOwner,
    private val previewView: PreviewView? = null,
    private val options: Options = Options(),
    private val listener: (BreathResult) -> Unit,
) : AutoCloseable {

    data class Options(
        val config: BreathConfig = BreathConfig(),
        val useFrontCamera: Boolean = true,
        /** Analysis resolution; the ROI scales with the person, 640x480 is plenty. */
        val analysisSize: Size = Size(640, 480),
        val useGpuForPose: Boolean = false,
        /** Requested camera frame rate (e.g. 30 or 60); the closest supported range is used. */
        val targetFps: Int = 30,
        /**
         * Use the widest view available on the chosen side ("0.5x"): the camera
         * with the widest field of view, at its minimum zoom ratio.
         */
        val wideAngle: Boolean = false,
        /** Measure the phone's own motion with the gyroscope and remove it. */
        val compensatePhoneMotion: Boolean = true,
    )

    /** Camera frame-rate range actually requested, or null if the camera default is used. */
    var fpsRange: Range<Int>? = null
        private set

    /** Human-readable description of the camera in use (id, field of view, zoom). */
    var cameraDescription: String = ""
        private set

    private val main = Handler(Looper.getMainLooper())
    private val executor: ExecutorService = Executors.newSingleThreadExecutor()
    private val analyzer = BreathAnalyzer(context, options.config, options.useGpuForPose) { r ->
        main.post { if (running) listener(r) }
    }
    private var provider: ProcessCameraProvider? = null
    private var analysis: ImageAnalysis? = null
    private var preview: Preview? = null
    @Volatile private var running = false

    private val displayManager = context.getSystemService(Context.DISPLAY_SERVICE) as DisplayManager

    // Keeps frames upright when the screen rotates without recreating the
    // activity (e.g. 180° flip between the two landscape orientations).
    private val displayListener = object : DisplayManager.DisplayListener {
        override fun onDisplayAdded(displayId: Int) = Unit
        override fun onDisplayRemoved(displayId: Int) = Unit
        override fun onDisplayChanged(displayId: Int) {
            val r = currentRotation()
            setTargetRotation(r)
            analyzer.displayRotation = r
        }
    }

    val isFrontCamera: Boolean get() = options.useFrontCamera

    /** Phase of your breathing guide (optional). */
    var guideSegment: GuideSegment?
        get() = analyzer.guideSegment
        set(value) { analyzer.guideSegment = value }

    fun start() {
        if (running) return
        running = true
        analyzer.reset()
        val future = ProcessCameraProvider.getInstance(context)
        future.addListener({
            if (!running) return@addListener
            val p = future.get()
            provider = p
            val facing = if (options.useFrontCamera) CameraSelector.LENS_FACING_FRONT
            else CameraSelector.LENS_FACING_BACK
            val candidates = p.availableCameraInfos.filter { it.lensFacing == facing }
            candidates.forEach { Log.i(TAG, "camera ${describe(it)}") }
            val chosen = if (options.wideAngle) candidates.maxByOrNull { fovDegrees(it) } else null
            val selector = chosen?.cameraSelector
                ?: if (options.useFrontCamera) CameraSelector.DEFAULT_FRONT_CAMERA
                else CameraSelector.DEFAULT_BACK_CAMERA
            val resolution = ResolutionSelector.Builder()
                .setResolutionStrategy(ResolutionStrategy(options.analysisSize,
                    ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER))
                .build()
            val rotation = currentRotation()
            fpsRange = pickFpsRange(p, selector, options.targetFps)
            val a = ImageAnalysis.Builder()
                .setResolutionSelector(resolution)
                .setTargetRotation(rotation)
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .also { applyFps(it) }
                .build()
                .also { it.setAnalyzer(executor, analyzer) }
            analysis = a
            val useCases = mutableListOf<androidx.camera.core.UseCase>(a)
            previewView?.let { pv ->
                val pr = Preview.Builder().setTargetRotation(rotation).also { applyFps(it) }.build()
                    .also { it.surfaceProvider = pv.surfaceProvider }
                preview = pr
                useCases += pr
            }
            p.unbindAll()
            val camera = p.bindToLifecycle(lifecycleOwner, selector, *useCases.toTypedArray())
            if (options.wideAngle) {
                val minZoom = camera.cameraInfo.zoomState.value?.minZoomRatio ?: 1f
                camera.cameraControl.setZoomRatio(minZoom)
            }
            cameraDescription = describe(camera.cameraInfo, if (options.wideAngle) "min zoom" else "")
            analyzer.displayRotation = rotation
            if (options.compensatePhoneMotion) {
                analyzer.cameraGeometry = geometry(camera.cameraInfo)
                analyzer.startSensors()
            }
            Log.i(TAG, "using camera $cameraDescription")
            displayManager.registerDisplayListener(displayListener, main)
        }, ContextCompat.getMainExecutor(context))
    }

    /**
     * Screen rotation ([Surface.ROTATION_0] etc.) the frames should be upright
     * for. Updated automatically while running; call it yourself only if you
     * rotate your UI independently of the display.
     */
    fun setTargetRotation(rotation: Int) {
        analysis?.targetRotation = rotation
        preview?.targetRotation = rotation
    }

    @SuppressLint("UnsafeOptInUsageError")
    @OptIn(ExperimentalCamera2Interop::class)
    private fun applyFps(builder: ExtendableBuilder<*>) {
        val range = fpsRange ?: return
        Camera2Interop.Extender(builder)
            .setCaptureRequestOption(CaptureRequest.CONTROL_AE_TARGET_FPS_RANGE, range)
    }

    /** Prefer a fixed [fps, fps] range, else the range reaching fps with the highest minimum. */
    @SuppressLint("UnsafeOptInUsageError")
    @OptIn(ExperimentalCamera2Interop::class)
    private fun pickFpsRange(p: ProcessCameraProvider, selector: CameraSelector, fps: Int): Range<Int>? {
        return try {
            val info = selector.filter(p.availableCameraInfos).firstOrNull() ?: return null
            val ranges = Camera2CameraInfo.from(info)
                .getCameraCharacteristic(CameraCharacteristics.CONTROL_AE_AVAILABLE_TARGET_FPS_RANGES)
                ?: return null
            Log.i(TAG, "camera fps ranges: ${ranges.joinToString()}")
            ranges.filter { it.upper == fps }.maxByOrNull { it.lower }
                ?: ranges.filter { it.upper >= fps }.minByOrNull { it.upper }
        } catch (t: Throwable) {
            Log.w(TAG, "fps range query failed", t)
            null
        }
    }

    /** Horizontal field of view in degrees (from focal length and sensor size). */
    @SuppressLint("UnsafeOptInUsageError")
    @OptIn(ExperimentalCamera2Interop::class)
    private fun fovDegrees(info: CameraInfo): Double = try {
        val c = Camera2CameraInfo.from(info)
        val focal = c.getCameraCharacteristic(CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS)
            ?.minOrNull()
        val sensor = c.getCameraCharacteristic(CameraCharacteristics.SENSOR_INFO_PHYSICAL_SIZE)
        if (focal == null || sensor == null || focal <= 0f) 0.0
        else Math.toDegrees(2 * atan(max(sensor.width, sensor.height) / (2.0 * focal)))
    } catch (t: Throwable) {
        0.0
    }

    @SuppressLint("UnsafeOptInUsageError")
    @OptIn(ExperimentalCamera2Interop::class)
    private fun describe(info: CameraInfo, extra: String = ""): String {
        val id = try { Camera2CameraInfo.from(info).cameraId } catch (t: Throwable) { "?" }
        val z = info.zoomState.value
        val zoom = if (z != null) "zoom %.2f-%.1f".format(z.minZoomRatio, z.maxZoomRatio) else ""
        return "id $id fov %.0f° $zoom $extra".format(fovDegrees(info)).trim()
    }

    @SuppressLint("UnsafeOptInUsageError")
    @OptIn(ExperimentalCamera2Interop::class)
    private fun geometry(info: CameraInfo): CameraGeometry? = try {
        val c = Camera2CameraInfo.from(info)
        val focal = c.getCameraCharacteristic(CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS)
            ?.minOrNull()
        val sensor = c.getCameraCharacteristic(CameraCharacteristics.SENSOR_INFO_PHYSICAL_SIZE)
        val source = c.getCameraCharacteristic(CameraCharacteristics.SENSOR_INFO_TIMESTAMP_SOURCE)
        if (focal == null || sensor == null || focal <= 0f) null
        else CameraGeometry(
            isFront = info.lensFacing == CameraSelector.LENS_FACING_FRONT,
            focalOverSensor = focal / max(sensor.width, sensor.height).toDouble(),
            realtimeTimestamps = source == CameraCharacteristics.SENSOR_INFO_TIMESTAMP_SOURCE_REALTIME,
        ).also { Log.i(TAG, "camera geometry $it") }
    } catch (t: Throwable) {
        Log.w(TAG, "camera geometry unavailable", t)
        null
    }

    private fun currentRotation(): Int =
        previewView?.display?.rotation
            ?: displayManager.getDisplay(Display.DEFAULT_DISPLAY)?.rotation
            ?: Surface.ROTATION_0

    fun stop() {
        running = false
        analyzer.stopSensors()
        displayManager.unregisterDisplayListener(displayListener)
        provider?.unbindAll()
        analysis = null
        preview = null
    }

    override fun close() {
        stop()
        executor.execute { analyzer.close() }
        executor.shutdown()
    }

    companion object {
        private const val TAG = "BreathSession"
    }
}
