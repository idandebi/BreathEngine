package com.breathengine.sample

import android.Manifest
import android.content.pm.ActivityInfo
import android.content.pm.PackageManager
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.os.SystemClock
import android.widget.Button
import android.widget.TextView
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import com.breathengine.BreathConfig
import com.breathengine.BreathResult
import com.breathengine.BreathSession
import com.breathengine.GuideSegment
import com.breathengine.SignalSource

class MainActivity : ComponentActivity() {

    private var session: BreathSession? = null
    private lateinit var overlay: OverlayView
    private lateinit var stateText: TextView
    private lateinit var votesText: TextView
    private lateinit var infoText: TextView
    private lateinit var guideText: TextView
    private lateinit var recButton: Button
    private lateinit var guideButton: Button

    // Guide patterns in seconds: inhale, hold, exhale, hold (0 = skip).
    private val patterns = listOf(
        "off" to null,
        "4-4-4-4" to listOf(4.0, 4.0, 4.0, 4.0),
        "4-0-4-0" to listOf(4.0, 0.0, 4.0, 0.0),
        "4-7-8" to listOf(4.0, 7.0, 8.0, 0.0),
    )
    private val segs = listOf(GuideSegment.INHALE, GuideSegment.HOLD_TOP, GuideSegment.EXHALE,
        GuideSegment.HOLD_BOTTOM)
    private var patternIdx = 0
    private var guideStart = 0L
    private val ui = Handler(Looper.getMainLooper())
    private var fps = 0f
    private var lastResultAt = 0L
    private var logger: SessionLogger? = null
    private var recStart = 0L
    private var landscape = false

    private val requestCamera =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
            if (granted) startSession()
            else Toast.makeText(this, "Camera permission is required", Toast.LENGTH_LONG).show()
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        landscape = prefs.getBoolean(KEY_LANDSCAPE, false)
        // Changing this recreates the activity; the session restarts with the new rotation.
        requestedOrientation = if (landscape) ActivityInfo.SCREEN_ORIENTATION_SENSOR_LANDSCAPE
        else ActivityInfo.SCREEN_ORIENTATION_PORTRAIT

        setContentView(R.layout.activity_main)
        overlay = findViewById(R.id.overlay)
        stateText = findViewById(R.id.state)
        votesText = findViewById(R.id.stateVotes)
        infoText = findViewById(R.id.info)
        guideText = findViewById(R.id.guide)
        recButton = findViewById(R.id.recButton)
        guideButton = findViewById(R.id.guideButton)

        patternIdx = savedInstanceState?.getInt(KEY_PATTERN) ?: 0
        guideStart = savedInstanceState?.getLong(KEY_GUIDE_START) ?: SystemClock.elapsedRealtime()
        guideButton.text = "Guide: ${patterns[patternIdx].first}"
        guideButton.setOnClickListener {
            if (logger != null) return@setOnClickListener  // keep the guide fixed while recording
            setPattern((patternIdx + 1) % patterns.size)
        }

        findViewById<Button>(R.id.orientationButton).apply {
            text = if (landscape) "Landscape" else "Portrait"
            setOnClickListener {
                if (logger != null) return@setOnClickListener
                prefs.edit().putBoolean(KEY_LANDSCAPE, !landscape).apply()
                requestedOrientation = if (!landscape) ActivityInfo.SCREEN_ORIENTATION_SENSOR_LANDSCAPE
                else ActivityInfo.SCREEN_ORIENTATION_PORTRAIT
            }
        }
        recButton.setOnClickListener { if (logger == null) startRecording() else stopRecording() }

        findViewById<Button>(R.id.signalButton).apply {
            text = "Signal: ${SIGNALS[signalIdx].first}"
            setOnClickListener {
                if (logger != null) return@setOnClickListener
                prefs.edit().putInt(KEY_SIGNAL, (signalIdx + 1) % SIGNALS.size).apply()
                recreate()
            }
        }
        findViewById<Button>(R.id.camButton).apply {
            text = "Cam: ${CAM_MODES[camMode]}"
            setOnClickListener {
                if (logger != null) return@setOnClickListener
                prefs.edit().putInt(KEY_CAM, (camMode + 1) % CAM_MODES.size).apply()
                recreate()
            }
        }
        findViewById<Button>(R.id.fpsButton).apply {
            text = "$targetFps FPS"
            setOnClickListener {
                if (logger != null) return@setOnClickListener
                prefs.edit().putInt(KEY_FPS, if (targetFps == 60) 30 else 60).apply()
                recreate()
            }
        }

        // Tap the status panel to switch shoulder detection between GPU and CPU.
        stateText.setOnClickListener {
            if (logger != null) return@setOnClickListener
            prefs.edit().putBoolean(KEY_GPU, !useGpu).apply()
            recreate()
        }
        ui.post(guideTick)

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            == PackageManager.PERMISSION_GRANTED) startSession()
        else requestCamera.launch(Manifest.permission.CAMERA)
    }

    private val useGpu get() = prefs.getBoolean(KEY_GPU, true)
    private val signalIdx get() = prefs.getInt(KEY_SIGNAL, 3).coerceIn(0, SIGNALS.size - 1)
    private val signalSource get() = SIGNALS[signalIdx].second
    private val targetFps get() = prefs.getInt(KEY_FPS, 30)
    private val camMode get() = prefs.getInt(KEY_CAM, 1)  // index into CAM_MODES

    private fun setPattern(idx: Int) {
        patternIdx = idx
        guideStart = SystemClock.elapsedRealtime()
        guideButton.text = "Guide: ${patterns[patternIdx].first}"
        if (patterns[patternIdx].second == null) {
            session?.guideSegment = null
            guideText.text = ""
        }
    }

    private fun startSession() {
        val config = BreathConfig(
            signalSource = signalSource,
            gridSmart = SIGNALS[signalIdx].first != "fixed grid",
            // At 60 fps skip the (slow) Farneback comparison when it is not the selected signal.
            computeFarneback = signalSource == SignalSource.FARNEBACK || targetFps <= 30,
        )
        val s = BreathSession(this, this, findViewById<PreviewView>(R.id.preview),
            BreathSession.Options(config = config, useGpuForPose = useGpu, targetFps = targetFps,
                useFrontCamera = camMode < 2, wideAngle = camMode % 2 == 1)) {
            onResult(it)
        }
        overlay.mirror = s.isFrontCamera
        session = s
        s.start()
    }

    // ------------------------------------------------------------ recording

    private fun startRecording() {
        // The guide is the reference label, so always record with one.
        if (patterns[patternIdx].second == null) setPattern(1) else setPattern(patternIdx)
        logger = SessionLogger(this, patterns[patternIdx].first, landscape,
            "${SIGNALS[signalIdx].first.replace(" ", "")}_${targetFps}fps_${CAM_MODES[camMode].replace(" ", "")}")
        recStart = SystemClock.elapsedRealtime()
        Toast.makeText(this, "Recording numbers only. Follow the guide.", Toast.LENGTH_SHORT).show()
    }

    private fun stopRecording() {
        val l = logger ?: return
        logger = null
        l.close()
        recButton.text = "● REC"
        Toast.makeText(this, "Saved ${l.rows} frames:\n${l.file.name}", Toast.LENGTH_LONG).show()
    }

    // -------------------------------------------------------------- results

    private fun onResult(r: BreathResult) {
        val now = SystemClock.elapsedRealtime()
        if (lastResultAt > 0) {
            val inst = 1000f / (now - lastResultAt).coerceAtLeast(1)
            fps = if (fps == 0f) inst else fps * 0.9f + inst * 0.1f
        }
        lastResultAt = now

        stateText.text = if (r.shaking) "MOVING" else r.stateSmooth.name
        stateText.setTextColor(OverlayView.stateColor(r.stateSmooth))
        votesText.text = r.stateVotes.name
        votesText.setTextColor(OverlayView.stateColor(r.stateVotes))
        infoText.text = String.format(
            "signal %+6.1f  level %+6.1f  conf %3.0f%%  votes +%d/-%d\n" +
                "chest %+6.1f  belly %+6.1f  phone %+5.1f,%+5.1f px %s\n" +
                "points %d  flow %+6.1f  breaths %d  rate %4.1f/min\n" +
                "%4.1f fps (cam %s)  engine %4.1f  pose %4.1f ms\n%s",
            r.signal, r.level, r.confidence * 100, r.votesInhale, r.votesExhale,
            r.dyChest, r.dyBelly, r.camDx, r.camDy, if (r.shaking) "SHAKING" else "",
            r.featureCount, r.dyFlow, r.breathCount, r.breathRateBpm,
            fps, session?.fpsRange?.toString() ?: "default", r.engineMs, r.poseMs,
            session?.cameraDescription ?: "")
        overlay.update(r)

        logger?.let {
            it.log(r, currentGuide())
            val sec = (now - recStart) / 1000
            recButton.text = "■ STOP %d:%02d".format(sec / 60, sec % 60)
        }
    }

    private fun currentGuide(): GuideSegment? {
        val steps = patterns[patternIdx].second ?: return null
        var t = ((SystemClock.elapsedRealtime() - guideStart) / 1000.0) % steps.sum()
        for (i in steps.indices) {
            if (steps[i] <= 0) continue
            if (t < steps[i]) return segs[i]
            t -= steps[i]
        }
        return segs.last()
    }

    private val guideTick = object : Runnable {
        override fun run() {
            val steps = patterns[patternIdx].second
            if (steps != null) {
                var t = ((SystemClock.elapsedRealtime() - guideStart) / 1000.0) % steps.sum()
                var seg = segs.last()
                var left = 0.0
                for (i in steps.indices) {
                    if (steps[i] <= 0) continue
                    if (t < steps[i]) { seg = segs[i]; left = steps[i] - t; break }
                    t -= steps[i]
                }
                session?.guideSegment = seg
                val word = when (seg) {
                    GuideSegment.INHALE -> "▲ INHALE"
                    GuideSegment.EXHALE -> "▼ EXHALE"
                    else -> "■ HOLD"
                }
                guideText.text = "Guide: $word  ${"%.1f".format(left)}s"
            }
            ui.postDelayed(this, 50)
        }
    }

    override fun onSaveInstanceState(outState: Bundle) {
        super.onSaveInstanceState(outState)
        outState.putInt(KEY_PATTERN, patternIdx)
        outState.putLong(KEY_GUIDE_START, guideStart)
    }

    override fun onPause() {
        stopRecording()
        super.onPause()
    }

    override fun onDestroy() {
        ui.removeCallbacksAndMessages(null)
        session?.close()
        super.onDestroy()
    }

    private val prefs by lazy { getSharedPreferences("demo", MODE_PRIVATE) }

    companion object {
        private const val KEY_LANDSCAPE = "landscape"
        private const val KEY_GPU = "gpu"
        private const val KEY_SIGNAL = "signal2"
        private val SIGNALS = listOf("flow" to SignalSource.FARNEBACK,
            "points" to SignalSource.FEATURES, "fixed grid" to SignalSource.GRID,
            "smart grid" to SignalSource.GRID)
        private const val KEY_FPS = "fps"
        private const val KEY_CAM = "cam"
        private val CAM_MODES = listOf("front", "front 0.5x", "back", "back 0.5x")
        private const val KEY_PATTERN = "pattern"
        private const val KEY_GUIDE_START = "guideStart"
    }
}
