package com.breathengine

import android.graphics.Rect

/** Breathing state for one camera frame. Codes match the C API. */
enum class BreathState(val code: Int) {
    /** No usable measurement (no person / shoulders not visible). */
    SKIP(0),
    /** Person tracked, but no clear breathing direction right now. */
    NO_DETECT(1),
    INHALE(2),
    EXHALE(3);

    companion object {
        fun fromCode(code: Int): BreathState = entries.firstOrNull { it.code == code } ?: SKIP
    }
}

/**
 * Phase of the breathing guide the user is following. Optional: when supplied,
 * the engine only reports the direction that the guide expects.
 */
enum class GuideSegment(val code: Int) {
    INHALE(0), HOLD_TOP(1), EXHALE(2), HOLD_BOTTOM(3)
}

/** Shoulder landmarks (MediaPipe pose 11 = left, 12 = right) in frame pixels. */
data class Shoulders(val leftX: Float, val leftY: Float, val rightX: Float, val rightY: Float)

data class BreathResult(
    val state: BreathState,
    /** State before guide validation. */
    val rawState: BreathState,
    /** Signed motion score; > 0 means the chest/shoulders moved up. */
    val dyFlow: Float,
    val votesInhale: Int,
    val votesExhale: Int,
    /** Analysed region in frame pixels, or null. */
    val roi: Rect?,
    /** The body moved and the region was re-anchored this frame. */
    val reanchored: Boolean,
    val flowValid: Boolean,
    val engineMs: Float,
    val shoulders: Shoulders?,
    /** Upright frame size the coordinates refer to. */
    val frameWidth: Int,
    val frameHeight: Int,
    val timestampMs: Long,
    /** Total time including pose detection, when produced by [BreathAnalyzer]. */
    val totalMs: Float = engineMs,
    /** Frame conversion (camera image -> upright bitmap), [BreathAnalyzer] only. */
    val convertMs: Float = 0f,
    /** Shoulder detection (MediaPipe), [BreathAnalyzer] only. */
    val poseMs: Float = 0f,
    /** Raw decision of the original vote-window classifier. */
    val stateVotes: BreathState = BreathState.SKIP,
    /** Raw decision of the smooth (v2) classifier. */
    val stateSmooth: BreathState = BreathState.SKIP,
    /** Smooth classifier: filtered motion (> 0 = moving up). */
    val level: Float = 0f,
    /** 0..1, how clearly the current direction is detected (smooth classifier). */
    val confidence: Float = 0f,
    /** Relative breath position: rises while inhaling, falls while exhaling. */
    val breathPosition: Float = 0f,
    /** Inhale onsets since start/reset (smooth classifier). */
    val breathCount: Int = 0,
    /** Breaths per minute, 0 if not enough data yet. */
    val breathRateBpm: Float = 0f,
    /** Value fed to the classifiers this frame (from [BreathConfig.signalSource]). */
    val signal: Float = 0f,
    val signalValid: Boolean = false,
    /** Feature-tracking signal (same units as [dyFlow]). */
    val dyFeatures: Float = 0f,
    /** Valid point tracks this frame. */
    val featureCount: Int = 0,
    /** Median horizontal motion of the points (pixels): body sway. */
    val featureDx: Float = 0f,
    /** GRID: signal of the upper half (chest) and lower half (belly) of the grid. */
    val dyChest: Float = 0f,
    val dyBelly: Float = 0f,
    /** Image shift caused by the phone moving this frame (pixels, from the gyroscope). */
    val camDx: Float = 0f,
    val camDy: Float = 0f,
    /** The phone moved too much to measure breathing this frame. */
    val shaking: Boolean = false,
    /** Tracked points x0,y0,x1,y1,... in frame pixels. */
    val points: FloatArray = FloatArray(0),
)

/** Which motion measurement feeds the classifiers. */
enum class SignalSource(val code: Int) {
    /** Dense Farneback flow in the shoulder ROI (original Unity method). */
    FARNEBACK(0),
    /** Median vertical motion of tracked feature points (collar, folds, outline). */
    FEATURES(1),
    /** Dense grid of points over chest and belly, re-placed every frame. */
    GRID(2),
}

/** Which classifier drives [BreathResult.state]. Both always run. */
enum class BreathClassifier(val code: Int) {
    /** Original Unity vote window (9 of 11 frames). */
    VOTES(0),
    /** v2: smoothed signal with hysteresis; fewer gaps, frame-rate independent. */
    SMOOTH(1),
}

/**
 * Engine parameters. Defaults reproduce the v1 algorithm; see
 * docs/ALGORITHM.md. Field names are read from JNI - do not rename.
 */
data class BreathConfig(
    val shoulderSmoothing: Double = 0.0,
    val roiWidthFactor: Double = 0.7,
    val roiHeightDivisor: Double = 4.0,
    val roiShrinkFactor: Double = 1.2,
    val roiStretchFactor: Double = 0.8,
    val roiZoom: Double = 0.85,
    val reanchorDivisor: Double = 3.0,
    val lowLight: Boolean = true,
    val claheClip: Double = 2.0,
    val claheTiles: Int = 8,
    val gamma: Double = 0.9,
    val lowLightFullFramePasses: Int = 2,
    val lowLightRoiPass: Boolean = true,
    val fbPyrScale: Double = 0.5,
    val fbLevels: Int = 3,
    val fbWinsize: Int = 15,
    val fbIterations: Int = 3,
    val fbPolyN: Int = 5,
    val fbPolySigma: Double = 1.2,
    val innerXFactor: Double = 1.2,
    val innerYFactor: Double = 1.95,
    val dyGain: Double = 50.0,
    val window: Int = 11,
    val signThreshold: Double = 1.0,
    val thresholdInhale: Double = 9.0,
    val thresholdExhale: Double = 9.0,
    val holdPenalty: Double = 2.0,
    val useGuideValidation: Boolean = true,
    val classifier: BreathClassifier = BreathClassifier.VOTES,
    val refFps: Double = 30.0,
    val v2Clip: Double = 30.0,
    val v2Tau: Double = 0.2,
    val v2On: Double = 3.0,
    val v2Off: Double = 1.0,
    val v2HoldTime: Double = 0.4,
    val v2MinDwell: Double = 0.3,
    val v2PositionTau: Double = 8.0,
    val signalSource: SignalSource = SignalSource.FARNEBACK,
    /** Also compute Farneback when FEATURES is selected (for comparison/logging). */
    val computeFarneback: Boolean = true,
    /** Also track feature points when FARNEBACK is selected (cheap). */
    val computeFeatures: Boolean = true,
    val featMaxPoints: Int = 80,
    val featRedetectBelow: Int = 40,
    val featMinPoints: Int = 6,
    val featQuality: Double = 0.005,
    val featMinDistance: Double = 0.04,
    val featWin: Int = 21,
    val featLevels: Int = 3,
    val featFbMax: Double = 0.5,
    val featGain: Double = 100.0,
    val featRegionUp: Double = 0.0,
    val featRegionDown: Double = 0.70,
    val featRegionSide: Double = 0.15,
    /** GRID: points across the torso and from the shoulders downwards. */
    val gridCols: Int = 12,
    val gridRows: Int = 16,
    /** GRID: vertical extent below the shoulder line, x shoulder width. */
    val gridTop: Double = 0.10,
    val gridDown: Double = 1.40,
    /** GRID: margin inside the shoulders, x shoulder width (avoids background). */
    val gridInset: Double = 0.08,
    /** GRID: LK minimum eigenvalue; higher rejects more plain-fabric points. */
    val gridMinEig: Double = 3e-4,
    /** GRID: pick the best-textured pixel in each cell (true) or the cell centre (false). */
    val gridSmart: Boolean = true,
    /** GRID: grid spans from the ROI top down to this far below the shoulders (upper belly), x shoulder width. */
    val gridTorsoDown: Double = 1.00,
    /** Subtract the phone's own motion (gyroscope) from the measurement. */
    val camCompensate: Boolean = true,
    /** Phone motion above this (image pixels per second) = "shaking", no measurement. */
    val camShakePxPerSec: Double = 120.0,
    /** Keep "shaking" this long after the motion stops (seconds). */
    val camShakeHold: Double = 0.2,
)

class BreathEngineException(message: String) : RuntimeException(message)
