package com.breathengine

import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.os.Handler
import android.os.HandlerThread
import android.view.Surface

/**
 * Camera geometry needed to turn phone rotation into image motion.
 * @param focalOverSensor focal length / long side of the sensor (both mm):
 *        focal length in pixels = focalOverSensor * long side of the frame.
 * @param realtimeTimestamps camera timestamps use the elapsedRealtime clock
 *        (SENSOR_INFO_TIMESTAMP_SOURCE_REALTIME), like sensor events.
 */
data class CameraGeometry(
    val isFront: Boolean,
    val focalOverSensor: Double,
    val realtimeTimestamps: Boolean,
)

/** Image motion caused by the phone between two frames (upright pixels, radians). */
data class CameraMotion(val dx: Float, val dy: Float, val roll: Float)

/**
 * Records the gyroscope and integrates the phone's rotation between two
 * camera frames. Rotation is what makes a hand-held image shake; it is
 * converted to the shift it causes in the upright frame.
 */
class GyroTracker(context: Context) : SensorEventListener, AutoCloseable {

    private val sm = context.getSystemService(Context.SENSOR_SERVICE) as SensorManager
    private val gyro: Sensor? = sm.getDefaultSensor(Sensor.TYPE_GYROSCOPE)
    private var thread: HandlerThread? = null

    // Ring buffer of samples: time (ns, elapsedRealtime clock) and rate (rad/s).
    private val n = 2048
    private val t = LongArray(n)
    private val w = Array(3) { FloatArray(n) }
    private var head = 0  // next write index
    private var count = 0
    private val lock = Any()

    val available: Boolean get() = gyro != null

    fun start() {
        val g = gyro ?: return
        if (thread != null) return
        val th = HandlerThread("BreathGyro").also { it.start() }
        thread = th
        // 200 Hz: the highest rate allowed without the HIGH_SAMPLING_RATE permission.
        sm.registerListener(this, g, 5000, Handler(th.looper))
    }

    fun stop() {
        sm.unregisterListener(this)
        thread?.quitSafely()
        thread = null
        synchronized(lock) { count = 0 }
    }

    override fun close() = stop()

    override fun onSensorChanged(e: SensorEvent) {
        synchronized(lock) {
            t[head] = e.timestamp
            w[0][head] = e.values[0]
            w[1][head] = e.values[1]
            w[2][head] = e.values[2]
            head = (head + 1) % n
            if (count < n) count++
        }
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) = Unit

    /**
     * Rotation (radians about device x, y, z) between [fromNs] and [toNs]
     * (elapsedRealtime clock), or null if the gyroscope has no data there.
     * Each sample's rate is held over the interval that ends at its timestamp.
     */
    fun integrate(fromNs: Long, toNs: Long): DoubleArray? {
        if (toNs <= fromNs) return null
        synchronized(lock) {
            if (count < 2) return null
            val first = (head - count + n) % n
            val last = (head - 1 + n) % n
            if (t[first] > fromNs) return null           // history does not reach back
            if (toNs - t[last] > 50_000_000L) return null  // data stale (> 50 ms)
            val out = DoubleArray(3)
            var prevT = t[first]
            for (k in 1 until count) {
                val i = (first + k) % n
                val a = maxOf(prevT, fromNs)
                val b = minOf(t[i], toNs)
                if (b > a) {
                    val dt = (b - a) / 1e9
                    out[0] += w[0][i] * dt; out[1] += w[1][i] * dt; out[2] += w[2][i] * dt
                }
                prevT = t[i]
                if (prevT >= toNs) break
            }
            if (toNs > t[last]) {  // extrapolate the newest rate up to the frame
                val dt = (toNs - maxOf(t[last], fromNs)) / 1e9
                out[0] += w[0][last] * dt; out[1] += w[1][last] * dt; out[2] += w[2][last] * dt
            }
            return out
        }
    }

    companion object {
        /**
         * Image motion (upright frame of [width] x [height]) for a phone
         * rotation [rot] (radians about device x, y, z) with the display at
         * [displayRotation] (Surface.ROTATION_*).
         */
        fun toImageMotion(rot: DoubleArray, geom: CameraGeometry, displayRotation: Int,
                          width: Int, height: Int, gain: Double = 1.0): CameraMotion {
            // Screen right (r) and screen up (u) in device coordinates.
            val (rx, ry) = when (displayRotation) {
                Surface.ROTATION_90 -> 0.0 to 1.0
                Surface.ROTATION_180 -> -1.0 to 0.0
                Surface.ROTATION_270 -> 0.0 to -1.0
                else -> 1.0 to 0.0
            }
            val (ux, uy) = -ry to rx  // u = r rotated 90° counter-clockwise
            val aboutR = rot[0] * rx + rot[1] * ry
            val aboutU = rot[0] * ux + rot[1] * uy
            val f = geom.focalOverSensor * maxOf(width, height) * gain
            // Back camera looks along -z, front along +z (and its frames are not mirrored).
            val dx = f * aboutU
            val dy = if (geom.isFront) -f * aboutR else f * aboutR
            val roll = if (geom.isFront) rot[2] else -rot[2]
            return CameraMotion(dx.toFloat(), dy.toFloat(), roll.toFloat())
        }
    }
}
