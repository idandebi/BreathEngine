package com.breathengine.sample

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.util.AttributeSet
import android.view.View
import com.breathengine.BreathResult
import com.breathengine.BreathState
import kotlin.math.abs
import kotlin.math.max

/** Draws the analysed region, the shoulders and a dyFlow graph over the preview. */
class OverlayView(context: Context, attrs: AttributeSet?) : View(context, attrs) {

    var mirror = true
    private var last: BreathResult? = null
    private val history = ArrayDeque<Triple<Float, Float, BreathState>>()

    private val roiPaint = Paint().apply { style = Paint.Style.STROKE; strokeWidth = 5f }
    private val dotPaint = Paint().apply { color = Color.YELLOW; isAntiAlias = true }
    private val ptPaint = Paint().apply { color = Color.CYAN; isAntiAlias = true }
    private val linePaint = Paint().apply { strokeWidth = 4f; isAntiAlias = true }
    private val gridPaint = Paint().apply { color = Color.argb(90, 255, 255, 255); strokeWidth = 2f }
    private val bgPaint = Paint().apply { color = Color.argb(160, 0, 0, 0) }
    private val rawPaint = Paint().apply { color = Color.argb(110, 255, 255, 255); strokeWidth = 2f }

    fun update(r: BreathResult) {
        last = r
        history.addLast(Triple(r.signal, r.level, r.stateSmooth))
        while (history.size > 150) history.removeFirst()
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val r = last ?: return
        // PreviewView uses FILL_CENTER: scale to cover, centre, mirror for front camera.
        val s = max(width / r.frameWidth.toFloat(), height / r.frameHeight.toFloat())
        val ox = (width - r.frameWidth * s) / 2f
        val oy = (height - r.frameHeight * s) / 2f
        fun mx(x: Float) = (x * s + ox).let { if (mirror) width - it else it }
        fun my(y: Float) = y * s + oy

        r.roi?.let { roi ->
            roiPaint.color = if (r.reanchored) Color.RED else Color.GREEN
            val x1 = mx(roi.left.toFloat())
            val x2 = mx(roi.right.toFloat())
            canvas.drawRect(minOf(x1, x2), my(roi.top.toFloat()), maxOf(x1, x2),
                my(roi.bottom.toFloat()), roiPaint)
        }
        val pts = r.points
        var i = 0
        while (i + 1 < pts.size) {
            canvas.drawCircle(mx(pts[i]), my(pts[i + 1]), 5f, ptPaint)
            i += 2
        }
        r.shoulders?.let {
            canvas.drawCircle(mx(it.leftX), my(it.leftY), 12f, dotPaint)
            canvas.drawCircle(mx(it.rightX), my(it.rightY), 12f, dotPaint)
        }
        drawGraph(canvas)
    }

    private fun drawGraph(canvas: Canvas) {
        val h = 220f
        val top = height - h - 140f
        canvas.drawRect(0f, top, width.toFloat(), top + h, bgPaint)
        val scale = max(5f, history.maxOfOrNull { abs(it.second) * 1.5f }?.coerceAtMost(40f) ?: 5f)
        val mid = top + h / 2
        fun y(v: Float) = mid - v.coerceIn(-scale, scale) / scale * (h / 2 - 8)
        canvas.drawLine(0f, mid, width.toFloat(), mid, gridPaint)
        val step = width / 149f
        var prev: Triple<Float, Float, BreathState>? = null
        history.forEachIndexed { i, p ->
            prev?.let { q ->
                // thin: raw dy per frame; thick: smoothed level coloured by the v2 state
                canvas.drawLine((i - 1) * step, y(q.first), i * step, y(p.first), rawPaint)
                linePaint.color = stateColor(p.third)
                canvas.drawLine((i - 1) * step, y(q.second), i * step, y(p.second), linePaint)
            }
            prev = p
        }
    }

    companion object {
        fun stateColor(s: BreathState) = when (s) {
            BreathState.INHALE -> Color.rgb(80, 200, 80)
            BreathState.EXHALE -> Color.rgb(60, 140, 230)
            BreathState.NO_DETECT -> Color.rgb(200, 200, 200)
            BreathState.SKIP -> Color.rgb(120, 120, 120)
        }
    }
}
