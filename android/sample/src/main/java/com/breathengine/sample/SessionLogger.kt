package com.breathengine.sample

import android.content.Context
import com.breathengine.BreathResult
import com.breathengine.GuideSegment
import java.io.BufferedWriter
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/**
 * Writes one CSV row per frame - numbers only, no images. The guide phase is
 * the reference label for tuning (python/tools/analyze_log.py).
 * Files: /sdcard/Android/data/com.breathengine.sample/files/logs/
 */
class SessionLogger(context: Context, pattern: String, landscape: Boolean, tag: String) {
    val file: File
    private val out: BufferedWriter
    var rows = 0
        private set

    init {
        val dir = File(context.getExternalFilesDir(null), "logs").apply { mkdirs() }
        val stamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
        file = File(dir, "breath_${stamp}_${pattern}_${if (landscape) "land" else "port"}_$tag.csv")
        out = file.bufferedWriter()
        out.write("t_ms,guide,dy,flow_valid,reanchored,has_shoulders,state_votes,state_smooth," +
            "level,votes_pos,votes_neg,roi_x,roi_y,roi_w,roi_h,lx,ly,rx,ry,frame_w,frame_h," +
            "engine_ms,pose_ms,signal,signal_valid,dy_features,feature_count,feature_dx,dy_chest,dy_belly,cam_dx,cam_dy,shaking\n")
    }

    fun log(r: BreathResult, guide: GuideSegment?) {
        val s = r.shoulders
        val roi = r.roi
        out.write(buildString {
            append(r.timestampMs).append(',')
            append(guide?.name ?: "").append(',')
            append("%.4f".format(Locale.US, r.dyFlow)).append(',')
            append(if (r.flowValid) 1 else 0).append(',')
            append(if (r.reanchored) 1 else 0).append(',')
            append(if (s != null) 1 else 0).append(',')
            append(r.stateVotes.name).append(',')
            append(r.stateSmooth.name).append(',')
            append("%.4f".format(Locale.US, r.level)).append(',')
            append(r.votesInhale).append(',').append(r.votesExhale).append(',')
            if (roi != null) append("${roi.left},${roi.top},${roi.width()},${roi.height()},")
            else append(",,,,")
            if (s != null) append("%.1f,%.1f,%.1f,%.1f,".format(Locale.US, s.leftX, s.leftY, s.rightX, s.rightY))
            else append(",,,,")
            append(r.frameWidth).append(',').append(r.frameHeight).append(',')
            append("%.2f,%.2f,".format(Locale.US, r.engineMs, r.poseMs))
            append("%.4f,%d,%.4f,%d,%.4f,%.4f,%.4f,%.3f,%.3f,%d".format(Locale.US, r.signal,
                if (r.signalValid) 1 else 0, r.dyFeatures, r.featureCount, r.featureDx,
                r.dyChest, r.dyBelly, r.camDx, r.camDy, if (r.shaking) 1 else 0))
            append('\n')
        })
        rows++
    }

    fun close() = out.close()
}
