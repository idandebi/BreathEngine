// JNI bridge: com.breathengine.NativeBridge <-> BreathEngine C API.
#include <android/bitmap.h>
#include <android/log.h>
#include <jni.h>

#include "breath_engine/breath_engine.h"

#define LOG_TAG "BreathEngine"
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

namespace {

// Output layout shared with NativeBridge.kt (RESULT_SIZE floats).
enum {
    R_STATE, R_RAW, R_DY, R_POS, R_NEG, R_ROI_X, R_ROI_Y, R_ROI_W, R_ROI_H,
    R_REANCHORED, R_FLOW_VALID, R_MS, R_STATE_VOTES, R_STATE_SMOOTH, R_LEVEL,
    R_CONFIDENCE, R_POSITION, R_BREATHS, R_RATE, R_SIGNAL, R_SIGNAL_VALID, R_DY_FEAT,
    R_FEAT_COUNT, R_FEAT_DX, R_DY_CHEST, R_DY_BELLY, R_CAM_DX, R_CAM_DY, R_SHAKING, R_POINT_COUNT, R_POINTS,
    R_SIZE = R_POINTS + 2 * BE_MAX_POINTS
};

be_engine* handle(jlong h) { return reinterpret_cast<be_engine*>(h); }

double get_d(JNIEnv* env, jobject o, jclass c, const char* name, double def) {
    jfieldID f = env->GetFieldID(c, name, "D");
    if (!f) { env->ExceptionClear(); return def; }
    return env->GetDoubleField(o, f);
}
int get_i(JNIEnv* env, jobject o, jclass c, const char* name, int def) {
    jfieldID f = env->GetFieldID(c, name, "I");
    if (!f) { env->ExceptionClear(); return def; }
    return env->GetIntField(o, f);
}
int get_z(JNIEnv* env, jobject o, jclass c, const char* name, int def) {
    jfieldID f = env->GetFieldID(c, name, "Z");
    if (!f) { env->ExceptionClear(); return def; }
    return env->GetBooleanField(o, f) ? 1 : 0;
}

be_config read_config(JNIEnv* env, jobject cfg) {
    be_config c = be_default_config();
    if (!cfg) return c;
    jclass k = env->GetObjectClass(cfg);
#define D(field, name) c.field = get_d(env, cfg, k, name, c.field)
#define I(field, name) c.field = get_i(env, cfg, k, name, c.field)
#define Z(field, name) c.field = get_z(env, cfg, k, name, c.field)
    D(shoulder_smoothing, "shoulderSmoothing");
    D(roi_width_factor, "roiWidthFactor");
    D(roi_height_divisor, "roiHeightDivisor");
    D(roi_shrink_factor, "roiShrinkFactor");
    D(roi_stretch_factor, "roiStretchFactor");
    D(roi_zoom, "roiZoom");
    D(reanchor_divisor, "reanchorDivisor");
    Z(low_light, "lowLight");
    D(clahe_clip, "claheClip");
    I(clahe_tiles, "claheTiles");
    D(gamma, "gamma");
    I(low_light_full_frame_passes, "lowLightFullFramePasses");
    Z(low_light_roi_pass, "lowLightRoiPass");
    D(fb_pyr_scale, "fbPyrScale");
    I(fb_levels, "fbLevels");
    I(fb_winsize, "fbWinsize");
    I(fb_iterations, "fbIterations");
    I(fb_poly_n, "fbPolyN");
    D(fb_poly_sigma, "fbPolySigma");
    D(inner_x_factor, "innerXFactor");
    D(inner_y_factor, "innerYFactor");
    D(dy_gain, "dyGain");
    I(window, "window");
    D(sign_threshold, "signThreshold");
    D(threshold_inhale, "thresholdInhale");
    D(threshold_exhale, "thresholdExhale");
    D(hold_penalty, "holdPenalty");
    Z(use_guide_validation, "useGuideValidation");
    D(ref_fps, "refFps");
    D(v2_clip, "v2Clip");
    D(v2_tau, "v2Tau");
    D(v2_on, "v2On");
    D(v2_off, "v2Off");
    D(v2_hold_time, "v2HoldTime");
    D(v2_min_dwell, "v2MinDwell");
    D(v2_position_tau, "v2PositionTau");
    Z(compute_farneback, "computeFarneback");
    Z(compute_features, "computeFeatures");
    I(feat_max_points, "featMaxPoints");
    I(feat_redetect_below, "featRedetectBelow");
    I(feat_min_points, "featMinPoints");
    D(feat_quality, "featQuality");
    D(feat_min_distance, "featMinDistance");
    I(feat_win, "featWin");
    I(feat_levels, "featLevels");
    D(feat_fb_max, "featFbMax");
    D(feat_gain, "featGain");
    D(feat_region_up, "featRegionUp");
    D(feat_region_down, "featRegionDown");
    D(feat_region_side, "featRegionSide");
    I(grid_cols, "gridCols");
    I(grid_rows, "gridRows");
    D(grid_top, "gridTop");
    D(grid_down, "gridDown");
    D(grid_inset, "gridInset");
    D(grid_min_eig, "gridMinEig");
    Z(grid_smart, "gridSmart");
    D(grid_torso_down, "gridTorsoDown");
    Z(cam_compensate, "camCompensate");
    D(cam_shake_px_s, "camShakePxPerSec");
    D(cam_shake_hold, "camShakeHold");
#undef D
#undef I
#undef Z
    env->DeleteLocalRef(k);
    return c;
}

jint write_result(JNIEnv* env, const be_result& r, jfloatArray out) {
    if (!out || env->GetArrayLength(out) < R_SIZE) return BE_ERR_INVALID_ARG;
    jfloat v[R_SIZE];
    v[R_STATE] = r.state;
    v[R_RAW] = r.raw_state;
    v[R_DY] = r.dy_flow;
    v[R_POS] = r.votes_pos;
    v[R_NEG] = r.votes_neg;
    v[R_ROI_X] = r.roi_x;
    v[R_ROI_Y] = r.roi_y;
    v[R_ROI_W] = r.roi_w;
    v[R_ROI_H] = r.roi_h;
    v[R_REANCHORED] = r.reanchored;
    v[R_FLOW_VALID] = r.flow_valid;
    v[R_MS] = r.process_ms;
    v[R_STATE_VOTES] = r.state_votes;
    v[R_STATE_SMOOTH] = r.state_smooth;
    v[R_LEVEL] = r.level;
    v[R_CONFIDENCE] = r.confidence;
    v[R_POSITION] = r.breath_position;
    v[R_BREATHS] = r.breath_count;
    v[R_RATE] = r.breath_rate_bpm;
    v[R_SIGNAL] = r.signal;
    v[R_SIGNAL_VALID] = r.signal_valid;
    v[R_DY_FEAT] = r.dy_features;
    v[R_FEAT_COUNT] = r.feature_count;
    v[R_FEAT_DX] = r.feature_dx;
    v[R_DY_CHEST] = r.dy_chest;
    v[R_DY_BELLY] = r.dy_belly;
    v[R_CAM_DX] = r.cam_dx;
    v[R_CAM_DY] = r.cam_dy;
    v[R_SHAKING] = r.shaking;
    v[R_POINT_COUNT] = r.point_count;
    const int n = r.point_count < 0 ? 0 : (r.point_count > BE_MAX_POINTS ? BE_MAX_POINTS : r.point_count);
    v[R_POINT_COUNT] = n;
    for (int i = 0; i < 2 * n; ++i) v[R_POINTS + i] = r.points[i];
    env->SetFloatArrayRegion(out, 0, R_POINTS + 2 * n, v);
    return BE_OK;
}

}  // namespace

extern "C" {

JNIEXPORT jstring JNICALL
Java_com_breathengine_NativeBridge_nativeVersion(JNIEnv* env, jclass) {
    return env->NewStringUTF(be_version());
}

JNIEXPORT jlong JNICALL
Java_com_breathengine_NativeBridge_nativeCreate(JNIEnv* env, jclass, jobject config,
                                                jint classifier, jint signal_source) {
    be_config c = read_config(env, config);
    c.classifier = classifier;
    c.signal_source = signal_source;
    return reinterpret_cast<jlong>(be_create(&c));
}

JNIEXPORT void JNICALL
Java_com_breathengine_NativeBridge_nativeDestroy(JNIEnv*, jclass, jlong h) {
    be_destroy(handle(h));
}

JNIEXPORT void JNICALL
Java_com_breathengine_NativeBridge_nativeReset(JNIEnv*, jclass, jlong h) {
    be_reset(handle(h));
}

JNIEXPORT void JNICALL
Java_com_breathengine_NativeBridge_nativeSetCameraMotion(JNIEnv*, jclass, jlong h, jfloat dx,
                                                         jfloat dy, jfloat roll) {
    const be_camera_motion m{dx, dy, roll};
    be_set_camera_motion(handle(h), &m);
}

JNIEXPORT jint JNICALL
Java_com_breathengine_NativeBridge_nativeProcessBitmap(
        JNIEnv* env, jclass, jlong h, jobject bitmap, jboolean has_shoulders,
        jfloat lx, jfloat ly, jfloat rx, jfloat ry, jint guide, jdouble timestamp_ms,
        jfloatArray out) {
    if (!h || !bitmap) return BE_ERR_INVALID_ARG;
    AndroidBitmapInfo info;
    if (AndroidBitmap_getInfo(env, bitmap, &info) != ANDROID_BITMAP_RESULT_SUCCESS ||
        info.format != ANDROID_BITMAP_FORMAT_RGBA_8888) {
        LOGE("bitmap must be ARGB_8888");
        return BE_ERR_INVALID_ARG;
    }
    void* px = nullptr;
    if (AndroidBitmap_lockPixels(env, bitmap, &px) != ANDROID_BITMAP_RESULT_SUCCESS) {
        return BE_ERR_INTERNAL;
    }
    be_image img{static_cast<const uint8_t*>(px), static_cast<int32_t>(info.width),
                 static_cast<int32_t>(info.height), static_cast<int32_t>(info.stride),
                 BE_FORMAT_RGBA, 0, nullptr, 0};
    be_shoulders s{lx, ly, rx, ry};
    be_result r{};
    const jint st =
        be_process(handle(h), &img, has_shoulders ? &s : nullptr, guide, timestamp_ms, &r);
    AndroidBitmap_unlockPixels(env, bitmap);
    if (st != BE_OK) return st;
    return write_result(env, r, out);
}

JNIEXPORT jint JNICALL
Java_com_breathengine_NativeBridge_nativeProcessBuffer(
        JNIEnv* env, jclass, jlong h, jobject buffer, jint width, jint height, jint stride,
        jint format, jint rotation, jobject uv_buffer, jint uv_stride, jboolean has_shoulders,
        jfloat lx, jfloat ly, jfloat rx, jfloat ry, jint guide, jdouble timestamp_ms,
        jfloatArray out) {
    if (!h || !buffer) return BE_ERR_INVALID_ARG;
    auto* data = static_cast<const uint8_t*>(env->GetDirectBufferAddress(buffer));
    const uint8_t* uv =
        uv_buffer ? static_cast<const uint8_t*>(env->GetDirectBufferAddress(uv_buffer)) : nullptr;
    if (!data) return BE_ERR_INVALID_ARG;  // must be a direct ByteBuffer
    be_image img{data, width, height, stride, format, rotation, uv, uv_stride};
    be_shoulders s{lx, ly, rx, ry};
    be_result r{};
    const jint st =
        be_process(handle(h), &img, has_shoulders ? &s : nullptr, guide, timestamp_ms, &r);
    if (st != BE_OK) return st;
    return write_result(env, r, out);
}

}  // extern "C"
