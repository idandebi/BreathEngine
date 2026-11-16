/*
 * BreathEngine - camera-based breathing detection.
 * Stable C API (used by the Android JNI and iOS bindings).
 *
 * Usage per camera frame:
 *   be_image img = { data, w, h, stride, BE_FORMAT_RGBA, 0 };
 *   be_shoulders s = { lx, ly, rx, ry, 1 };     // pose landmarks 11/12, pixels
 *   be_result r;
 *   be_process(engine, &img, &s, BE_GUIDE_NONE, timestamp_ms, &r);
 *
 * Coordinates: pixels of the UPRIGHT frame (after `rotation` is applied),
 * origin top-left, y down. The shoulders must come from that same frame.
 */
#ifndef BREATH_ENGINE_H
#define BREATH_ENGINE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32) && defined(BE_BUILD_SHARED)
#  define BE_API __declspec(dllexport)
#elif defined(_WIN32) && defined(BE_USE_SHARED)
#  define BE_API __declspec(dllimport)
#elif defined(__GNUC__)
#  define BE_API __attribute__((visibility("default")))
#else
#  define BE_API
#endif

#define BE_API_VERSION 2

typedef enum {
    BE_STATE_SKIP = 0,       /* no usable measurement (no person found) */
    BE_STATE_NO_DETECT = 1,  /* person found, no clear direction        */
    BE_STATE_INHALE = 2,
    BE_STATE_EXHALE = 3
} be_state;

typedef enum {
    BE_GUIDE_NONE = -1,
    BE_GUIDE_INHALE = 0,
    BE_GUIDE_HOLD_TOP = 1,
    BE_GUIDE_EXHALE = 2,
    BE_GUIDE_HOLD_BOTTOM = 3
} be_guide_segment;

typedef enum {
    BE_FORMAT_RGB = 0,
    BE_FORMAT_BGR = 1,
    BE_FORMAT_RGBA = 2,
    BE_FORMAT_BGRA = 3,
    BE_FORMAT_NV21 = 4,   /* Android camera: Y plane + interleaved VU */
    BE_FORMAT_NV12 = 5    /* iOS camera:     Y plane + interleaved UV */
} be_pixel_format;

typedef enum {
    BE_OK = 0,
    BE_ERR_INVALID_ARG = -1,
    BE_ERR_INTERNAL = -2
} be_status;

typedef struct {
    const uint8_t* data;   /* RGB/BGR/RGBA/BGRA pixels, or Y plane for NV21/NV12 */
    int32_t width;         /* as stored in memory (before rotation)             */
    int32_t height;
    int32_t stride;        /* bytes per row of `data`                           */
    int32_t format;        /* be_pixel_format                                   */
    int32_t rotation;      /* clockwise degrees to make it upright: 0/90/180/270 */
    const uint8_t* uv;     /* NV21/NV12 only: interleaved chroma plane           */
    int32_t uv_stride;     /* NV21/NV12 only                                     */
} be_image;

typedef struct {
    float left_x, left_y;    /* landmark 11 */
    float right_x, right_y;  /* landmark 12 */
} be_shoulders;

typedef struct {
    int32_t struct_size;     /* = sizeof(be_config), set by be_default_config */

    double shoulder_smoothing;

    double roi_width_factor;
    double roi_height_divisor;
    double roi_shrink_factor;
    double roi_stretch_factor;
    double roi_zoom;
    double reanchor_divisor;

    int32_t low_light;
    double clahe_clip;
    int32_t clahe_tiles;
    double gamma;
    int32_t low_light_full_frame_passes;
    int32_t low_light_roi_pass;

    double fb_pyr_scale;
    int32_t fb_levels;
    int32_t fb_winsize;
    int32_t fb_iterations;
    int32_t fb_poly_n;
    double fb_poly_sigma;

    double inner_x_factor;
    double inner_y_factor;
    double dy_gain;

    int32_t window;
    double sign_threshold;
    double threshold_inhale;
    double threshold_exhale;
    double hold_penalty;
    int32_t use_guide_validation;

    /* Which classifier drives be_result.state (both always run). */
    int32_t classifier;        /* be_classifier                                 */

    /* Smooth classifier (v2): time-based, fps independent.                    */
    double ref_fps;            /* dy_flow is normalised to this frame rate      */
    double v2_clip;            /* |normalised dy| is clipped to this (outliers) */
    double v2_tau;             /* smoothing time constant, seconds              */
    double v2_on;              /* |level| above this starts inhale / exhale     */
    double v2_off;             /* |level| below this counts as "not moving"     */
    double v2_hold_time;       /* seconds below v2_off before NO_DETECT         */
    double v2_min_dwell;       /* seconds before inhale <-> exhale may switch   */
    double v2_position_tau;    /* drift removal for breath_position, seconds    */

    /* Motion signal that feeds the classifiers.                               */
    int32_t signal_source;     /* be_signal_source                              */
    int32_t compute_farneback; /* run Farneback even if not selected (logging)  */
    int32_t compute_features;  /* run feature tracking even if not selected     */

    /* Feature-point tracking (Shi-Tomasi + pyramidal Lucas-Kanade).            */
    int32_t feat_max_points;      /* points kept in the torso region            */
    int32_t feat_redetect_below;  /* detect new points when fewer remain        */
    int32_t feat_min_points;      /* fewer valid tracks -> no measurement       */
    double feat_quality;          /* goodFeaturesToTrack qualityLevel           */
    double feat_min_distance;     /* min spacing, fraction of shoulder width    */
    int32_t feat_win;             /* LK window size (pixels)                    */
    int32_t feat_levels;          /* LK pyramid levels                          */
    double feat_fb_max;           /* forward-backward error limit (pixels)      */
    double feat_gain;             /* signal = -median(dy px) * gain (= dyFlow)  */
    double feat_region_up;        /* region above shoulder line, x shoulder w   */
    double feat_region_down;      /* region below shoulder line, x shoulder w   */
    double feat_region_side;      /* region beyond each shoulder, x shoulder w  */

    /* Dense grid (BE_SIGNAL_GRID): spread evenly from the ROI top (chest) down */
    /* to the upper belly, ROI width. The grid is placed in                     */
    /* the previous frame every frame and tracked into the current frame.      */
    int32_t grid_cols;            /* points across the torso                    */
    int32_t grid_rows;            /* points from the shoulder line downwards    */
    double grid_top;              /* (no ROI only) starts below the shoulders   */
    double grid_down;             /* grid ends this far below, x shoulder w     */
    double grid_inset;            /* stay inside the shoulders by this, x sw    */
    double grid_min_eig;          /* LK min eigenvalue: rejects textureless pts */
    int32_t grid_smart;           /* 1 = best-textured pixel in each grid cell  */
    double grid_torso_down;       /* with ROI: grid spans ROI top down to this  */
                                  /* far below the shoulders, x sw (upper belly)*/

    /* Camera (phone) motion from the gyroscope, see be_set_camera_motion.     */
    int32_t cam_compensate;       /* subtract camera motion from the signal     */
    double cam_shake_px_s;        /* camera motion above this (px/s) = shaking  */
    double cam_shake_hold;        /* stay "shaking" this long after, seconds    */
} be_config;

typedef enum {
    BE_CLASSIFIER_VOTES = 0,   /* original Unity vote window (default)          */
    BE_CLASSIFIER_SMOOTH = 1   /* v2: smoothed signal + hysteresis              */
} be_classifier;

typedef enum {
    BE_SIGNAL_FARNEBACK = 0,   /* dense flow in the shoulder ROI (Unity)        */
    BE_SIGNAL_FEATURES = 1,    /* median vertical motion of tracked points      */
    BE_SIGNAL_GRID = 2         /* dense point grid over chest and belly         */
} be_signal_source;

#define BE_MAX_POINTS 256

typedef struct {
    int32_t state;           /* be_state of the selected classifier           */
    int32_t raw_state;       /* selected classifier, before guide validation  */
    float dy_flow;           /* motion score, > 0 = chest moving up          */
    int32_t votes_pos;
    int32_t votes_neg;
    int32_t roi_x, roi_y, roi_w, roi_h;  /* analysed region, roi_w=0 if none */
    int32_t reanchored;      /* ROI re-anchored this frame (body moved)       */
    int32_t flow_valid;      /* dy_flow freshly computed this frame           */
    float process_ms;

    int32_t state_votes;     /* raw state of the vote classifier              */
    int32_t state_smooth;    /* raw state of the smooth classifier            */
    float level;             /* smooth classifier: filtered motion            */
    float confidence;        /* 0..1, strength of the current direction       */
    float breath_position;   /* integrated motion: rises on inhale (relative) */
    int32_t breath_count;    /* inhale onsets since reset                     */
    float breath_rate_bpm;   /* breaths per minute, 0 if unknown              */

    float signal;            /* value fed to the classifiers this frame       */
    int32_t signal_valid;    /* signal is a fresh measurement                 */
    float dy_features;       /* feature-tracking signal (same units as dy)    */
    int32_t feature_count;   /* valid tracks this frame                       */
    float feature_dx;        /* median horizontal motion (px), body sway      */
    float dy_chest;          /* grid: upper half (chest) signal               */
    float dy_belly;          /* grid: lower half (belly) signal               */
    float cam_dx, cam_dy;    /* camera-induced image shift this frame (px)    */
    int32_t shaking;         /* phone moving too much: no measurement         */
    int32_t point_count;     /* tracked points returned in `points`           */
    float points[2 * BE_MAX_POINTS];  /* x0,y0,x1,y1,... upright pixels       */
} be_result;

typedef struct be_engine be_engine;

/* Image motion caused by the camera itself between the previous frame and the
 * next one passed to be_process, in upright pixels (e.g. from the gyroscope):
 * a scene point at (x, y) appears to move by
 *     (dx + roll * (y - cy),  dy - roll * (x - cx))
 * where (cx, cy) is the image centre and roll is in radians. */
typedef struct {
    float dx, dy;
    float roll;
} be_camera_motion;

BE_API const char* be_version(void);
BE_API be_config be_default_config(void);
BE_API be_engine* be_create(const be_config* config /* NULL = defaults */);
BE_API void be_destroy(be_engine* engine);
BE_API void be_reset(be_engine* engine);
/* Applies to the next be_process call only. NULL = unknown (no compensation). */
BE_API void be_set_camera_motion(be_engine* engine, const be_camera_motion* motion);

/* shoulders == NULL means "no person detected in this frame".
 * timestamp_ms: frame time in milliseconds (monotonic); < 0 = assume ref_fps. */
BE_API int32_t be_process(be_engine* engine, const be_image* image,
                          const be_shoulders* shoulders, int32_t guide_segment,
                          double timestamp_ms, be_result* out);

#ifdef __cplusplus
}
#endif
#endif /* BREATH_ENGINE_H */
