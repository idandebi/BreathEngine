/*
 * BreathEngine - camera-based breathing detection.
 * Stable C API (used by the Android JNI and iOS bindings).
 *
 * Usage per camera frame:
 *   be_image img = { data, w, h, stride, BE_FORMAT_RGBA, 0 };
 *   be_shoulders s = { lx, ly, rx, ry, 1 };     // pose landmarks 11/12, pixels
 *   be_result r;
 *   be_process(engine, &img, &s, BE_GUIDE_NONE, &r);
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

#define BE_API_VERSION 1

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
} be_config;

typedef struct {
    int32_t state;           /* be_state                                     */
    int32_t raw_state;       /* before guide validation                      */
    float dy_flow;           /* motion score, > 0 = chest moving up          */
    int32_t votes_pos;
    int32_t votes_neg;
    int32_t roi_x, roi_y, roi_w, roi_h;  /* analysed region, roi_w=0 if none */
    int32_t reanchored;      /* ROI re-anchored this frame (body moved)       */
    int32_t flow_valid;      /* dy_flow freshly computed this frame           */
    float process_ms;
} be_result;

typedef struct be_engine be_engine;

BE_API const char* be_version(void);
BE_API be_config be_default_config(void);
BE_API be_engine* be_create(const be_config* config /* NULL = defaults */);
BE_API void be_destroy(be_engine* engine);
BE_API void be_reset(be_engine* engine);

/* shoulders == NULL means "no person detected in this frame". */
BE_API int32_t be_process(be_engine* engine, const be_image* image,
                          const be_shoulders* shoulders, int32_t guide_segment,
                          be_result* out);

#ifdef __cplusplus
}
#endif
#endif /* BREATH_ENGINE_H */
