#include <cstring>
#include <exception>

#include "breath_engine/breath_engine.h"
#include "breath_engine/engine.hpp"

#ifndef BE_VERSION_STRING
#define BE_VERSION_STRING "0.1.0"
#endif

struct be_engine {
    breath::Engine engine;
    cv::Mat rgb;
    explicit be_engine(const be_config& c) : engine(c) {}
};

extern "C" {

const char* be_version(void) { return BE_VERSION_STRING; }

be_config be_default_config(void) {
    be_config c;
    std::memset(&c, 0, sizeof(c));
    c.struct_size = sizeof(be_config);
    c.shoulder_smoothing = 0.0;
    c.roi_width_factor = 0.7;
    c.roi_height_divisor = 4.0;
    c.roi_shrink_factor = 1.2;
    c.roi_stretch_factor = 0.8;
    c.roi_zoom = 0.85;
    c.reanchor_divisor = 3.0;
    c.low_light = 1;
    c.clahe_clip = 2.0;
    c.clahe_tiles = 8;
    c.gamma = 0.9;
    c.low_light_full_frame_passes = 2;
    c.low_light_roi_pass = 1;
    c.fb_pyr_scale = 0.5;
    c.fb_levels = 3;
    c.fb_winsize = 15;
    c.fb_iterations = 3;
    c.fb_poly_n = 5;
    c.fb_poly_sigma = 1.2;
    c.inner_x_factor = 1.2;
    c.inner_y_factor = 1.95;
    c.dy_gain = 50.0;
    c.window = 11;
    c.sign_threshold = 1.0;
    c.threshold_inhale = 9.0;
    c.threshold_exhale = 9.0;
    c.hold_penalty = 2.0;
    c.use_guide_validation = 1;
    return c;
}

be_engine* be_create(const be_config* config) {
    try {
        be_config c = be_default_config();
        if (config) {
            // Accept configs from older/newer callers: copy the common prefix.
            const size_t n = config->struct_size > 0
                                 ? std::min(sizeof(be_config), static_cast<size_t>(config->struct_size))
                                 : sizeof(be_config);
            std::memcpy(&c, config, n);
            c.struct_size = sizeof(be_config);
        }
        return new be_engine(c);
    } catch (...) {
        return nullptr;
    }
}

void be_destroy(be_engine* e) { delete e; }

void be_reset(be_engine* e) {
    if (e) e->engine.reset();
}

int32_t be_process(be_engine* e, const be_image* image, const be_shoulders* s, int32_t seg,
                   be_result* out) {
    if (!e || !image || !out) return BE_ERR_INVALID_ARG;
    try {
        if (!breath::to_upright_rgb(*image, e->rgb)) return BE_ERR_INVALID_ARG;
        breath::Shoulders sh;
        if (s) sh = breath::Shoulders{{s->left_x, s->left_y}, {s->right_x, s->right_y}};
        const breath::Result r = e->engine.process(e->rgb, s ? &sh : nullptr, seg);
        std::memset(out, 0, sizeof(*out));
        out->state = static_cast<int32_t>(r.state);
        out->raw_state = static_cast<int32_t>(r.raw_state);
        out->dy_flow = static_cast<float>(r.dy_flow);
        out->votes_pos = r.votes_pos;
        out->votes_neg = r.votes_neg;
        if (r.roi) {
            out->roi_x = r.roi->x;
            out->roi_y = r.roi->y;
            out->roi_w = r.roi->w;
            out->roi_h = r.roi->h;
        }
        out->reanchored = r.reanchored;
        out->flow_valid = r.flow_valid;
        out->process_ms = static_cast<float>(r.process_ms);
        return BE_OK;
    } catch (const std::exception&) {
        return BE_ERR_INTERNAL;
    } catch (...) {
        return BE_ERR_INTERNAL;
    }
}

}  // extern "C"
