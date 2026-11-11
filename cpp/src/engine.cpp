#include "breath_engine/engine.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>

#include <opencv2/video/tracking.hpp>

namespace breath {

// ------------------------------------------------------------------ helpers

int unity_round(double v) {
    // Mathf.RoundToInt / Python round(): ties to even.
    return static_cast<int>(std::nearbyint(v));
}

static int unity_clamp(int v, int lo, int hi) {
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

Rect clamp_rect(const Rect& r, int cols, int rows) {
    const int x = unity_clamp(r.x, 0, std::max(0, cols - 1));
    const int y = unity_clamp(r.y, 0, std::max(0, rows - 1));
    const int max_w = std::max(0, cols - x);
    const int max_h = std::max(0, rows - y);
    const int w = unity_clamp(r.w, 1, max_w);
    const int h = unity_clamp(r.h, 1, max_h);
    if (w <= 0 || h <= 0) return Rect{};
    return Rect{x, y, w, h};
}

// --------------------------------------------------------------- RoiTracker

void RoiTracker::reset() {
    calibrated_ = false;
    calib_w_ = calib_h_ = 0;
    ax_ = ay_ = -1;
    prev_safe_ = Rect{};
}

void RoiTracker::size_from(double sd) {
    calib_w_ = sd * cfg_.roi_width_factor;
    calib_h_ = sd * cfg_.roi_width_factor / cfg_.roi_height_divisor;
}

std::optional<Rect> RoiTracker::update(const Shoulders& s, int W, int H, bool& reanchored) {
    const double cx = (s.left.x + s.right.x) / 2.0;
    const double cy = (s.left.y + s.right.y) / 2.0;
    const double sd = std::abs(s.right.x - s.left.x);

    if (!calibrated_) {
        size_from(sd);
        ax_ = cx;
        ay_ = cy;
        calibrated_ = true;
    }
    reanchored = false;
    if (std::abs(cx - ax_) > calib_w_ / cfg_.reanchor_divisor ||
        std::abs(cy - ay_) > calib_h_ / cfg_.reanchor_divisor) {
        ax_ = cx;
        ay_ = cy;
        size_from(sd);
        reanchored = true;
    }

    const int roi_w = unity_round(calib_w_);
    const int r = unity_round(calib_h_ * cfg_.roi_shrink_factor);
    const int h_fixed = unity_round(r * (1.0 + double(cfg_.roi_stretch_factor)));
    const int y_bottom_raw = unity_round(ay_ + r * double(cfg_.roi_stretch_factor));

    // Mirrored space, exactly as Unity computes it (see docs/ALGORITHM.md).
    const int y1 = unity_clamp(H - y_bottom_raw, 0, H - 1);
    const int x1 = unity_clamp(unity_round(ax_ - roi_w / 2.0), 0, W - 1);
    Rect rect{x1, y1, roi_w, h_fixed};

    if (rect.w > 0 && rect.h > 0) {
        const double z = cfg_.roi_zoom;
        const int nw = std::max(1, unity_round(rect.w * z));
        const int nh = std::max(1, unity_round(rect.h * z));
        const int ccx = rect.x + rect.w / 2;
        const int ccy = rect.y + rect.h / 2;
        rect = clamp_rect(Rect{ccx - nw / 2, ccy - nh / 2, nw, nh}, W, H);
        rect.y = H - (rect.y + rect.h);  // back to top-down coordinates
    }

    if (rect.empty()) {
        if (!prev_safe_.empty()) return prev_safe_;
        return std::nullopt;
    }
    prev_safe_ = rect;
    return rect;
}

// ----------------------------------------------------- SignWindowClassifier

void SignWindowClassifier::reset() {
    buf_.assign(std::max(1, cfg_.window), 0);
    idx_ = 0;
    pos_ = neg_ = 0;
}

State SignWindowClassifier::push(double dy) {
    const double thr = cfg_.sign_threshold;
    const int sign = dy > thr ? 1 : (dy < -thr ? -1 : 0);
    const int old = buf_[idx_];
    if (old == 1) --pos_;
    else if (old == -1) --neg_;
    buf_[idx_] = sign;
    idx_ = (idx_ + 1) % buf_.size();
    if (sign == 1) ++pos_;
    else if (sign == -1) ++neg_;

    if (pos_ >= cfg_.threshold_inhale) return State::Inhale;
    if (neg_ >= cfg_.threshold_exhale) return State::Exhale;
    return State::NoDetect;
}

State SignWindowClassifier::validate(State raw, int seg) const {
    if (!cfg_.use_guide_validation || seg == BE_GUIDE_NONE) return raw;
    double th_in = cfg_.threshold_inhale, th_ex = cfg_.threshold_exhale;
    if (seg == BE_GUIDE_HOLD_TOP || seg == BE_GUIDE_HOLD_BOTTOM) {
        th_in += cfg_.hold_penalty;
        th_ex += cfg_.hold_penalty;
    }
    const bool in_ok = pos_ >= th_in;
    const bool ex_ok = neg_ >= th_ex;
    if (seg == BE_GUIDE_INHALE) return in_ok ? State::Inhale : State::NoDetect;
    if (seg == BE_GUIDE_EXHALE) return ex_ok ? State::Exhale : State::NoDetect;
    if (in_ok) return State::Inhale;
    if (ex_ok) return State::Exhale;
    return State::NoDetect;
}

// ------------------------------------------------------------------- Engine

Engine::Engine(const be_config& cfg) : cfg_(cfg), roi_(cfg_), clf_(cfg_) {
    clahe_ = cv::createCLAHE(cfg_.clahe_clip, cv::Size(cfg_.clahe_tiles, cfg_.clahe_tiles));
    lut_.create(1, 256, CV_8UC1);
    for (int i = 0; i < 256; ++i) {
        // Same as the Python reference (float64, round half to even).
        const double v = std::nearbyint(255.0 * std::pow(i / 255.0, cfg_.gamma));
        lut_.at<uchar>(i) = cv::saturate_cast<uchar>(std::clamp(v, 0.0, 255.0));
    }
    reset();
}

void Engine::reset() {
    roi_.reset();
    clf_.reset();
    prev_gray_.release();
    smoothed_.reset();
    dy_flow_ = 0.0;
}

cv::Mat Engine::low_light(const cv::Mat& rgb) {
    cv::cvtColor(rgb, ycrcb_, cv::COLOR_RGB2YCrCb);
    cv::extractChannel(ycrcb_, luma_, 0);
    clahe_->apply(luma_, luma_);
    cv::insertChannel(luma_, ycrcb_, 0);
    cv::Mat out;
    cv::cvtColor(ycrcb_, out, cv::COLOR_YCrCb2RGB);
    cv::LUT(out, lut_, out);
    return out;
}

std::optional<double> Engine::dy_score(const cv::Mat& flow) const {
    if (flow.empty() || flow.channels() < 2) return std::nullopt;
    const int rows = flow.rows, cols = flow.cols;
    const int inner_w = cols / 4, inner_h = rows / 4;
    const int cx = cols / 2, cy = rows / 2;
    auto clampi = [](int v, int lo, int hi) { return std::min(std::max(v, lo), hi); };
    const int xi1 = clampi(static_cast<int>(cx - inner_w * double(cfg_.inner_x_factor)), 0, cols);
    const int xi2 = clampi(static_cast<int>(cx + inner_w * double(cfg_.inner_x_factor)), 0, cols);
    const int yi1 = clampi(static_cast<int>(cy - inner_h * double(cfg_.inner_y_factor)), 0, rows);
    const int yi2 = clampi(static_cast<int>(cy + inner_h * double(cfg_.inner_y_factor)), 0, rows);
    const int sub_w = std::max(0, xi2 - xi1), sub_h = std::max(0, yi2 - yi1);
    if (sub_w < 2 || sub_h < 2) return std::nullopt;

    double total = 0.0, inner = 0.0;
    for (int y = 0; y < rows; ++y) {
        const cv::Vec2f* p = flow.ptr<cv::Vec2f>(y);
        const bool row_in = y >= yi1 && y < yi2;
        for (int x = 0; x < cols; ++x) {
            const double dy = p[x][1];
            total += dy;
            if (row_in && x >= xi1 && x < xi2) inner += dy;
        }
    }
    const long total_px = long(rows) * cols;
    const long inner_px = long(sub_w) * sub_h;
    const long outer_px = total_px - inner_px;
    const double inner_mean = inner_px > 0 ? inner / inner_px : 0.0;
    const double outer_mean = outer_px > 0 ? (total - inner) / outer_px : inner_mean;
    return (-inner_mean - outer_mean) * cfg_.dy_gain;
}

Result Engine::process(const cv::Mat& frame_rgb, const Shoulders* shoulders, int seg) {
    const auto t0 = std::chrono::steady_clock::now();
    Result res;
    if (!shoulders) {
        res.state = res.raw_state = State::Skip;
        res.dy_flow = dy_flow_;
        res.votes_pos = clf_.count_pos();
        res.votes_neg = clf_.count_neg();
        return res;
    }

    Shoulders s = *shoulders;
    const double a = cfg_.shoulder_smoothing;
    if (smoothed_ && a > 0.0) {
        const double t = 1.0 - a;
        auto lerp = [t](cv::Point2d u, cv::Point2d v) { return u + (v - u) * t; };
        s = Shoulders{lerp(smoothed_->left, s.left), lerp(smoothed_->right, s.right)};
    }
    smoothed_ = s;

    bool reanchored = false;
    const auto rect = roi_.update(s, frame_rgb.cols, frame_rgb.rows, reanchored);
    if (rect) {
        cv::Mat img = frame_rgb;
        if (cfg_.low_light)
            for (int i = 0; i < cfg_.low_light_full_frame_passes; ++i) img = low_light(img);
        cv::Mat roi = img(cv::Rect(rect->x, rect->y, rect->w, rect->h));

        if (reanchored) {
            dy_flow_ = 0.0;
            cv::cvtColor(roi, prev_gray_, cv::COLOR_RGB2GRAY);
        } else if (prev_gray_.empty() || prev_gray_.size() != roi.size()) {
            cv::cvtColor(roi, prev_gray_, cv::COLOR_RGB2GRAY);
        } else {
            if (cfg_.low_light && cfg_.low_light_roi_pass) roi = low_light(roi.clone());
            cv::cvtColor(roi, gray_, cv::COLOR_RGB2GRAY);
            cv::calcOpticalFlowFarneback(prev_gray_, gray_, flow_, cfg_.fb_pyr_scale, cfg_.fb_levels,
                                         cfg_.fb_winsize, cfg_.fb_iterations, cfg_.fb_poly_n,
                                         cfg_.fb_poly_sigma, 0);
            if (auto score = dy_score(flow_)) {
                dy_flow_ = *score;
                res.flow_valid = true;
            }
            std::swap(prev_gray_, gray_);
        }
        res.roi = *rect;
    }

    res.raw_state = clf_.push(dy_flow_);
    res.state = clf_.validate(res.raw_state, seg);
    res.dy_flow = dy_flow_;
    res.votes_pos = clf_.count_pos();
    res.votes_neg = clf_.count_neg();
    res.reanchored = reanchored;
    res.process_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    return res;
}

// ------------------------------------------------------- image conversion

bool to_upright_rgb(const be_image& img, cv::Mat& out) {
    if (!img.data || img.width <= 0 || img.height <= 0) return false;
    const int W = img.width, H = img.height;
    cv::Mat rgb;
    switch (img.format) {
        case BE_FORMAT_RGB:
            cv::Mat(H, W, CV_8UC3, const_cast<uint8_t*>(img.data), img.stride).copyTo(rgb);
            break;
        case BE_FORMAT_BGR:
            cv::cvtColor(cv::Mat(H, W, CV_8UC3, const_cast<uint8_t*>(img.data), img.stride), rgb,
                         cv::COLOR_BGR2RGB);
            break;
        case BE_FORMAT_RGBA:
            cv::cvtColor(cv::Mat(H, W, CV_8UC4, const_cast<uint8_t*>(img.data), img.stride), rgb,
                         cv::COLOR_RGBA2RGB);
            break;
        case BE_FORMAT_BGRA:
            cv::cvtColor(cv::Mat(H, W, CV_8UC4, const_cast<uint8_t*>(img.data), img.stride), rgb,
                         cv::COLOR_BGRA2RGB);
            break;
        case BE_FORMAT_NV21:
        case BE_FORMAT_NV12: {
            if (!img.uv || (W & 1) || (H & 1)) return false;
            cv::Mat y(H, W, CV_8UC1, const_cast<uint8_t*>(img.data), img.stride);
            cv::Mat uv(H / 2, W / 2, CV_8UC2, const_cast<uint8_t*>(img.uv), img.uv_stride);
            cv::cvtColorTwoPlane(y, uv, rgb, img.format == BE_FORMAT_NV21 ? cv::COLOR_YUV2RGB_NV21
                                                                          : cv::COLOR_YUV2RGB_NV12);
            break;
        }
        default:
            return false;
    }
    switch (((img.rotation % 360) + 360) % 360) {
        case 0: out = rgb; break;
        case 90: cv::rotate(rgb, out, cv::ROTATE_90_CLOCKWISE); break;
        case 180: cv::rotate(rgb, out, cv::ROTATE_180); break;
        case 270: cv::rotate(rgb, out, cv::ROTATE_90_COUNTERCLOCKWISE); break;
        default: return false;
    }
    return true;
}

}  // namespace breath
