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

// --------------------------------------------------------- SmoothClassifier

void SmoothClassifier::reset() {
    state_ = State::Skip;
    level_ = position_ = t_ = 0;
    since_change_ = quiet_ = missing_ = 0;
    breaths_ = 0;
    onsets_.clear();
}

State SmoothClassifier::update(double dy, bool valid, bool present, double dt) {
    dt = std::max(dt, 0.0);
    t_ += dt;
    if (!present) {
        missing_ += dt;
        if (missing_ > 1.0) {
            level_ = 0;
            quiet_ = 0;
        }
        if (state_ != State::Skip) {
            state_ = State::Skip;
            since_change_ = 0;
        } else {
            since_change_ += dt;
        }
        return state_;
    }
    missing_ = 0;
    if (valid && dt > 0) {
        const double v = std::clamp(dy / (dt * cfg_.ref_fps), -cfg_.v2_clip, cfg_.v2_clip);
        const double a = 1.0 - std::exp(-dt / cfg_.v2_tau);
        level_ += a * (v - level_);
        position_ += v * dt * cfg_.ref_fps;
    }
    position_ *= std::exp(-dt / cfg_.v2_position_tau);
    since_change_ += dt;

    const bool up = level_ > cfg_.v2_on;
    const bool down = level_ < -cfg_.v2_on;
    State next = state_;
    if (state_ == State::Inhale) {
        if (down && since_change_ >= cfg_.v2_min_dwell) {
            next = State::Exhale;
        } else if (level_ < cfg_.v2_off) {
            quiet_ += dt;
            if (quiet_ >= cfg_.v2_hold_time) next = State::NoDetect;
        } else {
            quiet_ = 0;
        }
    } else if (state_ == State::Exhale) {
        if (up && since_change_ >= cfg_.v2_min_dwell) {
            next = State::Inhale;
        } else if (level_ > -cfg_.v2_off) {
            quiet_ += dt;
            if (quiet_ >= cfg_.v2_hold_time) next = State::NoDetect;
        } else {
            quiet_ = 0;
        }
    } else {
        next = up ? State::Inhale : (down ? State::Exhale : State::NoDetect);
    }

    if (next != state_) {
        if (next == State::Inhale) {
            ++breaths_;
            onsets_.push_back(t_);
            if (onsets_.size() > 6) onsets_.erase(onsets_.begin());
        }
        state_ = next;
        since_change_ = 0;
        quiet_ = 0;
    }
    return state_;
}

State SmoothClassifier::validate(State raw, int seg) const {
    if (!cfg_.use_guide_validation || seg == BE_GUIDE_NONE) return raw;
    if (seg == BE_GUIDE_INHALE && raw == State::Exhale) return State::NoDetect;
    if (seg == BE_GUIDE_EXHALE && raw == State::Inhale) return State::NoDetect;
    return raw;
}

double SmoothClassifier::confidence() const {
    const double scale = 2.0 * cfg_.v2_on;
    if (state_ == State::Inhale) return std::clamp(level_ / scale, 0.0, 1.0);
    if (state_ == State::Exhale) return std::clamp(-level_ / scale, 0.0, 1.0);
    return 0.0;
}

double SmoothClassifier::rate_bpm() const {
    if (onsets_.size() < 2 || t_ - onsets_.back() > 15.0) return 0.0;
    const double mean = (onsets_.back() - onsets_.front()) / double(onsets_.size() - 1);
    return mean > 0 ? 60.0 / mean : 0.0;
}

// ----------------------------------------------------------- FeatureTracker

void FeatureTracker::reset() {
    prev_.release();
    pts_.clear();
}

static double median_of(std::vector<double> v) {
    if (v.empty()) return 0.0;
    const size_t m = v.size() / 2;
    std::nth_element(v.begin(), v.begin() + m, v.end());
    double med = v[m];
    if (v.size() % 2 == 0) {
        const double lo = *std::max_element(v.begin(), v.begin() + m);
        med = (med + lo) / 2.0;
    }
    return med;
}

void FeatureTracker::detect(const cv::Mat& gray, const cv::Rect& region, double sd) {
    const int want = cfg_.feat_max_points - static_cast<int>(pts_.size());
    if (want <= 0 || region.area() <= 0) return;
    const double min_dist = std::max(3.0, cfg_.feat_min_distance * sd);
    // Mask: inside the region and away from points we already track.
    cv::Mat mask = cv::Mat::zeros(gray.size(), CV_8UC1);
    mask(region).setTo(255);
    for (const auto& p : pts_) cv::circle(mask, p, static_cast<int>(min_dist), 0, -1);
    std::vector<cv::Point2f> found;
    cv::goodFeaturesToTrack(gray, found, want, cfg_.feat_quality, min_dist, mask, 5);
    pts_.insert(pts_.end(), found.begin(), found.end());
}

FeatureTracker::Out FeatureTracker::update(const cv::Mat& gray, const Shoulders& s,
                                           bool body_moved) {
    Out out;
    const double sd = std::abs(s.right.x - s.left.x);
    if (sd < 20) {
        reset();
        return out;
    }
    const double x0 = std::min(s.left.x, s.right.x) - cfg_.feat_region_side * sd;
    const double x1 = std::max(s.left.x, s.right.x) + cfg_.feat_region_side * sd;
    const double y0 = std::min(s.left.y, s.right.y) - cfg_.feat_region_up * sd;
    const double y1 = std::max(s.left.y, s.right.y) + cfg_.feat_region_down * sd;
    const cv::Rect region = cv::Rect(cv::Point(int(x0), int(y0)), cv::Point(int(x1), int(y1))) &
                            cv::Rect(0, 0, gray.cols, gray.rows);

    if (prev_.empty() || prev_.size() != gray.size() || pts_.empty()) {
        pts_.clear();
        detect(gray, region, sd);
        gray.copyTo(prev_);
        return out;
    }

    std::vector<cv::Point2f> next, back;
    std::vector<uchar> st1, st2;
    std::vector<float> err;
    const cv::Size win(cfg_.feat_win, cfg_.feat_win);
    const cv::TermCriteria crit(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 30, 0.01);
    cv::calcOpticalFlowPyrLK(prev_, gray, pts_, next, st1, err, win, cfg_.feat_levels, crit);
    cv::calcOpticalFlowPyrLK(gray, prev_, next, back, st2, err, win, cfg_.feat_levels, crit);

    std::vector<cv::Point2f> kept;
    std::vector<double> dys, dxs;
    kept.reserve(pts_.size());
    const cv::Rect keep_area(region.x - region.width / 4, region.y - region.height / 4,
                             region.width * 3 / 2, region.height * 3 / 2);
    for (size_t i = 0; i < pts_.size(); ++i) {
        if (!st1[i] || !st2[i]) continue;
        const cv::Point2f d = back[i] - pts_[i];
        if (std::hypot(d.x, d.y) > cfg_.feat_fb_max) continue;
        if (!keep_area.contains(cv::Point(int(next[i].x), int(next[i].y)))) continue;
        kept.push_back(next[i]);
        dys.push_back(next[i].y - pts_[i].y);
        dxs.push_back(next[i].x - pts_[i].x);
    }

    // Robust median: drop tracks far from the consensus (MAD), take median again.
    if (static_cast<int>(dys.size()) >= cfg_.feat_min_points && !body_moved) {
        const double med = median_of(dys);
        std::vector<double> dev(dys.size());
        for (size_t i = 0; i < dys.size(); ++i) dev[i] = std::abs(dys[i] - med);
        const double mad = median_of(dev) * 1.4826;
        const double lim = 3.0 * mad + 0.05;
        std::vector<double> inl;
        for (double v : dys)
            if (std::abs(v - med) <= lim) inl.push_back(v);
        if (static_cast<int>(inl.size()) >= cfg_.feat_min_points) {
            out.valid = true;
            out.signal = -median_of(inl) * cfg_.feat_gain;
            out.dx = median_of(dxs);
            out.count = static_cast<int>(inl.size());
        }
    }

    pts_ = std::move(kept);
    if (body_moved) pts_.clear();  // body moved: start fresh on the new position
    if (static_cast<int>(pts_.size()) < cfg_.feat_redetect_below) detect(gray, region, sd);
    gray.copyTo(prev_);
    return out;
}

// -------------------------------------------------------------- GridTracker

// Median after dropping values far from the consensus (3 x MAD).
static bool robust_median(const std::vector<double>& v, int min_n, double& med_out, int* n_out) {
    if (static_cast<int>(v.size()) < min_n) return false;
    const double med = median_of(v);
    std::vector<double> dev(v.size());
    for (size_t i = 0; i < v.size(); ++i) dev[i] = std::abs(v[i] - med);
    const double lim = 3.0 * median_of(dev) * 1.4826 + 0.05;
    std::vector<double> inl;
    for (double x : v)
        if (std::abs(x - med) <= lim) inl.push_back(x);
    if (static_cast<int>(inl.size()) < min_n) return false;
    med_out = median_of(inl);
    if (n_out) *n_out = static_cast<int>(inl.size());
    return true;
}

GridTracker::Out GridTracker::update(const cv::Mat& gray, const Shoulders& s, bool body_moved,
                                     const CamMotion* cam, const Rect* roi) {
    Out out;
    pts_.clear();
    const double sd = std::abs(s.right.x - s.left.x);
    const bool have_prev = !prev_.empty() && prev_.size() == gray.size();
    if (sd < 20 || !have_prev || body_moved || cfg_.grid_cols < 1 || cfg_.grid_rows < 1) {
        gray.copyTo(prev_);
        return out;
    }
    // Areas to fill: {x0, x1, y0, y1, first row, number of rows}. Rows below
    // grid_rows / 2 count as chest, the rest as belly.
    struct Band { double x0, x1, y0, y1; int row0, rows; };
    std::vector<Band> bands;
    const double ys = 0.5 * (s.left.y + s.right.y);
    if (roi && !roi->empty()) {
        // One even grid from the ROI top (chest) down to the upper belly.
        const double y1 = std::max<double>(roi->y + roi->h - 1, ys + cfg_.grid_torso_down * sd);
        bands.push_back({double(roi->x), double(roi->x + roi->w - 1), double(roi->y), y1, 0,
                         cfg_.grid_rows});
    } else {
        bands.push_back({std::min(s.left.x, s.right.x) + cfg_.grid_inset * sd,
                         std::max(s.left.x, s.right.x) - cfg_.grid_inset * sd,
                         ys + cfg_.grid_top * sd, ys + cfg_.grid_down * sd, 0, cfg_.grid_rows});
    }
    const float m = cfg_.feat_win * 0.5f;  // keep the LK window inside the image
    const int cols = cfg_.grid_cols;
    std::vector<cv::Point2f> grid;
    std::vector<int> row_of;
    for (const Band& bd : bands) {
        const int rows = bd.rows;
        if (rows < 1) continue;
        if (cfg_.grid_smart) {
            // One point per cell: the most trackable pixel (largest minimum
            // eigenvalue of the gradient matrix) inside that cell of the
            // previous frame. Even coverage, best texture available per cell.
            // Note: cv::Rect(Point, Point) would swap inverted corners, so
            // build it explicitly; the torso may be partly outside the frame.
            const int ax0 = int(std::ceil(std::max<double>(bd.x0, m)));
            const int ay0 = int(std::ceil(std::max<double>(bd.y0, m)));
            const int ax1 = int(std::min<double>(bd.x1, gray.cols - 1 - m));
            const int ay1 = int(std::min<double>(bd.y1, gray.rows - 1 - m));
            const cv::Rect area(ax0, ay0, std::max(0, ax1 - ax0), std::max(0, ay1 - ay0));
            if (area.width < cols || area.height < rows) continue;
            // Score over the same window LK tracks with, so a cell near a
            // single mark (common in a small ROI) still yields a usable point.
            const int block = std::max(3, cfg_.feat_win | 1);
            cv::cornerMinEigenVal(prev_(area), eig_, block, 3);
            for (int r = 0; r < rows; ++r) {
                const int cy0 = area.height * r / rows;
                const int cy1 = area.height * (r + 1) / rows;
                for (int c = 0; c < cols; ++c) {
                    const int cx0 = area.width * c / cols;
                    const int cx1 = area.width * (c + 1) / cols;
                    if (cx1 <= cx0 || cy1 <= cy0) continue;
                    double best = 0;
                    cv::Point at;
                    cv::minMaxLoc(eig_(cv::Rect(cx0, cy0, cx1 - cx0, cy1 - cy0)), nullptr, &best,
                                  nullptr, &at);
                    if (best <= 0) continue;
                    grid.emplace_back(float(area.x + cx0 + at.x), float(area.y + cy0 + at.y));
                    row_of.push_back(bd.row0 + r);
                }
            }
        } else {
            for (int r = 0; r < rows; ++r) {
                const double y = rows == 1 ? 0.5 * (bd.y0 + bd.y1) : bd.y0 + (bd.y1 - bd.y0) * r / (rows - 1);
                for (int c = 0; c < cols; ++c) {
                    const double x = cols == 1 ? 0.5 * (bd.x0 + bd.x1)
                                               : bd.x0 + (bd.x1 - bd.x0) * c / (cols - 1);
                    if (x < m || y < m || x > gray.cols - 1 - m || y > gray.rows - 1 - m) continue;
                    grid.emplace_back(float(x), float(y));
                    row_of.push_back(bd.row0 + r);
                }
            }
        }
    }
    if (grid.empty()) {
        gray.copyTo(prev_);
        return out;
    }

    // Camera motion: start the search where the point should be, and subtract
    // that part from the measured motion (only the body's own motion remains).
    std::vector<cv::Point2f> cam_d(grid.size(), cv::Point2f(0, 0));
    if (cam)
        for (size_t i = 0; i < grid.size(); ++i)
            cam_d[i] = cam->at(grid[i].x, grid[i].y, gray.cols, gray.rows);
    std::vector<cv::Point2f> next(grid.size()), back(grid);
    for (size_t i = 0; i < grid.size(); ++i) next[i] = grid[i] + cam_d[i];
    std::vector<uchar> st1, st2;
    std::vector<float> err;
    const cv::Size win(cfg_.feat_win, cfg_.feat_win);
    const cv::TermCriteria crit(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 30, 0.01);
    const int flags = cam ? cv::OPTFLOW_USE_INITIAL_FLOW : 0;
    cv::calcOpticalFlowPyrLK(prev_, gray, grid, next, st1, err, win, cfg_.feat_levels, crit, flags,
                             cfg_.grid_min_eig);
    cv::calcOpticalFlowPyrLK(gray, prev_, next, back, st2, err, win, cfg_.feat_levels, crit, flags,
                             cfg_.grid_min_eig);

    std::vector<double> all, chest, belly, dxs;
    const int half = cfg_.grid_rows / 2;  // upper rows = chest, lower = belly
    for (size_t i = 0; i < grid.size(); ++i) {
        if (!st1[i] || !st2[i]) continue;
        const cv::Point2f d = back[i] - grid[i];
        if (std::hypot(d.x, d.y) > cfg_.feat_fb_max) continue;
        const double dy = next[i].y - grid[i].y - cam_d[i].y;
        all.push_back(dy);
        (row_of[i] < half ? chest : belly).push_back(dy);
        dxs.push_back(next[i].x - grid[i].x - cam_d[i].x);
        pts_.push_back(next[i]);
    }
    double med = 0;
    if (robust_median(all, cfg_.feat_min_points, med, &out.count)) {
        out.valid = true;
        out.signal = -med * cfg_.feat_gain;
        out.dx = median_of(dxs);
        double c = 0, b = 0;
        if (robust_median(chest, std::max(3, cfg_.feat_min_points / 2), c, nullptr))
            out.chest = -c * cfg_.feat_gain;
        if (robust_median(belly, std::max(3, cfg_.feat_min_points / 2), b, nullptr))
            out.belly = -b * cfg_.feat_gain;
    }
    gray.copyTo(prev_);
    return out;
}

// ------------------------------------------------------------------- Engine

Engine::Engine(const be_config& cfg)
    : cfg_(cfg), roi_(cfg_), clf_(cfg_), smooth_(cfg_), feat_(cfg_), grid_(cfg_) {
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
    smooth_.reset();
    feat_.reset();
    grid_.reset();
    cam_.reset();
    shake_left_ = 0.0;
    last_ts_ = -1.0;
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

void Engine::fill_smooth(Result& res, int seg) const {
    res.level = smooth_.level();
    res.confidence = smooth_.confidence();
    res.breath_position = smooth_.position();
    res.breath_count = smooth_.breath_count();
    res.breath_rate_bpm = smooth_.rate_bpm();
    if (cfg_.classifier == BE_CLASSIFIER_SMOOTH) {
        res.raw_state = res.state_smooth;
        res.state = smooth_.validate(res.state_smooth, seg);
    }
}

Result Engine::process(const cv::Mat& frame_rgb, const Shoulders* shoulders, int seg,
                       double timestamp_ms) {
    const auto t0 = std::chrono::steady_clock::now();
    double dt = 1.0 / cfg_.ref_fps;
    if (timestamp_ms >= 0) {
        if (last_ts_ >= 0) dt = std::max(0.0, (timestamp_ms - last_ts_) / 1000.0);
        last_ts_ = timestamp_ms;
    }

    Result res;
    if (!shoulders) {
        res.state = res.raw_state = res.state_votes = State::Skip;
        res.state_smooth = smooth_.update(0.0, false, false, dt);
        grid_.reset();
        cam_.reset();
        res.dy_flow = dy_flow_;
        res.votes_pos = clf_.count_pos();
        res.votes_neg = clf_.count_neg();
        fill_smooth(res, seg);
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
    const bool use_grid = cfg_.signal_source == BE_SIGNAL_GRID;
    const bool use_features = cfg_.signal_source == BE_SIGNAL_FEATURES || use_grid;
    const bool run_farneback = !use_features || cfg_.compute_farneback;

    // Phone motion (gyroscope). Small motion is subtracted; large motion makes
    // the measurement unreliable, so it is reported as "shaking" instead.
    const std::optional<CamMotion> cam = cam_;
    cam_.reset();
    const CamMotion* comp = (cam && cfg_.cam_compensate) ? &*cam : nullptr;
    if (cam) {
        res.cam_dx = cam->dx;
        res.cam_dy = cam->dy;
        // Largest shift over the image (translation + roll at the corners).
        const double r = std::abs(cam->roll) * 0.5 * std::hypot(frame_rgb.cols, frame_rgb.rows);
        const double px = std::hypot(cam->dx, cam->dy) + r;
        const double fdt = dt > 0 ? dt : 1.0 / cfg_.ref_fps;
        if (px / fdt > cfg_.cam_shake_px_s) shake_left_ = cfg_.cam_shake_hold;
    }
    res.shaking = shake_left_ > 0;
    shake_left_ = std::max(0.0, shake_left_ - dt);
    if (rect && !run_farneback) res.roi = *rect;
    if (rect && run_farneback) {
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
                // A uniform shift d in the flow adds -2 * d * gain to the score.
                if (comp) {
                    const auto c = comp->at(rect->x + 0.5 * rect->w, rect->y + 0.5 * rect->h,
                                            frame_rgb.cols, frame_rgb.rows);
                    dy_flow_ += 2.0 * c.y * cfg_.dy_gain;
                }
                res.flow_valid = true;
            }
            std::swap(prev_gray_, gray_);
        }
        res.roi = *rect;
    }

    if (use_grid) {
        cv::cvtColor(frame_rgb, gray_full_, cv::COLOR_RGB2GRAY);
        const auto g = grid_.update(gray_full_, s, reanchored, comp, rect ? &*rect : nullptr);
        res.dy_features = g.valid ? g.signal : 0.0;
        res.dy_chest = g.chest;
        res.dy_belly = g.belly;
        res.feature_count = g.count;
        res.feature_dx = g.dx;
        res.points = grid_.points();
        res.signal = res.dy_features;
        res.signal_valid = g.valid;
    } else if (use_features || cfg_.compute_features) {
        cv::cvtColor(frame_rgb, gray_full_, cv::COLOR_RGB2GRAY);
        const auto f = feat_.update(gray_full_, s, reanchored);
        res.dy_features = f.valid ? f.signal : 0.0;
        res.feature_count = f.count;
        res.feature_dx = f.dx;
        res.points = feat_.points();
        if (use_features) {
            res.signal = res.dy_features;
            res.signal_valid = f.valid;
        }
    }
    if (!use_features) {
        // Unity behaviour: the vote window also counts a stale value.
        res.signal = dy_flow_;
        res.signal_valid = res.flow_valid;
    }
    if (res.shaking) {
        res.signal = 0.0;
        res.signal_valid = false;
    }

    res.raw_state = res.state_votes = clf_.push(res.signal);
    res.state = clf_.validate(res.raw_state, seg);
    res.state_smooth = smooth_.update(res.signal, res.signal_valid, true, dt);
    fill_smooth(res, seg);
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
