// C++ API. Mirrors python/breath_engine (the reference implementation).
#pragma once

#include <optional>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "breath_engine.h"

namespace breath {

enum class State : int { Skip = 0, NoDetect = 1, Inhale = 2, Exhale = 3 };

struct Shoulders {
    cv::Point2d left, right;
};

// Camera-induced image motion for one frame (see be_camera_motion).
struct CamMotion {
    double dx = 0, dy = 0, roll = 0;
    // Predicted displacement of a scene point at (x, y) in a w x h image.
    cv::Point2f at(double x, double y, int w, int h) const {
        return {float(dx + roll * (y - 0.5 * h)), float(dy - roll * (x - 0.5 * w))};
    }
};

struct Rect {
    int x = 0, y = 0, w = 0, h = 0;
    bool empty() const { return w <= 0 || h <= 0; }
    bool operator==(const Rect& o) const { return x == o.x && y == o.y && w == o.w && h == o.h; }
};

struct Result {
    State state = State::Skip;
    State raw_state = State::Skip;
    double dy_flow = 0.0;
    int votes_pos = 0, votes_neg = 0;
    std::optional<Rect> roi;
    bool reanchored = false;
    bool flow_valid = false;
    double process_ms = 0.0;

    State state_votes = State::Skip;   // raw, vote classifier
    State state_smooth = State::Skip;  // raw, smooth classifier
    double level = 0.0;
    double confidence = 0.0;
    double breath_position = 0.0;
    int breath_count = 0;
    double breath_rate_bpm = 0.0;

    double signal = 0.0;
    bool signal_valid = false;
    double dy_features = 0.0;
    int feature_count = 0;
    double feature_dx = 0.0;
    double dy_chest = 0.0;
    double dy_belly = 0.0;
    double cam_dx = 0.0, cam_dy = 0.0;
    bool shaking = false;
    std::vector<cv::Point2f> points;
};

// Tracks distinctive points (collar, shoulder outline, shirt folds, texture)
// in a torso region and reports their robust median vertical motion.
class FeatureTracker {
public:
    explicit FeatureTracker(const be_config& cfg) : cfg_(cfg) {}
    void reset();
    struct Out {
        bool valid = false;
        double signal = 0;  // -median(dy) * gain, > 0 = moving up
        double dx = 0;      // median horizontal motion, pixels
        int count = 0;      // valid tracks
    };
    Out update(const cv::Mat& gray, const Shoulders& s, bool body_moved);
    const std::vector<cv::Point2f>& points() const { return pts_; }

private:
    void detect(const cv::Mat& gray, const cv::Rect& region, double shoulder_dist);
    const be_config& cfg_;
    cv::Mat prev_;
    std::vector<cv::Point2f> pts_;
};

// Dense grid over chest and belly. Every frame a fixed grid (relative to the
// shoulders) is placed in the previous frame and tracked into the current one,
// so there is no drift and no re-detection; textureless points are rejected
// by the LK minimum-eigenvalue test and the forward-backward check.
class GridTracker {
public:
    explicit GridTracker(const be_config& cfg) : cfg_(cfg) {}
    void reset() { prev_.release(); pts_.clear(); }
    struct Out {
        bool valid = false;
        double signal = 0, chest = 0, belly = 0, dx = 0;
        int count = 0;
    };
    // `roi`: the grid spans the ROI width, from the ROI top down to
    // grid_torso_down below the shoulders (chest to upper belly). Without a
    // ROI the area is derived from the shoulders (grid_top/grid_down/inset).
    Out update(const cv::Mat& gray, const Shoulders& s, bool body_moved,
               const CamMotion* cam = nullptr, const Rect* area = nullptr);
    const std::vector<cv::Point2f>& points() const { return pts_; }  // valid this frame

private:
    const be_config& cfg_;
    cv::Mat prev_, eig_;
    std::vector<cv::Point2f> pts_;
};

// Exposed for tests.
int unity_round(double v);
Rect clamp_rect(const Rect& r, int cols, int rows);

class RoiTracker {
public:
    explicit RoiTracker(const be_config& cfg) : cfg_(cfg) { reset(); }
    void reset();
    // Returns the ROI (or nullopt) and sets `reanchored`.
    std::optional<Rect> update(const Shoulders& s, int frame_w, int frame_h, bool& reanchored);

private:
    void size_from(double shoulder_dist);
    const be_config& cfg_;
    bool calibrated_ = false;
    double calib_w_ = 0, calib_h_ = 0, ax_ = -1, ay_ = -1;
    Rect prev_safe_;
};

class SignWindowClassifier {
public:
    explicit SignWindowClassifier(const be_config& cfg) : cfg_(cfg) { reset(); }
    void reset();
    State push(double dy_flow);
    State validate(State raw, int guide_segment) const;
    int count_pos() const { return pos_; }
    int count_neg() const { return neg_; }

private:
    const be_config& cfg_;
    std::vector<int> buf_;
    size_t idx_ = 0;
    int pos_ = 0, neg_ = 0;
};

// v2: smoothed, time-based classifier with hysteresis. Works on the dy_flow
// stream only, so it can be replayed offline from logs (python/breath_engine
// /smooth.py is the reference).
class SmoothClassifier {
public:
    explicit SmoothClassifier(const be_config& cfg) : cfg_(cfg) { reset(); }
    void reset();
    // present: shoulders found; valid: dy is a fresh flow measurement.
    State update(double dy, bool valid, bool present, double dt);
    State validate(State raw, int guide_segment) const;
    double level() const { return level_; }
    double confidence() const;
    double position() const { return position_; }
    int breath_count() const { return breaths_; }
    double rate_bpm() const;

private:
    const be_config& cfg_;
    State state_ = State::Skip;
    double level_ = 0, position_ = 0, t_ = 0;
    double since_change_ = 0, quiet_ = 0, missing_ = 0;
    int breaths_ = 0;
    std::vector<double> onsets_;
};

class Engine {
public:
    explicit Engine(const be_config& cfg = be_default_config());
    void reset();
    // frame_rgb: CV_8UC3, RGB, upright. timestamp_ms < 0: assume ref_fps.
    Result process(const cv::Mat& frame_rgb, const Shoulders* shoulders,
                   int guide_segment = BE_GUIDE_NONE, double timestamp_ms = -1.0);
    const be_config& config() const { return cfg_; }
    // Camera motion for the next process() call only.
    void set_camera_motion(const CamMotion& m) { cam_ = m; }
    void clear_camera_motion() { cam_.reset(); }

private:
    cv::Mat low_light(const cv::Mat& rgb);
    void fill_smooth(Result& res, int guide_segment) const;
    std::optional<double> dy_score(const cv::Mat& flow) const;

    be_config cfg_;
    RoiTracker roi_;
    SignWindowClassifier clf_;
    SmoothClassifier smooth_;
    FeatureTracker feat_;
    GridTracker grid_;
    double last_ts_ = -1.0;
    cv::Mat gray_full_;
    cv::Ptr<cv::CLAHE> clahe_;
    cv::Mat lut_;
    cv::Mat prev_gray_;
    std::optional<Shoulders> smoothed_;
    double dy_flow_ = 0.0;
    std::optional<CamMotion> cam_;
    double shake_left_ = 0.0;
    // Scratch buffers reused between frames.
    cv::Mat ycrcb_, luma_, work_, gray_, flow_;
};

// Converts any be_image to an upright RGB Mat (copy).
bool to_upright_rgb(const be_image& img, cv::Mat& out);

}  // namespace breath
