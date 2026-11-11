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

class Engine {
public:
    explicit Engine(const be_config& cfg = be_default_config());
    void reset();
    // frame_rgb: CV_8UC3, RGB, upright.
    Result process(const cv::Mat& frame_rgb, const Shoulders* shoulders,
                   int guide_segment = BE_GUIDE_NONE);
    const be_config& config() const { return cfg_; }

private:
    cv::Mat low_light(const cv::Mat& rgb);
    std::optional<double> dy_score(const cv::Mat& flow) const;

    be_config cfg_;
    RoiTracker roi_;
    SignWindowClassifier clf_;
    cv::Ptr<cv::CLAHE> clahe_;
    cv::Mat lut_;
    cv::Mat prev_gray_;
    std::optional<Shoulders> smoothed_;
    double dy_flow_ = 0.0;
    // Scratch buffers reused between frames.
    cv::Mat ycrcb_, luma_, work_, gray_, flow_;
};

// Converts any be_image to an upright RGB Mat (copy).
bool to_upright_rgb(const be_image& img, cv::Mat& out);

}  // namespace breath
