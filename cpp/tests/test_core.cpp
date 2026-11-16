// Minimal self-contained tests (no framework). Mirrors python/tests.
#include <cmath>
#include <cstdio>
#include <functional>
#include <random>
#include <string>
#include <vector>

#include <opencv2/imgproc.hpp>

#include "breath_engine/breath_engine.h"
#include "breath_engine/engine.hpp"

using namespace breath;

static int g_failed = 0, g_checks = 0;
#define CHECK(cond)                                                              \
    do {                                                                         \
        ++g_checks;                                                              \
        if (!(cond)) {                                                           \
            ++g_failed;                                                          \
            std::printf("  FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond);        \
        }                                                                        \
    } while (0)

static const int W = 640, H = 480;
static const Shoulders kShoulders{{220.0, 240.0}, {420.0, 240.0}};

static cv::Mat textured_canvas(int rows, int cols, unsigned seed = 0) {
    cv::Mat noise(rows, cols, CV_8UC1);
    std::mt19937 rng(seed);
    for (int y = 0; y < rows; ++y)
        for (int x = 0; x < cols; ++x) noise.at<uchar>(y, x) = static_cast<uchar>(rng() & 0xFF);
    cv::GaussianBlur(noise, noise, cv::Size(0, 0), 2.0);
    cv::normalize(noise, noise, 0, 255, cv::NORM_MINMAX);
    cv::Mat rgb;
    cv::cvtColor(noise, rgb, cv::COLOR_GRAY2RGB);
    return rgb;
}

static cv::Mat shifted(const cv::Mat& canvas, double off) {
    cv::Mat m = (cv::Mat_<double>(2, 3) << 1, 0, 0, 0, 1, -(100 + off));
    cv::Mat out;
    cv::warpAffine(canvas, out, m, cv::Size(W, H), cv::INTER_LINEAR);
    return out;
}

static void test_classifier() {
    be_config cfg = be_default_config();
    SignWindowClassifier clf(cfg);
    State st = State::NoDetect;
    for (int i = 0; i < 8; ++i) CHECK(clf.push(5.0) == State::NoDetect);
    st = clf.push(5.0);
    CHECK(st == State::Inhale);
    for (int i = 0; i < 11; ++i) clf.push(-5.0);
    CHECK(clf.count_neg() == 11 && clf.count_pos() == 0);
    CHECK(clf.push(0.5) == State::Exhale);

    SignWindowClassifier c2(cfg);
    for (int i = 0; i < 11; ++i) c2.push(1.0);  // strict threshold
    CHECK(c2.count_pos() == 0);

    SignWindowClassifier g(cfg);
    for (int i = 0; i < 10; ++i) g.push(5.0);
    CHECK(g.validate(State::Inhale, BE_GUIDE_INHALE) == State::Inhale);
    CHECK(g.validate(State::Inhale, BE_GUIDE_EXHALE) == State::NoDetect);
    CHECK(g.validate(State::Inhale, BE_GUIDE_HOLD_TOP) == State::NoDetect);
    g.push(5.0);
    CHECK(g.validate(State::Inhale, BE_GUIDE_HOLD_TOP) == State::Inhale);
    CHECK(g.validate(State::Inhale, BE_GUIDE_NONE) == State::Inhale);
}

static void test_roi() {
    be_config cfg = be_default_config();
    RoiTracker tr(cfg);
    bool re = true;
    auto r = tr.update(kShoulders, W, H, re);
    CHECK(r.has_value() && !re);
    CHECK(r->w == 119 && r->h == 65);
    CHECK(r->x == 320 - 59);
    const double cy = r->y + r->h / 2.0;
    CHECK(cy >= 230 && cy <= 240);

    tr.update(Shoulders{{220, 250}, {420, 250}}, W, H, re);
    CHECK(!re);
    tr.update(Shoulders{{220, 252}, {420, 252}}, W, H, re);
    CHECK(re);

    CHECK((clamp_rect(Rect{-5, -5, 20, 20}, 10, 10) == Rect{0, 0, 10, 10}));
    CHECK((clamp_rect(Rect{8, 8, 20, 20}, 10, 10) == Rect{8, 8, 2, 2}));
    CHECK(unity_round(2.5) == 2 && unity_round(3.5) == 4 && unity_round(-0.5) == 0);
}

static std::vector<Result> run_offsets(Engine& e, const std::vector<double>& offs) {
    const cv::Mat canvas = textured_canvas(H + 200, W);
    std::vector<Result> out;
    for (double o : offs) out.push_back(e.process(shifted(canvas, o), &kShoulders));
    return out;
}

static void test_engine_motion() {
    {
        Engine e;
        CHECK(e.process(cv::Mat::zeros(H, W, CV_8UC3), nullptr).state == State::Skip);
    }
    {
        Engine e;
        std::vector<double> up;
        for (int i = 0; i < 15; ++i) up.push_back(i);
        auto r = run_offsets(e, up);
        CHECK(r[1].flow_valid && r[1].dy_flow > 1.0);
        CHECK(r.back().state == State::Inhale);
    }
    {
        Engine e;
        std::vector<double> down;
        for (int i = 0; i < 15; ++i) down.push_back(30 - i);
        CHECK(run_offsets(e, down).back().state == State::Exhale);
    }
    {
        Engine e;
        auto r = run_offsets(e, std::vector<double>(15, 0.0));
        CHECK(std::abs(r.back().dy_flow) < 1.0 && r.back().state == State::NoDetect);
    }
}

static void test_c_api_formats_and_rotation() {
    CHECK(std::string(be_version()).size() > 0);
    const cv::Mat canvas = textured_canvas(H + 200, W);
    auto run = [&](int format, int rotation) {
        be_engine* e = be_create(nullptr);
        be_result res{};
        std::vector<int> states;
        for (int i = 0; i < 15; ++i) {
            cv::Mat rgb = shifted(canvas, i);  // upright frame
            // Store it the way a camera would deliver it (rotated the other way).
            cv::Mat stored = rgb;
            if (rotation == 90) cv::rotate(rgb, stored, cv::ROTATE_90_COUNTERCLOCKWISE);
            if (rotation == 270) cv::rotate(rgb, stored, cv::ROTATE_90_CLOCKWISE);
            cv::Mat px;
            if (format == BE_FORMAT_BGR) cv::cvtColor(stored, px, cv::COLOR_RGB2BGR);
            else if (format == BE_FORMAT_RGBA) cv::cvtColor(stored, px, cv::COLOR_RGB2RGBA);
            else if (format == BE_FORMAT_BGRA) cv::cvtColor(stored, px, cv::COLOR_RGB2BGRA);
            else px = stored;
            be_image img{px.data, px.cols, px.rows, static_cast<int32_t>(px.step), format, rotation,
                         nullptr, 0};
            be_shoulders s{220, 240, 420, 240};
            CHECK(be_process(e, &img, &s, BE_GUIDE_NONE, -1.0, &res) == BE_OK);
            states.push_back(res.state);
        }
        be_destroy(e);
        return states;
    };
    const auto ref = run(BE_FORMAT_RGB, 0);
    CHECK(ref.back() == BE_STATE_INHALE);
    CHECK(run(BE_FORMAT_BGR, 0) == ref);
    CHECK(run(BE_FORMAT_RGBA, 0) == ref);
    CHECK(run(BE_FORMAT_BGRA, 0) == ref);
    CHECK(run(BE_FORMAT_RGB, 90) == ref);
    CHECK(run(BE_FORMAT_RGB, 270) == ref);

    // NV21: Y + VU planes from an RGB frame -> engine still detects inhale.
    be_engine* e = be_create(nullptr);
    be_result res{};
    for (int i = 0; i < 15; ++i) {
        cv::Mat rgb = shifted(canvas, i), i420;
        cv::cvtColor(rgb, i420, cv::COLOR_RGB2YUV_I420);
        cv::Mat y = i420.rowRange(0, H);
        const uchar* u = i420.ptr(H);
        const uchar* v = u + (W / 2) * (H / 2);
        std::vector<uchar> vu(W * H / 2);
        for (int k = 0; k < W * H / 4; ++k) {
            vu[2 * k] = v[k];
            vu[2 * k + 1] = u[k];
        }
        be_image img{y.data, W, H, W, BE_FORMAT_NV21, 0, vu.data(), W};
        be_shoulders s{220, 240, 420, 240};
        CHECK(be_process(e, &img, &s, BE_GUIDE_NONE, -1.0, &res) == BE_OK);
    }
    CHECK(res.state == BE_STATE_INHALE);
    be_image bad{nullptr, 0, 0, 0, BE_FORMAT_RGB, 0, nullptr, 0};
    CHECK(be_process(e, &bad, nullptr, BE_GUIDE_NONE, -1.0, &res) == BE_ERR_INVALID_ARG);
    be_destroy(e);
}

static void test_smooth_classifier() {
    be_config cfg = be_default_config();
    SmoothClassifier sc(cfg);
    const double dt = 1.0 / 30;
    State st = State::Skip;
    for (int i = 0; i < 30; ++i) st = sc.update(8.0, true, true, dt);
    CHECK(st == State::Inhale);
    for (double v : {-2.0, 6.0, -1.0, 7.0, 5.0, -3.0, 8.0}) CHECK(sc.update(v, true, true, dt) == State::Inhale);
    for (int i = 0; i < 30; ++i) st = sc.update(-8.0, true, true, dt);
    CHECK(st == State::Exhale);
    for (int i = 0; i < 60; ++i) st = sc.update(0.0, true, true, dt);
    CHECK(st == State::NoDetect);
    CHECK(sc.breath_count() == 1);
    CHECK(sc.update(0.0, false, false, dt) == State::Skip);

    // Guide validation never reports the opposite direction.
    SmoothClassifier g(cfg);
    for (int i = 0; i < 30; ++i) g.update(8.0, true, true, dt);
    CHECK(g.validate(State::Inhale, BE_GUIDE_EXHALE) == State::NoDetect);
    CHECK(g.validate(State::Inhale, BE_GUIDE_HOLD_TOP) == State::Inhale);
}

static void test_grid_signal() {
    be_config cfg = be_default_config();
    cfg.signal_source = BE_SIGNAL_GRID;
    const cv::Mat canvas = textured_canvas(H + 200, W);
    Engine e(cfg);
    Result r;
    for (int i = 0; i < 15; ++i) r = e.process(shifted(canvas, i * 0.5), &kShoulders, BE_GUIDE_NONE, i * 33.3);
    std::printf("  grid: %d valid points, dy %.1f chest %.1f belly %.1f\n", r.feature_count,
                r.dy_features, r.dy_chest, r.dy_belly);
    CHECK(r.signal_valid && r.feature_count >= 30);
    CHECK(std::abs(r.dy_features - 50.0) < 5.0);
    CHECK(std::abs(r.dy_chest - 50.0) < 5.0 && std::abs(r.dy_belly - 50.0) < 5.0);
    CHECK(r.state_smooth == State::Inhale);
    {   // Points spread evenly from the ROI (chest) down to the upper belly.
        CHECK(r.roi.has_value());
        const Rect roi = r.roi.value_or(Rect{});
        int in_roi = 0, below = 0;
        for (const auto& p : r.points) {
            if (p.y >= roi.y - 2 && p.y <= roi.y + roi.h + 2) ++in_roi;
            else if (p.y > roi.y + roi.h) ++below;
        }
        std::printf("  grid spread: %d in ROI, %d below it\n", in_roi, below);
        CHECK(in_roi >= 24 && below >= 100);
    }
    {   // Fixed grid gives the same answer on this textured image.
        be_config c = cfg;
        c.grid_smart = 0;
        Engine fx(c);
        for (int i = 0; i < 15; ++i) r = fx.process(shifted(canvas, i * 0.5), &kShoulders, BE_GUIDE_NONE, i * 33.3);
        CHECK(r.signal_valid && std::abs(r.dy_features - 50.0) < 5.0);
    }
    {   // Mostly plain shirt with a few sparse marks inside the ROI: the smart
        // grid still finds enough of them for a valid, accurate measurement.
        cv::Mat sparse(H + 200, W, CV_8UC3, cv::Scalar(225, 225, 225));
        cv::RNG rng(7);
        for (int k = 0; k < 120; ++k)
            cv::circle(sparse, cv::Point(rng.uniform(0, W), rng.uniform(0, H + 200)), 2,
                       cv::Scalar(90, 90, 90), -1);
        cv::GaussianBlur(sparse, sparse, cv::Size(3, 3), 0);
        int counts[2];
        for (int smart = 0; smart < 2; ++smart) {
            be_config c = cfg;
            c.grid_smart = smart;
            Engine g(c);
            for (int i = 0; i < 10; ++i) r = g.process(shifted(sparse, i * 0.5), &kShoulders, BE_GUIDE_NONE, i * 33.3);
            counts[smart] = r.feature_count;
            if (smart) CHECK(r.signal_valid && std::abs(r.dy_features - 50.0) < 8.0);
        }
        std::printf("  sparse texture: fixed grid %d points, smart grid %d points\n", counts[0], counts[1]);
        CHECK(counts[1] >= cfg.feat_min_points);
    }
    // Flat (textureless) image: every point rejected, no measurement.
    Engine f(cfg);
    const cv::Mat flat(H, W, CV_8UC3, cv::Scalar(230, 230, 230));
    for (int i = 0; i < 5; ++i) r = f.process(flat, &kShoulders, BE_GUIDE_NONE, i * 33.3);
    CHECK(!r.signal_valid && r.feature_count == 0);

    // Shoulders near / beyond the frame edges must never throw.
    const std::vector<Shoulders> edge = {
        {{W * 0.3, H - 5.0}, {W * 0.7, H - 5.0}},      // torso below the frame
        {{W - 30.0, H * 0.4}, {W + 200.0, H * 0.4}},   // off the right edge
        {{-250.0, H * 0.4}, {10.0, H * 0.4}},          // off the left edge
        {{W * 0.3, -300.0}, {W * 0.7, -300.0}},        // above the frame
    };
    for (int smart = 0; smart < 2; ++smart) {
        be_config c = cfg;
        c.grid_smart = smart;
        Engine g(c);
        bool threw = false;
        try {
            for (const auto& sh : edge)
                for (int i = 0; i < 3; ++i) g.process(shifted(canvas, i * 0.5), &sh, BE_GUIDE_NONE, i * 33.3);
        } catch (...) {
            threw = true;
        }
        CHECK(!threw);
    }
}

// Body "breathes" up 0.5 px/frame while the phone shakes the whole image.
static void test_camera_motion() {
    be_config cfg = be_default_config();
    cfg.signal_source = BE_SIGNAL_GRID;
    cfg.cam_shake_px_s = 1e9;  // test compensation only
    const cv::Mat canvas = textured_canvas(H + 200, W);
    auto run = [&](bool give_motion) {
        Engine e(cfg);
        Result r;
        double err = 0;
        int n = 0;
        double cam_prev = 0;
        for (int i = 0; i < 40; ++i) {
            const double cam_off = 4.0 * std::sin(i * 0.9);  // hand shake, px
            if (give_motion && i > 0) e.set_camera_motion(CamMotion{0.0, -(cam_off - cam_prev), 0.0});
            r = e.process(shifted(canvas, i * 0.5 + cam_off), &kShoulders, BE_GUIDE_NONE, i * 33.3);
            cam_prev = cam_off;
            if (i >= 3 && r.signal_valid) {
                err += std::abs(r.dy_features - 50.0);
                ++n;
            }
        }
        return n ? err / n : 1e9;
    };
    const double without = run(false), with = run(true);
    std::printf("  shaking phone: error without gyro %.1f, with gyro %.1f\n", without, with);
    CHECK(without > 100.0);
    CHECK(with < 5.0);

    // Big motion -> shaking, no measurement; recovers afterwards.
    be_config c2 = be_default_config();
    c2.signal_source = BE_SIGNAL_GRID;
    Engine e(c2);
    Result r;
    for (int i = 0; i < 5; ++i) r = e.process(shifted(canvas, i * 0.5), &kShoulders, BE_GUIDE_NONE, i * 33.3);
    e.set_camera_motion(CamMotion{0.0, 15.0, 0.0});
    r = e.process(shifted(canvas, 2.5), &kShoulders, BE_GUIDE_NONE, 5 * 33.3);
    CHECK(r.shaking && !r.signal_valid);
    for (int i = 6; i < 20; ++i) r = e.process(shifted(canvas, i * 0.5), &kShoulders, BE_GUIDE_NONE, i * 33.3);
    CHECK(!r.shaking && r.signal_valid);
}

static void test_feature_signal() {
    be_config cfg = be_default_config();
    cfg.signal_source = BE_SIGNAL_FEATURES;
    const cv::Mat canvas = textured_canvas(H + 200, W);
    {
        Engine e(cfg);
        Result r;
        for (int i = 0; i < 15; ++i) r = e.process(shifted(canvas, i * 0.5), &kShoulders, BE_GUIDE_NONE, i * 33.3);
        CHECK(r.signal_valid && r.feature_count >= 20);
        CHECK(std::abs(r.dy_features - 50.0) < 5.0);  // 0.5 px up per frame * gain 100
        CHECK(r.state_votes == State::Inhale && r.state_smooth == State::Inhale);
        CHECK(!r.points.empty());
    }
    {
        Engine e(cfg);
        Result r;
        for (int i = 0; i < 15; ++i) r = e.process(shifted(canvas, 10 - i * 0.5), &kShoulders, BE_GUIDE_NONE, i * 33.3);
        CHECK(r.dy_features < -45.0 && r.state_smooth == State::Exhale);
    }
    {
        Engine e(cfg);
        Result r;
        for (int i = 0; i < 15; ++i) r = e.process(shifted(canvas, 0), &kShoulders, BE_GUIDE_NONE, i * 33.3);
        CHECK(std::abs(r.dy_features) < 1.0 && r.state_smooth == State::NoDetect);
    }
    {   // Sub-pixel breathing (3 px amplitude, 4 s period): direction mostly right.
        for (int src : {int(BE_SIGNAL_FARNEBACK), int(BE_SIGNAL_FEATURES), int(BE_SIGNAL_GRID), 3}) {
            be_config c = cfg;
            c.signal_source = src == 3 ? BE_SIGNAL_GRID : src;
            c.grid_smart = src == 3 ? 0 : 1;
            Engine e(c);
            int ok = 0, n = 0;
            for (int i = 0; i < 240; ++i) {
                const double t = i / 30.0;
                const double off = 3.0 * std::sin(2 * CV_PI * t / 4.0);
                const double vel = std::cos(2 * CV_PI * t / 4.0);
                Result r = e.process(shifted(canvas, off), &kShoulders, BE_GUIDE_NONE, i * 1000.0 / 30);
                if (i < 30) continue;
                ++n;
                if ((r.state_smooth == State::Inhale && vel > 0) || (r.state_smooth == State::Exhale && vel < 0)) ++ok;
            }
            std::printf("  sine, %s + smooth: %.0f%% of frames correct\n",
                        src == 3 ? "fixed grid" : src == BE_SIGNAL_GRID ? "smart grid" : src == BE_SIGNAL_FEATURES ? "features" : "farneback", 100.0 * ok / n);
            CHECK(ok > 0.75 * n);
        }
    }
}

int main() {
    const std::vector<std::pair<const char*, std::function<void()>>> tests = {
        {"classifier", test_classifier},
        {"smooth_classifier", test_smooth_classifier},
        {"feature_signal", test_feature_signal},
        {"grid_signal", test_grid_signal},
        {"camera_motion", test_camera_motion},
        {"roi", test_roi},
        {"engine_motion", test_engine_motion},
        {"c_api_formats_and_rotation", test_c_api_formats_and_rotation},
    };
    for (auto& [name, fn] : tests) {
        const int before = g_failed;
        fn();
        std::printf("%s %s\n", g_failed == before ? "PASS" : "FAIL", name);
    }
    std::printf("%d checks, %d failed\n", g_checks, g_failed);
    return g_failed ? 1 : 0;
}
