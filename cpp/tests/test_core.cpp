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
            CHECK(be_process(e, &img, &s, BE_GUIDE_NONE, &res) == BE_OK);
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
        CHECK(be_process(e, &img, &s, BE_GUIDE_NONE, &res) == BE_OK);
    }
    CHECK(res.state == BE_STATE_INHALE);
    be_image bad{nullptr, 0, 0, 0, BE_FORMAT_RGB, 0, nullptr, 0};
    CHECK(be_process(e, &bad, nullptr, BE_GUIDE_NONE, &res) == BE_ERR_INVALID_ARG);
    be_destroy(e);
}

int main() {
    const std::vector<std::pair<const char*, std::function<void()>>> tests = {
        {"classifier", test_classifier},
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
