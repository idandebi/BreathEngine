// Runs the C++ engine over a .besq sequence file and writes one CSV row per
// frame. python/tools/parity.py writes the sequences and compares the output
// with the Python reference engine.
//
// .besq format (little endian):
//   char[4] "BESQ", int32 version=1, int32 width, int32 height, int32 frames
//   per frame: int32 guide_segment, int32 has_shoulders,
//              float32 lx, ly, rx, ry, uint8 rgb[height*width*3]
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <vector>

#include "breath_engine/breath_engine.h"

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: be_parity <in.besq> <out.csv> [--no-low-light]\n");
        return 2;
    }
    std::ifstream in(argv[1], std::ios::binary);
    if (!in) {
        std::fprintf(stderr, "cannot open %s\n", argv[1]);
        return 2;
    }
    char magic[4];
    int32_t version, w, h, n;
    in.read(magic, 4);
    in.read(reinterpret_cast<char*>(&version), 4);
    in.read(reinterpret_cast<char*>(&w), 4);
    in.read(reinterpret_cast<char*>(&h), 4);
    in.read(reinterpret_cast<char*>(&n), 4);
    if (std::memcmp(magic, "BESQ", 4) != 0 || version != 1) {
        std::fprintf(stderr, "bad file\n");
        return 2;
    }

    be_config cfg = be_default_config();
    for (int i = 3; i < argc; ++i)
        if (std::strcmp(argv[i], "--no-low-light") == 0) cfg.low_light = 0;
    be_engine* e = be_create(&cfg);

    FILE* out = std::fopen(argv[2], "w");
    std::fprintf(out, "frame,state,raw_state,dy_flow,votes_pos,votes_neg,roi_x,roi_y,roi_w,roi_h,reanchored,flow_valid,ms\n");
    std::vector<uint8_t> px(static_cast<size_t>(w) * h * 3);
    double total_ms = 0;
    for (int f = 0; f < n; ++f) {
        int32_t seg, has;
        float sh[4];
        in.read(reinterpret_cast<char*>(&seg), 4);
        in.read(reinterpret_cast<char*>(&has), 4);
        in.read(reinterpret_cast<char*>(sh), 16);
        in.read(reinterpret_cast<char*>(px.data()), px.size());
        if (!in) {
            std::fprintf(stderr, "truncated at frame %d\n", f);
            return 2;
        }
        be_image img{px.data(), w, h, w * 3, BE_FORMAT_RGB, 0, nullptr, 0};
        be_shoulders s{sh[0], sh[1], sh[2], sh[3]};
        be_result r{};
        if (be_process(e, &img, has ? &s : nullptr, seg, &r) != BE_OK) {
            std::fprintf(stderr, "be_process failed at frame %d\n", f);
            return 1;
        }
        total_ms += r.process_ms;
        std::fprintf(out, "%d,%d,%d,%.6f,%d,%d,%d,%d,%d,%d,%d,%d,%.3f\n", f, r.state, r.raw_state,
                     r.dy_flow, r.votes_pos, r.votes_neg, r.roi_x, r.roi_y, r.roi_w, r.roi_h,
                     r.reanchored, r.flow_valid, r.process_ms);
    }
    std::fclose(out);
    be_destroy(e);
    std::printf("%d frames, mean %.2f ms/frame\n", n, n ? total_ms / n : 0.0);
    return 0;
}
