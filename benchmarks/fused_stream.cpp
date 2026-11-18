// Stretch (Phase 3.5): fused streaming / multithreaded pager microbench.
//
// Compares:
//   A) touch-sum over raw FP32 bytes (DRAM-bound)
//   B) decompress_to into reusable scratch then touch (serial)
//   C) parallel decompress of N copies (one compressor per thread)

#include "compressor.h"
#include "synthetic.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <span>
#include <thread>
#include <vector>

namespace {

uint64_t touch_sum(std::span<const uint8_t> bytes) {
    uint64_t s = 0;
    for (size_t i = 0; i < bytes.size(); i += 64) s += bytes[i];
    return s;
}

double seconds(std::chrono::steady_clock::time_point a, std::chrono::steady_clock::time_point b) {
    return std::chrono::duration<double>(b - a).count();
}

}  // namespace

int main(int argc, char** argv) {
    using namespace tc;
    size_t nbytes = 4u * 1024u * 1024u;
    int threads = static_cast<int>(std::thread::hardware_concurrency());
    if (argc > 1) nbytes = static_cast<size_t>(std::strtoull(argv[1], nullptr, 10));
    if (argc > 2) threads = std::atoi(argv[2]);
    if (threads < 1) threads = 1;
    if (nbytes < 4 || nbytes % 4 != 0) {
        std::fprintf(stderr, "nbytes must be a positive multiple of 4\n");
        return 2;
    }

    SynthSpec spec;
    spec.dist = Dist::Ar1Gaussian;
    spec.shape = {nbytes / 4};
    spec.rho = 0.99;
    spec.seed = 7;
    RawTensor raw = generate(spec);

    CompressOptions opts;
    opts.flags = FLAG_ALL;
    TensorCompressor c(opts);
    CompressedTensor mid = c.compress(raw);

    volatile uint64_t sink = 0;
    constexpr int kIters = 10;

    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < kIters; ++i) sink = touch_sum(raw.data);
    auto t1 = std::chrono::steady_clock::now();
    const double raw_gbs = (static_cast<double>(nbytes) * kIters / 1e9) / seconds(t0, t1);

    RawTensor scratch;
    c.decompress_to(mid, scratch);  // warm
    t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < kIters; ++i) {
        c.decompress_to(mid, scratch);
        sink = touch_sum(scratch.data);
    }
    t1 = std::chrono::steady_clock::now();
    const double fused_gbs = (static_cast<double>(nbytes) * kIters / 1e9) / seconds(t0, t1);

    std::vector<CompressedTensor> copies(static_cast<size_t>(threads), mid);
    t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < kIters; ++i) {
        std::vector<std::thread> workers;
        std::vector<uint64_t> partial(static_cast<size_t>(threads));
        for (int t = 0; t < threads; ++t) {
            workers.emplace_back([&, t]() {
                TensorCompressor local(opts);
                RawTensor out;
                local.decompress_to(copies[static_cast<size_t>(t)], out);
                partial[static_cast<size_t>(t)] = touch_sum(out.data);
            });
        }
        for (auto& w : workers) w.join();
        sink = std::accumulate(partial.begin(), partial.end(), uint64_t{0});
    }
    t1 = std::chrono::steady_clock::now();
    const double mt_gbs =
        (static_cast<double>(nbytes) * kIters * static_cast<double>(threads) / 1e9) / seconds(t0, t1);

    const double ratio = inspect_frame(mid.compressed_data).ratio;
    std::printf("fused_stream nbytes=%zu threads=%d ratio=%.3fx sink=%llu\n", nbytes, threads, ratio,
                static_cast<unsigned long long>(sink));
    std::printf("  raw touch:                 %.2f GB/s\n", raw_gbs);
    std::printf("  decompress_to + touch:     %.2f GB/s\n", fused_gbs);
    std::printf("  parallel decompress (agg): %.2f GB/s\n", mt_gbs);
    return 0;
}
