// tc_bench: honest ratio + throughput harness.
//
//   ./build/tc_bench [--quick] [--csv results/out.csv]
//
// Protocol: Release builds, 3 warm-up iters, 21 timed iters, report median
// (with min/p90). Hot regime uses one 64 KiB block; cold uses a large tensor.

#include "baselines.h"
#include "codec/byte_shuffle.h"
#include "compressor.h"
#include "huffman_experiment.h"
#include "synthetic.h"
#include "sysinfo.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <span>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

struct Stats {
    double median = 0;
    double min = 0;
    double p90 = 0;
};

Stats summarize(std::vector<double> samples) {
    Stats s;
    if (samples.empty()) return s;
    std::sort(samples.begin(), samples.end());
    s.min = samples.front();
    s.median = samples[samples.size() / 2];
    s.p90 = samples[(samples.size() * 9) / 10];
    return s;
}

double gbps(size_t bytes, double seconds) {
    if (seconds <= 0) return 0;
    return (static_cast<double>(bytes) / 1e9) / seconds;
}

uint64_t checksum(std::span<const uint8_t> data) {
    uint64_t h = 14695981039346656037ull;
    for (uint8_t b : data) {
        h ^= b;
        h *= 1099511628211ull;
    }
    return h;
}

struct Row {
    std::string dataset;
    std::string codec;
    std::string regime;
    double ratio = 0;
    double enc_gbs = 0;
    double dec_gbs = 0;
    double dec_min = 0;
    double dec_p90 = 0;
    size_t raw_bytes = 0;
    size_t comp_bytes = 0;
};

struct Dataset {
    std::string name;
    tc::RawTensor tensor;
};

std::vector<Dataset> make_datasets(bool quick) {
    std::vector<Dataset> out;
    auto add = [&](const char* name, tc::Dist dist, std::vector<size_t> shape, uint64_t seed,
                   double sigma = 0.02, double rho = 0.9) {
        tc::SynthSpec spec;
        spec.dist = dist;
        spec.shape = std::move(shape);
        spec.seed = seed;
        spec.sigma = sigma;
        spec.rho = rho;
        if (dist == tc::Dist::QuantizedInt8) spec.dtype = tc::DataType::INT8;
        out.push_back({name, tc::generate(spec)});
    };

    if (quick) {
        add("zeros_64k", tc::Dist::Zeros, {64, 256}, 1);           // 64 KiB
        add("gaussian_64k", tc::Dist::Gaussian, {64, 256}, 2);
        add("ar1_64k", tc::Dist::Ar1Gaussian, {64, 256}, 3, 0.02, 0.99);
        add("random_64k", tc::Dist::RandomBits, {64, 256}, 4);
        add("shared_64k", tc::Dist::SharedExponent, {64, 256}, 5);
    } else {
        add("zeros_hot", tc::Dist::Zeros, {64, 256}, 1);
        add("gaussian_hot", tc::Dist::Gaussian, {64, 256}, 2);
        add("ar1_hot", tc::Dist::Ar1Gaussian, {64, 256}, 3, 0.02, 0.99);
        add("random_hot", tc::Dist::RandomBits, {64, 256}, 4);
        add("shared_hot", tc::Dist::SharedExponent, {64, 256}, 5);
        // Cold: 32 MiB (fast enough on a laptop; still > typical L2/L3).
        add("zeros_cold", tc::Dist::Zeros, {2048, 4096}, 11);
        add("gaussian_cold", tc::Dist::Gaussian, {2048, 4096}, 12);
        add("ar1_cold", tc::Dist::Ar1Gaussian, {2048, 4096}, 13, 0.02, 0.99);
        add("random_cold", tc::Dist::RandomBits, {2048, 4096}, 14);
        add("int8_hot", tc::Dist::QuantizedInt8, {256, 256}, 15);
    }
    return out;
}

Row bench_tc(const Dataset& ds, uint16_t flags, const char* codec_name, bool quick) {
    tc::CompressOptions opts;
    opts.flags = flags;
    opts.block_size = tc::DEFAULT_BLOCK_SIZE;
    tc::TensorCompressor c(opts);

    const size_t raw = ds.tensor.data.size();
    std::vector<uint8_t> frame(c.max_frame_size(raw, opts.block_size));
    std::vector<uint8_t> out(raw);

    const int warm = 3;
    const int iters = quick ? 7 : 21;
    for (int i = 0; i < warm; ++i) {
        (void)c.compress_frame(ds.tensor.data, ds.tensor.shape, ds.tensor.dtype, frame);
    }

    std::vector<double> enc_s, dec_s;
    size_t comp_bytes = 0;
    for (int i = 0; i < iters; ++i) {
        const auto t0 = Clock::now();
        comp_bytes = c.compress_frame(ds.tensor.data, ds.tensor.shape, ds.tensor.dtype, frame);
        const auto t1 = Clock::now();
        enc_s.push_back(std::chrono::duration<double>(t1 - t0).count());
    }
    // Touch output to keep compress live; prepare a stable compressed buffer.
    std::vector<uint8_t> stable(frame.begin(), frame.begin() + static_cast<std::ptrdiff_t>(comp_bytes));
    for (int i = 0; i < warm; ++i) {
        (void)c.decompress_frame(stable, out);
    }
    for (int i = 0; i < iters; ++i) {
        const auto t0 = Clock::now();
        (void)c.decompress_frame(stable, out);
        const auto t1 = Clock::now();
        dec_s.push_back(std::chrono::duration<double>(t1 - t0).count());
        volatile uint64_t sink = checksum(out);
        (void)sink;
    }

    const Stats es = summarize(enc_s);
    const Stats ds_s = summarize(dec_s);
    Row row;
    row.dataset = ds.name;
    row.codec = codec_name;
    row.regime = raw <= 65536 ? "hot" : "cold";
    row.raw_bytes = raw;
    row.comp_bytes = comp_bytes;
    row.ratio = static_cast<double>(raw) / static_cast<double>(comp_bytes);
    row.enc_gbs = gbps(raw, es.median);
    row.dec_gbs = gbps(raw, ds_s.median);
    row.dec_min = gbps(raw, ds_s.min);
    row.dec_p90 = gbps(raw, ds_s.p90);
    return row;
}

Row bench_baseline(const Dataset& ds, tc::bench::Baseline& baseline, bool quick) {
    Row row;
    row.dataset = ds.name;
    row.codec = baseline.name();
    row.regime = ds.tensor.data.size() <= 65536 ? "hot" : "cold";
    row.raw_bytes = ds.tensor.data.size();
    if (!baseline.available()) {
        row.codec += " (not built)";
        return row;
    }

    const int warm = 3;
    const int iters = quick ? 7 : 21;
    std::vector<uint8_t> out(ds.tensor.data.size());
    auto packed = baseline.compress(ds.tensor.data);
    if (!packed.ok) {
        row.codec += " (error)";
        return row;
    }
    row.comp_bytes = packed.compressed_bytes();
    row.ratio = static_cast<double>(row.raw_bytes) / static_cast<double>(row.comp_bytes);

    for (int i = 0; i < warm; ++i) (void)baseline.compress(ds.tensor.data);
    std::vector<double> enc_s, dec_s;
    for (int i = 0; i < iters; ++i) {
        const auto t0 = Clock::now();
        auto r = baseline.compress(ds.tensor.data);
        const auto t1 = Clock::now();
        enc_s.push_back(std::chrono::duration<double>(t1 - t0).count());
        packed = std::move(r);
    }
    for (int i = 0; i < warm; ++i) (void)baseline.decompress(packed.compressed, out);
    for (int i = 0; i < iters; ++i) {
        const auto t0 = Clock::now();
        (void)baseline.decompress(packed.compressed, out);
        const auto t1 = Clock::now();
        dec_s.push_back(std::chrono::duration<double>(t1 - t0).count());
        volatile uint64_t sink = checksum(out);
        (void)sink;
    }
    const Stats es = summarize(enc_s);
    const Stats ds_s = summarize(dec_s);
    row.enc_gbs = gbps(row.raw_bytes, es.median);
    row.dec_gbs = gbps(row.raw_bytes, ds_s.median);
    row.dec_min = gbps(row.raw_bytes, ds_s.min);
    row.dec_p90 = gbps(row.raw_bytes, ds_s.p90);
    return row;
}

void print_table(const std::vector<Row>& rows) {
    std::printf("\n%-16s %-16s %-6s %8s %10s %10s %10s\n", "dataset", "codec", "regime", "ratio",
                "enc GB/s", "dec GB/s", "dec p90");
    for (const Row& r : rows) {
        std::printf("%-16s %-16s %-6s %8.3f %10.2f %10.2f %10.2f\n", r.dataset.c_str(), r.codec.c_str(),
                    r.regime.c_str(), r.ratio, r.enc_gbs, r.dec_gbs, r.dec_p90);
    }
}

void write_csv(const std::string& path, const tc::bench::HostInfo& host, const std::vector<Row>& rows) {
    std::ofstream out(path);
    out << "# chip=" << host.chip << "\n";
    out << "# compiler=" << host.compiler << "\n";
    out << "# build=" << host.build_type << "\n";
    out << "# git=" << host.git_sha << "\n";
    out << "# simd=" << host.simd << "\n";
    out << "# B1_GBs=" << host.bandwidth_1_gbs << " Ball_GBs=" << host.bandwidth_all_gbs << "\n";
    out << "dataset,codec,regime,ratio,enc_gbs,dec_gbs,dec_min_gbs,dec_p90_gbs,raw_bytes,comp_bytes\n";
    for (const Row& r : rows) {
        out << r.dataset << ',' << r.codec << ',' << r.regime << ',' << r.ratio << ',' << r.enc_gbs << ','
            << r.dec_gbs << ',' << r.dec_min << ',' << r.dec_p90 << ',' << r.raw_bytes << ','
            << r.comp_bytes << '\n';
    }
}

void print_breakeven(const tc::bench::HostInfo& host, const std::vector<Row>& rows) {
    std::printf("\nBreak-even (serial model): required decode GB/s = B1 * R / (R - 1)\n");
    std::printf("%-16s %-16s %8s %12s %12s %s\n", "dataset", "codec", "R", "required", "measured",
                "verdict");
    for (const Row& r : rows) {
        if (r.codec.find("tc-") != 0) continue;
        if (r.ratio <= 1.001) {
            std::printf("%-16s %-16s %8.3f %12s %12.2f never (R~1)\n", r.dataset.c_str(), r.codec.c_str(),
                        r.ratio, "-", r.dec_gbs);
            continue;
        }
        const double required = host.bandwidth_1_gbs * r.ratio / (r.ratio - 1.0);
        const char* verdict = r.dec_gbs >= required ? "WIN" : "LOSS";
        std::printf("%-16s %-16s %8.3f %12.2f %12.2f %s\n", r.dataset.c_str(), r.codec.c_str(), r.ratio,
                    required, r.dec_gbs, verdict);
    }
}

}  // namespace

int main(int argc, char** argv) {
    bool quick = false;
    std::string csv_path;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--quick") == 0) quick = true;
        else if (std::strcmp(argv[i], "--csv") == 0 && i + 1 < argc) csv_path = argv[++i];
        else {
            std::fprintf(stderr, "usage: tc_bench [--quick] [--csv PATH]\n");
            return 2;
        }
    }

    std::printf("=== TC-Runtime bench ===\n");
    const tc::bench::HostInfo host = tc::bench::collect_host_info();
    tc::bench::print_host_info(host);

    const auto datasets = make_datasets(quick);
    std::vector<Row> rows;

    // Ablations + full pipeline.
    struct Ablation {
        const char* name;
        uint16_t flags;
    };
    const Ablation ablations[] = {
        {"tc-full", tc::FLAG_ALL},
        {"tc-shuffle", tc::FLAG_SHUFFLE},
        {"tc-sx", static_cast<uint16_t>(tc::FLAG_SHUFFLE | tc::FLAG_XOR)},
        {"tc-xr", static_cast<uint16_t>(tc::FLAG_XOR | tc::FLAG_RLE)},
        {"tc-rle", tc::FLAG_RLE},
    };

    for (const Dataset& ds : datasets) {
        for (const Ablation& a : ablations) {
            // Skip non-full ablations on cold sets in full mode to keep runtime sane.
            if (!quick && ds.tensor.data.size() > 65536 && a.flags != tc::FLAG_ALL) continue;
            rows.push_back(bench_tc(ds, a.flags, a.name, quick));
        }
        for (tc::bench::Baseline* b : tc::bench::all_baselines()) {
            if (!quick && ds.tensor.data.size() > 65536 && std::string(b->name()).find("zlib-6") != std::string::npos)
                continue;  // zlib-6 on 32 MiB is slow; skip in cold
            rows.push_back(bench_baseline(ds, *b, quick));
        }
    }

    // Scalar vs SIMD decode on one hot set.
    {
        const Dataset* gaussian = nullptr;
        for (const Dataset& ds : datasets) {
            if (ds.name.find("gaussian") != std::string::npos) {
                gaussian = &ds;
                break;
            }
        }
        if (gaussian) {
            tc::codec::set_simd_enabled(false);
            Row scalar = bench_tc(*gaussian, tc::FLAG_ALL, "tc-full-scalar", quick);
            tc::codec::set_simd_enabled(true);
            Row simd = bench_tc(*gaussian, tc::FLAG_ALL, "tc-full-simd", quick);
            rows.push_back(scalar);
            rows.push_back(simd);
            std::printf("\nSIMD vs scalar decode on %s: scalar=%.2f GB/s  simd(%s)=%.2f GB/s\n",
                        gaussian->name.c_str(), scalar.dec_gbs, tc::codec::simd_kernel_name(),
                        simd.dec_gbs);
        }
    }

    // Huffman stretch (one dataset).
    for (const Dataset& ds : datasets) {
        if (ds.tensor.dtype != tc::DataType::FP32) continue;
        if (ds.name.find("gaussian") == std::string::npos) continue;
        auto h = tc::huffman::compress_exponent_plane(ds.tensor.data);
        std::printf("\nHuffman exponent experiment on %s: ratio=%.3fx (toy frame, not TCF1)\n",
                    ds.name.c_str(), h.ratio);
        break;
    }

    print_table(rows);
    print_breakeven(host, rows);

    if (!csv_path.empty()) {
        write_csv(csv_path, host, rows);
        std::printf("\nwrote %s\n", csv_path.c_str());
    }
    return 0;
}
