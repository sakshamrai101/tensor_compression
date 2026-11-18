#include "sysinfo.h"

#include "codec/byte_shuffle.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#if defined(__APPLE__)
#include <sys/sysctl.h>
#include <sys/utsname.h>
#endif

#ifndef TC_CXX_COMPILER_ID
#define TC_CXX_COMPILER_ID "unknown"
#endif
#ifndef TC_CXX_COMPILER_VERSION
#define TC_CXX_COMPILER_VERSION ""
#endif
#ifndef TC_BUILD_TYPE
#define TC_BUILD_TYPE ""
#endif
#ifndef TC_CXX_FLAGS
#define TC_CXX_FLAGS ""
#endif

namespace tc::bench {
namespace {

std::string sysctl_string(const char* name) {
#if defined(__APPLE__)
    char buf[256] = {};
    size_t len = sizeof(buf);
    if (sysctlbyname(name, buf, &len, nullptr, 0) == 0) return std::string(buf);
#else
    (void)name;
#endif
    return "unknown";
}

int sysctl_int(const char* name) {
#if defined(__APPLE__)
    int value = 0;
    size_t len = sizeof(value);
    if (sysctlbyname(name, &value, &len, nullptr, 0) == 0) return value;
#else
    (void)name;
#endif
    return 0;
}

double measure_memcpy_gbs(size_t bytes, int threads) {
    std::vector<uint8_t> src(bytes), dst(bytes);
    for (size_t i = 0; i < bytes; i += 4096) src[i] = static_cast<uint8_t>(i);
    const size_t chunk = bytes / static_cast<size_t>(threads);
    auto once = [&]() {
        std::vector<std::thread> workers;
        for (int t = 0; t < threads; ++t) {
            const size_t begin = static_cast<size_t>(t) * chunk;
            const size_t end = (t + 1 == threads) ? bytes : begin + chunk;
            workers.emplace_back([&src, &dst, begin, end]() {
                std::memcpy(dst.data() + begin, src.data() + begin, end - begin);
            });
        }
        for (auto& w : workers) w.join();
        // Touch a checksum so the copy is not optimized away.
        volatile uint8_t sink = dst[bytes / 2];
        (void)sink;
    };
    once();  // warm-up
    const auto t0 = std::chrono::steady_clock::now();
    constexpr int kIters = 3;
    for (int i = 0; i < kIters; ++i) once();
    const auto t1 = std::chrono::steady_clock::now();
    const double secs =
        std::chrono::duration<double>(t1 - t0).count() / static_cast<double>(kIters);
    return (static_cast<double>(bytes) / 1e9) / secs;
}

std::string git_sha() {
#if defined(TC_SOURCE_DIR)
    std::string cmd = "git -C \"";
    cmd += TC_SOURCE_DIR;
    cmd += "\" rev-parse --short HEAD 2>/dev/null";
    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) return "unknown";
    char buf[64] = {};
    const char* got = fgets(buf, sizeof(buf), pipe);
    pclose(pipe);
    if (!got) return "unknown";
    std::string sha(buf);
    while (!sha.empty() && (sha.back() == '\n' || sha.back() == '\r')) sha.pop_back();
    return sha.empty() ? "unknown" : sha;
#else
    return "unknown";
#endif
}

}  // namespace

HostInfo collect_host_info() {
    HostInfo info;
    // Apple Silicon often has no machdep.cpu.brand_string; compose a useful label.
    info.chip = sysctl_string("machdep.cpu.brand_string");
    const std::string model = sysctl_string("hw.model");
#if defined(__APPLE__) && defined(__aarch64__)
    if (info.chip == "unknown" || info.chip.empty()) {
        info.chip = "Apple Silicon";
        if (model != "unknown" && !model.empty()) info.chip += " (" + model + ")";
    }
#else
    if (info.chip == "unknown" || info.chip.empty()) info.chip = model;
#endif
#if defined(__APPLE__)
    utsname u{};
    if (uname(&u) == 0) {
        info.os = std::string(u.sysname) + " " + u.release;
    }
#endif
    info.compiler = std::string(TC_CXX_COMPILER_ID) + " " + TC_CXX_COMPILER_VERSION;
    info.build_type = TC_BUILD_TYPE;
    info.cxx_flags = TC_CXX_FLAGS;
    info.git_sha = git_sha();
    info.simd = codec::simd_kernel_name();
    info.p_cores = sysctl_int("hw.perflevel0.physicalcpu");
    info.e_cores = sysctl_int("hw.perflevel1.physicalcpu");
    const int64_t mem = []() -> int64_t {
#if defined(__APPLE__)
        int64_t m = 0;
        size_t len = sizeof(m);
        sysctlbyname("hw.memsize", &m, &len, nullptr, 0);
        return m;
#else
        return 0;
#endif
    }();
    info.ram_gib = static_cast<double>(mem) / (1024.0 * 1024.0 * 1024.0);

    constexpr size_t kBandBytes = 256ull * 1024ull * 1024ull;  // 256 MiB: faster than 1 GiB, still cold
    info.bandwidth_1_gbs = measure_memcpy_gbs(kBandBytes, 1);
    const unsigned hc = std::max(1u, std::thread::hardware_concurrency());
    info.bandwidth_all_gbs = measure_memcpy_gbs(kBandBytes, static_cast<int>(hc));
    return info;
}

void print_host_info(const HostInfo& info) {
    std::printf("host:       %s\n", info.chip.c_str());
    std::printf("os:         %s\n", info.os.c_str());
    std::printf("compiler:   %s\n", info.compiler.c_str());
    std::printf("build:      %s (%s)\n", info.build_type.c_str(), info.cxx_flags.c_str());
    std::printf("git:        %s\n", info.git_sha.c_str());
    std::printf("simd:       %s\n", info.simd.c_str());
    std::printf("cores:      P=%d E=%d (hw_concurrency=%u)\n", info.p_cores, info.e_cores,
                std::thread::hardware_concurrency());
    std::printf("ram:        %.1f GiB\n", info.ram_gib);
    std::printf("memcpy B1:  %.2f GB/s (single thread, 256 MiB)\n", info.bandwidth_1_gbs);
    std::printf("memcpy Ball:%.2f GB/s (all threads, 256 MiB)\n", info.bandwidth_all_gbs);
}

}  // namespace tc::bench
