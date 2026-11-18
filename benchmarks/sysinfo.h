#pragma once

#include <string>

namespace tc::bench {

struct HostInfo {
    std::string chip;
    std::string os;
    std::string compiler;
    std::string build_type;
    std::string cxx_flags;
    std::string git_sha;
    std::string simd;
    int p_cores = 0;
    int e_cores = 0;
    double ram_gib = 0.0;
    double bandwidth_1_gbs = 0.0;    // single-thread memcpy over ~1 GiB
    double bandwidth_all_gbs = 0.0;  // all-hardware-concurrency memcpy
};

HostInfo collect_host_info();
void print_host_info(const HostInfo& info);

}  // namespace tc::bench
