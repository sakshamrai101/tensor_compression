#include "codec/byte_shuffle.h"
#include "test_harness.h"
#include "test_util.h"

#include <string>
#include <vector>

using namespace tc::codec;

namespace {

bool kernels_match(size_t n_bytes, size_t elem_size, uint64_t seed) {
    const auto src = tctest::random_bytes(n_bytes, seed);
    std::vector<uint8_t> scalar_s(n_bytes), scalar_u(n_bytes);
    std::vector<uint8_t> simd_s(n_bytes), simd_u(n_bytes);

    set_simd_enabled(false);
    byte_shuffle(src.data(), scalar_s.data(), n_bytes, elem_size);
    byte_unshuffle(scalar_s.data(), scalar_u.data(), n_bytes, elem_size);

    set_simd_enabled(true);
    byte_shuffle(src.data(), simd_s.data(), n_bytes, elem_size);
    byte_unshuffle(simd_s.data(), simd_u.data(), n_bytes, elem_size);

    return scalar_s == simd_s && scalar_u == simd_u && scalar_u == src;
}

}  // namespace

TC_TEST(simd, dispatcher_reports_kernel) {
    set_simd_enabled(true);
    const char* name = simd_kernel_name();
    TC_CHECK(name != nullptr);
    TC_CHECK(std::string(name) == "neon" || std::string(name) == "avx2" ||
             std::string(name) == "scalar");
}

TC_TEST(simd, matches_scalar_fp32) {
    const size_t sizes[] = {0, 4, 8, 16, 64, 4 * 15, 4 * 16, 4 * 17, 4 * 63, 4 * 64 + 3, 4096};
    for (size_t n : sizes) {
        TC_CHECK_MSG(kernels_match(n, 4, 2000 + n), "n=" + std::to_string(n));
    }
}

TC_TEST(simd, matches_scalar_fp16) {
    const size_t sizes[] = {0, 2, 4, 2 * 15, 2 * 16, 2 * 17, 2 * 64 + 1, 2048};
    for (size_t n : sizes) {
        TC_CHECK_MSG(kernels_match(n, 2, 3000 + n), "n=" + std::to_string(n));
    }
}

TC_TEST(simd, disable_forces_scalar) {
    set_simd_enabled(false);
    TC_CHECK(std::string(simd_kernel_name()) == "scalar");
    set_simd_enabled(true);
}
