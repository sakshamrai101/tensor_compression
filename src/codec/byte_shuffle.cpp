#include "codec/byte_shuffle.h"

#include <atomic>
#include <cstring>

namespace tc::codec {

namespace {
std::atomic<bool> g_simd_enabled{true};
}  // namespace

void byte_shuffle_scalar(const uint8_t* src, uint8_t* dst, size_t n_bytes, size_t elem_size) {
    if (elem_size <= 1) {
        if (n_bytes != 0) std::memcpy(dst, src, n_bytes);
        return;
    }
    const size_t n_elems = n_bytes / elem_size;
    for (size_t j = 0; j < elem_size; ++j) {
        uint8_t* plane = dst + j * n_elems;
        for (size_t i = 0; i < n_elems; ++i) {
            plane[i] = src[i * elem_size + j];
        }
    }
    const size_t body = n_elems * elem_size;
    if (n_bytes != body) std::memcpy(dst + body, src + body, n_bytes - body);
}

void byte_unshuffle_scalar(const uint8_t* src, uint8_t* dst, size_t n_bytes, size_t elem_size) {
    if (elem_size <= 1) {
        if (n_bytes != 0) std::memcpy(dst, src, n_bytes);
        return;
    }
    const size_t n_elems = n_bytes / elem_size;
    for (size_t j = 0; j < elem_size; ++j) {
        const uint8_t* plane = src + j * n_elems;
        for (size_t i = 0; i < n_elems; ++i) {
            dst[i * elem_size + j] = plane[i];
        }
    }
    const size_t body = n_elems * elem_size;
    if (n_bytes != body) std::memcpy(dst + body, src + body, n_bytes - body);
}

void set_simd_enabled(bool enabled) { g_simd_enabled.store(enabled, std::memory_order_relaxed); }
bool simd_enabled() { return g_simd_enabled.load(std::memory_order_relaxed); }

const char* simd_kernel_name() {
    if (!simd_enabled()) return "scalar";
#if defined(TC_HAVE_NEON)
    return "neon";
#elif defined(TC_HAVE_AVX2)
    return cpu_has_avx2() ? "avx2" : "scalar";
#else
    return "scalar";
#endif
}

void byte_shuffle(const uint8_t* src, uint8_t* dst, size_t n_bytes, size_t elem_size) {
#if defined(TC_HAVE_NEON)
    if (simd_enabled() && (elem_size == 4 || elem_size == 2)) {
        byte_shuffle_neon(src, dst, n_bytes, elem_size);
        return;
    }
#elif defined(TC_HAVE_AVX2)
    if (simd_enabled() && (elem_size == 4 || elem_size == 2) && cpu_has_avx2()) {
        byte_shuffle_avx2(src, dst, n_bytes, elem_size);
        return;
    }
#endif
    byte_shuffle_scalar(src, dst, n_bytes, elem_size);
}

void byte_unshuffle(const uint8_t* src, uint8_t* dst, size_t n_bytes, size_t elem_size) {
#if defined(TC_HAVE_NEON)
    if (simd_enabled() && (elem_size == 4 || elem_size == 2)) {
        byte_unshuffle_neon(src, dst, n_bytes, elem_size);
        return;
    }
#elif defined(TC_HAVE_AVX2)
    if (simd_enabled() && (elem_size == 4 || elem_size == 2) && cpu_has_avx2()) {
        byte_unshuffle_avx2(src, dst, n_bytes, elem_size);
        return;
    }
#endif
    byte_unshuffle_scalar(src, dst, n_bytes, elem_size);
}

}  // namespace tc::codec
