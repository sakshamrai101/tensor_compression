#include "codec/byte_shuffle.h"

#include <cstring>
#include <immintrin.h>

// AVX2 byte-shuffle kernels for 4-byte and 2-byte elements. This TU is the
// only one compiled with -mavx2, and callers reach it only after
// cpu_has_avx2() returns true.
//
// x86 has no de-interleaving load like NEON's LD4, so the shuffle is built
// from two permutes: PSHUFB groups byte positions inside each 128-bit lane,
// then a cross-lane permute puts the two lanes' groups next to each other.
// The unshuffle is the classic unpack ladder (bytes -> words -> dwords).

namespace tc::codec {

bool cpu_has_avx2() {
    static const bool supported = __builtin_cpu_supports("avx2");
    return supported;
}

void byte_shuffle_avx2(const uint8_t* src, uint8_t* dst, size_t n_bytes, size_t elem_size) {
    if (elem_size != 4 && elem_size != 2) {
        byte_shuffle_scalar(src, dst, n_bytes, elem_size);
        return;
    }
    const size_t n_elems = n_bytes / elem_size;
    size_t i = 0;

    if (elem_size == 4) {
        uint8_t* p0 = dst;
        uint8_t* p1 = dst + n_elems;
        uint8_t* p2 = dst + 2 * n_elems;
        uint8_t* p3 = dst + 3 * n_elems;
        // Per lane: 4 elements -> [b0 x4 | b1 x4 | b2 x4 | b3 x4].
        const __m256i group = _mm256_setr_epi8(
            0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15,
            0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15);
        // Interleave the two lanes' dwords -> [b0 x8 | b1 x8 | b2 x8 | b3 x8].
        const __m256i merge = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
        for (; i + 8 <= n_elems; i += 8) {
            __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + 4 * i));
            v = _mm256_shuffle_epi8(v, group);
            v = _mm256_permutevar8x32_epi32(v, merge);
            const __m128i lo = _mm256_castsi256_si128(v);
            const __m128i hi = _mm256_extracti128_si256(v, 1);
            _mm_storel_epi64(reinterpret_cast<__m128i*>(p0 + i), lo);
            _mm_storel_epi64(reinterpret_cast<__m128i*>(p1 + i), _mm_unpackhi_epi64(lo, lo));
            _mm_storel_epi64(reinterpret_cast<__m128i*>(p2 + i), hi);
            _mm_storel_epi64(reinterpret_cast<__m128i*>(p3 + i), _mm_unpackhi_epi64(hi, hi));
        }
        for (; i < n_elems; ++i) {
            p0[i] = src[4 * i];
            p1[i] = src[4 * i + 1];
            p2[i] = src[4 * i + 2];
            p3[i] = src[4 * i + 3];
        }
    } else {
        uint8_t* p0 = dst;
        uint8_t* p1 = dst + n_elems;
        // Per lane: 8 elements -> [b0 x8 | b1 x8].
        const __m256i group = _mm256_setr_epi8(
            0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15,
            0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15);
        for (; i + 16 <= n_elems; i += 16) {
            __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + 2 * i));
            v = _mm256_shuffle_epi8(v, group);
            // qwords [l0.b0, l0.b1, l1.b0, l1.b1] -> [l0.b0, l1.b0, l0.b1, l1.b1]
            v = _mm256_permute4x64_epi64(v, 0xD8);
            _mm_storeu_si128(reinterpret_cast<__m128i*>(p0 + i), _mm256_castsi256_si128(v));
            _mm_storeu_si128(reinterpret_cast<__m128i*>(p1 + i), _mm256_extracti128_si256(v, 1));
        }
        for (; i < n_elems; ++i) {
            p0[i] = src[2 * i];
            p1[i] = src[2 * i + 1];
        }
    }

    const size_t body = n_elems * elem_size;
    if (n_bytes != body) std::memcpy(dst + body, src + body, n_bytes - body);
}

void byte_unshuffle_avx2(const uint8_t* src, uint8_t* dst, size_t n_bytes, size_t elem_size) {
    if (elem_size != 4 && elem_size != 2) {
        byte_unshuffle_scalar(src, dst, n_bytes, elem_size);
        return;
    }
    const size_t n_elems = n_bytes / elem_size;
    size_t i = 0;

    if (elem_size == 4) {
        const uint8_t* p0 = src;
        const uint8_t* p1 = src + n_elems;
        const uint8_t* p2 = src + 2 * n_elems;
        const uint8_t* p3 = src + 3 * n_elems;
        for (; i + 32 <= n_elems; i += 32) {
            const __m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p0 + i));
            const __m256i b = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p1 + i));
            const __m256i c = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p2 + i));
            const __m256i d = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p3 + i));
            // Unpacks work per 128-bit lane: lane 0 covers elements 0..15, lane 1 covers 16..31.
            const __m256i ab_lo = _mm256_unpacklo_epi8(a, b);   // b0b1 of elems 0..7   | 16..23
            const __m256i ab_hi = _mm256_unpackhi_epi8(a, b);   // b0b1 of elems 8..15  | 24..31
            const __m256i cd_lo = _mm256_unpacklo_epi8(c, d);
            const __m256i cd_hi = _mm256_unpackhi_epi8(c, d);
            const __m256i q0 = _mm256_unpacklo_epi16(ab_lo, cd_lo);     // elems 0..3   | 16..19
            const __m256i q1 = _mm256_unpackhi_epi16(ab_lo, cd_lo);     // elems 4..7   | 20..23
            const __m256i q2 = _mm256_unpacklo_epi16(ab_hi, cd_hi);     // elems 8..11  | 24..27
            const __m256i q3 = _mm256_unpackhi_epi16(ab_hi, cd_hi);     // elems 12..15 | 28..31
            uint8_t* out = dst + 4 * i;
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(out),      _mm256_permute2x128_si256(q0, q1, 0x20));
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + 32), _mm256_permute2x128_si256(q2, q3, 0x20));
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + 64), _mm256_permute2x128_si256(q0, q1, 0x31));
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + 96), _mm256_permute2x128_si256(q2, q3, 0x31));
        }
        for (; i < n_elems; ++i) {
            dst[4 * i] = p0[i];
            dst[4 * i + 1] = p1[i];
            dst[4 * i + 2] = p2[i];
            dst[4 * i + 3] = p3[i];
        }
    } else {
        const uint8_t* p0 = src;
        const uint8_t* p1 = src + n_elems;
        for (; i + 16 <= n_elems; i += 16) {
            const __m128i a = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p0 + i));
            const __m128i b = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p1 + i));
            _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + 2 * i), _mm_unpacklo_epi8(a, b));
            _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + 2 * i + 16), _mm_unpackhi_epi8(a, b));
        }
        for (; i < n_elems; ++i) {
            dst[2 * i] = p0[i];
            dst[2 * i + 1] = p1[i];
        }
    }

    const size_t body = n_elems * elem_size;
    if (n_bytes != body) std::memcpy(dst + body, src + body, n_bytes - body);
}

}  // namespace tc::codec
