#include "codec/byte_shuffle.h"

#include <arm_neon.h>
#include <cstring>

// NEON byte-shuffle kernels for 4-byte and 2-byte elements.
//
// AArch64 has de-interleaving loads and interleaving stores built in:
// LD4 reads 64 bytes and returns four 16-byte vectors holding bytes 0, 1, 2, 3
// of 16 consecutive elements, which is exactly one step of the shuffle. ST4 is
// its inverse and is one step of the unshuffle. LD2/ST2 do the same for 2-byte
// elements. The last fewer-than-16 elements use a per-element loop, and tail
// bytes (n_bytes % elem_size) are copied through, as in the scalar reference.

namespace tc::codec {

void byte_shuffle_neon(const uint8_t* src, uint8_t* dst, size_t n_bytes, size_t elem_size) {
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
        for (; i + 16 <= n_elems; i += 16) {
            const uint8x16x4_t v = vld4q_u8(src + 4 * i);
            vst1q_u8(p0 + i, v.val[0]);
            vst1q_u8(p1 + i, v.val[1]);
            vst1q_u8(p2 + i, v.val[2]);
            vst1q_u8(p3 + i, v.val[3]);
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
        for (; i + 16 <= n_elems; i += 16) {
            const uint8x16x2_t v = vld2q_u8(src + 2 * i);
            vst1q_u8(p0 + i, v.val[0]);
            vst1q_u8(p1 + i, v.val[1]);
        }
        for (; i < n_elems; ++i) {
            p0[i] = src[2 * i];
            p1[i] = src[2 * i + 1];
        }
    }

    const size_t body = n_elems * elem_size;
    if (n_bytes != body) std::memcpy(dst + body, src + body, n_bytes - body);
}

void byte_unshuffle_neon(const uint8_t* src, uint8_t* dst, size_t n_bytes, size_t elem_size) {
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
        for (; i + 16 <= n_elems; i += 16) {
            uint8x16x4_t v;
            v.val[0] = vld1q_u8(p0 + i);
            v.val[1] = vld1q_u8(p1 + i);
            v.val[2] = vld1q_u8(p2 + i);
            v.val[3] = vld1q_u8(p3 + i);
            vst4q_u8(dst + 4 * i, v);
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
            uint8x16x2_t v;
            v.val[0] = vld1q_u8(p0 + i);
            v.val[1] = vld1q_u8(p1 + i);
            vst2q_u8(dst + 2 * i, v);
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
