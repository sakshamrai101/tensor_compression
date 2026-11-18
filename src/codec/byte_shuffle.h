#pragma once

#include <cstddef>
#include <cstdint>

// Byte-shuffle: array-of-elements -> one plane per byte position.
//
//   input  (elem_size 4): e0b0 e0b1 e0b2 e0b3 | e1b0 e1b1 e1b2 e1b3 | ...
//   output              : e0b0 e1b0 e2b0 ...  | e0b1 e1b1 ...  | e0b2 ... | e0b3 ...
//
// Why: in a little-endian IEEE754 float, byte 3 holds the sign and the top 7
// exponent bits, byte 2 holds the last exponent bit and the top mantissa bits,
// and bytes 1 and 0 are low mantissa. Neighbouring weights share magnitude far
// more often than they share mantissa bits, so grouping byte 3 of every element
// puts the predictable bytes next to each other. This is the "shuffle" filter
// from Blosc/HDF5; it is byte-granular, not the bit-granular Bitshuffle.
//
// If n_bytes is not a multiple of elem_size, the trailing n_bytes % elem_size
// bytes are copied through unchanged. src and dst must not overlap.

namespace tc::codec {

// Reference implementations. These are the oracle: every SIMD kernel must
// produce byte-identical output.
void byte_shuffle_scalar(const uint8_t* src, uint8_t* dst, size_t n_bytes, size_t elem_size);
void byte_unshuffle_scalar(const uint8_t* src, uint8_t* dst, size_t n_bytes, size_t elem_size);

// Dispatching entry points: use a SIMD kernel when one was compiled in, the
// CPU supports it, and SIMD is enabled; otherwise the scalar reference.
void byte_shuffle(const uint8_t* src, uint8_t* dst, size_t n_bytes, size_t elem_size);
void byte_unshuffle(const uint8_t* src, uint8_t* dst, size_t n_bytes, size_t elem_size);

// Process-wide switch used by tests and benchmarks to compare scalar vs SIMD.
// Not thread-safe to flip while other threads are coding.
void set_simd_enabled(bool enabled);
bool simd_enabled();

// "neon", "avx2" or "scalar": the kernel the dispatcher would use right now.
const char* simd_kernel_name();

#if defined(TC_HAVE_NEON)
void byte_shuffle_neon(const uint8_t* src, uint8_t* dst, size_t n_bytes, size_t elem_size);
void byte_unshuffle_neon(const uint8_t* src, uint8_t* dst, size_t n_bytes, size_t elem_size);
#endif

#if defined(TC_HAVE_AVX2)
// Callers must check cpu_has_avx2() first.
bool cpu_has_avx2();
void byte_shuffle_avx2(const uint8_t* src, uint8_t* dst, size_t n_bytes, size_t elem_size);
void byte_unshuffle_avx2(const uint8_t* src, uint8_t* dst, size_t n_bytes, size_t elem_size);
#endif

}  // namespace tc::codec
