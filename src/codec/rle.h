#pragma once

#include <cstddef>
#include <cstdint>

// Zero-run + literal RLE (TCF1 v1).
//
// The encoded stream is a sequence of tokens:
//
//   0x00, k                  -> k+1 zero bytes      (1..256)
//   0x01, k, <k+1 bytes>     -> k+1 literal bytes   (1..256)
//
// Any other tag byte is invalid.
//
// Why this and not classic [value][run_len] pairs: classic RLE spends 2 bytes
// on every byte that does not repeat, so it doubles incompressible data, and
// float mantissa bytes are incompressible. Here incompressible bytes ride in
// literal tokens at 2 bytes of overhead per 256.
//
// The encoder only emits a zero token for runs of at least kMinZeroRun bytes.
// A zero token costs 2 bytes and forces a new 2-byte literal header after it,
// so a shorter run is cheaper left inside the surrounding literal. That rule
// bounds the worst case at n + n/128 + 2 bytes (under 0.8% expansion).

namespace tc::codec {

constexpr uint8_t kRleTokenZeroRun = 0x00;
constexpr uint8_t kRleTokenLiteral = 0x01;
constexpr size_t kRleMaxRun = 256;
constexpr size_t kMinZeroRun = 4;

// Capacity that rle_encode needs for an n-byte input.
constexpr size_t rle_max_encoded_size(size_t n) { return n + n / 128 + 16; }

// Encodes src[0..n) into dst and returns the encoded length.
// Throws std::length_error if cap < rle_max_encoded_size(n).
size_t rle_encode(const uint8_t* src, size_t n, uint8_t* dst, size_t cap);

// Decodes src[0..n) into dst, which must hold `expected` bytes, and returns
// `expected`. Throws std::runtime_error if the stream is truncated, contains an
// unknown token, or does not decode to exactly `expected` bytes. Never reads
// outside src[0..n) or writes outside dst[0..expected).
size_t rle_decode(const uint8_t* src, size_t n, uint8_t* dst, size_t expected);

}  // namespace tc::codec
