#pragma once

#include <cstddef>
#include <cstdint>

// XOR-delta over a byte stream:
//
//   encode: out[0] = in[0];  out[i] = in[i] ^ in[i-1]
//   decode: in[0]  = out[0]; in[i]  = out[i] ^ in[i-1]
//
// A run of equal bytes becomes one byte followed by zeros, which is what the
// RLE stage looks for. The same idea, on 64-bit doubles, is the core of
// Facebook's Gorilla time-series encoding.
//
// The two directions are not symmetric in cost. Encode reads only the input,
// so every output byte is independent and the loop vectorizes. Decode is a
// prefix XOR: byte i needs the already-decoded byte i-1, a loop-carried
// dependency of one XOR per byte.

namespace tc::codec {

// src and dst must not overlap.
void xor_delta_encode(const uint8_t* src, uint8_t* dst, size_t n);

// src == dst (in place) is allowed; partial overlap is not.
void xor_delta_decode(const uint8_t* src, uint8_t* dst, size_t n);

}  // namespace tc::codec
