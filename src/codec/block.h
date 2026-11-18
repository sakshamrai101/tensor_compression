#pragma once

#include <cstddef>
#include <cstdint>

// One block through the stage pipeline. This is the only place that knows the
// stage order; framing and the stored-raw fallback live in the compressor.
//
//   encode: [shuffle] -> [xor-delta] -> [rle]
//   decode: [un-rle]  -> [xor-undo]  -> [unshuffle]
//
// Stages are selected by the FLAG_* bits from format.h. None of these
// functions allocate.

namespace tc::codec {

// Output capacity encode_block needs for a raw_len-byte block.
size_t max_encoded_block_size(size_t raw_len);

// Encodes raw[0..raw_len) into dst and returns the encoded length, which can
// exceed raw_len on incompressible input. dst must hold
// max_encoded_block_size(raw_len) bytes; scratch_a and scratch_b must each hold
// raw_len bytes. No buffer may overlap another.
size_t encode_block(const uint8_t* raw, size_t raw_len, size_t elem_size, uint16_t flags,
                    uint8_t* dst, uint8_t* scratch_a, uint8_t* scratch_b);

// Decodes an encoded block into out[0..raw_len). scratch must hold raw_len
// bytes. Throws std::runtime_error if comp does not decode to exactly raw_len bytes.
void decode_block(const uint8_t* comp, size_t comp_len, size_t elem_size, uint16_t flags,
                  uint8_t* out, size_t raw_len, uint8_t* scratch);

}  // namespace tc::codec
