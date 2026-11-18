#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

// Stretch experiment (Phase 3.6): static Huffman coding of the FP32 exponent
// field after isolating it via byte-shuffle. Not part of the TCF1 pipeline —
// cited against ZipNN / DFloat11-style exponent entropy coding.

namespace tc::huffman {

struct HuffmanResult {
    size_t input_bytes = 0;
    size_t exponent_bytes = 0;       // one byte per FP32 element (byte plane 3)
    size_t encoded_exponent_bytes = 0;
    size_t total_frame_bytes = 0;    // mantissa planes stored raw + coded exponents + tiny header
    double ratio = 0.0;              // input_bytes / total_frame_bytes
};

// Codes only the high byte plane (sign+exponent-ish) of an FP32 buffer with a
// static Huffman table built from that plane's histogram. Other planes are
// stored raw. Returns the packed size of that toy frame.
HuffmanResult compress_exponent_plane(std::span<const uint8_t> fp32_bytes);

}  // namespace tc::huffman
