#pragma once

#include "format.h"
#include "tensor.h"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace tc {

struct CompressOptions {
    uint16_t flags = FLAG_ALL;                  // which stages to run (FLAG_* in format.h)
    uint32_t block_size = DEFAULT_BLOCK_SIZE;   // bytes of raw tensor per block

    // When true (the default), a block whose encoded form is not smaller than
    // its raw form is stored raw, so a frame never grows by more than the
    // header and block records. Tests and ablation benchmarks turn this off to
    // force every block through the stages.
    bool stored_fallback = true;
};

// Lossless tensor codec: byte-shuffle -> XOR-delta -> zero-run RLE per block,
// framed as TCF1. decompress(compress(x)) is byte-identical to x.
//
// An instance owns scratch buffers sized to one block, so it is cheap to reuse
// and must not be shared between threads. Invalid input throws
// std::invalid_argument (bad arguments) or std::runtime_error (bad frame).
class TensorCompressor {
public:
    TensorCompressor();
    explicit TensorCompressor(const CompressOptions& options);

    const CompressOptions& options() const { return opts_; }

    CompressedTensor compress(const RawTensor& tensor);
    RawTensor decompress(const CompressedTensor& compressed);

    // Same as compress(), reusing out's storage.
    void compress_to(const RawTensor& in, CompressedTensor& out);

    // Hot-path decode. Reuses out.data and out.shape: once `out` has been sized
    // by a first call (or by the caller), further calls on frames of the same
    // size perform no heap allocation.
    void decompress_to(const CompressedTensor& in, RawTensor& out);

    // --- buffer-level API (no tensor structs) -----------------------------------

    // Upper bound on the frame size for raw_bytes of input.
    static size_t max_frame_size(size_t raw_bytes, uint32_t block_size);

    // Compresses raw into out_frame (at least max_frame_size(raw.size(), block_size)
    // bytes) and returns the frame length. raw.size() must equal numel(shape) * element size.
    size_t compress_frame(std::span<const uint8_t> raw, const std::vector<size_t>& shape,
                          DataType dtype, std::span<uint8_t> out_frame);

    // Decompresses frame into out, whose size must equal the frame's
    // uncompressed_size. Returns the parsed header.
    Tcf1Header decompress_frame(std::span<const uint8_t> frame, std::span<uint8_t> out);

private:
    void reserve_scratch(size_t block_bytes, bool for_encode);

    CompressOptions opts_;
    std::vector<uint8_t> scratch_a_;
    std::vector<uint8_t> scratch_b_;
};

// Wraps an existing TCF1 frame (for example one read from disk), filling the
// convenience fields from its header. Throws std::runtime_error on a bad frame.
CompressedTensor compressed_from_frame(std::vector<uint8_t> frame);

}  // namespace tc
