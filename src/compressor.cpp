#include "compressor.h"

#include <stdexcept>

CompressedTensor TensorCompressor::compress(const RawTensor& tensor) {
    // Phase 0 stub — real pipeline is Phase 1 (architecture.md §5).
    (void)tensor;
    throw std::logic_error("TensorCompressor::compress not implemented (Phase 1)");
}

RawTensor TensorCompressor::decompress(const CompressedTensor& compressed) {
    (void)compressed;
    throw std::logic_error("TensorCompressor::decompress not implemented (Phase 1)");
}

void TensorCompressor::bit_shuffle_sse(const uint8_t* src, uint8_t* dst, size_t size) {
    // Phase 0 stub — scalar byte-shuffle is Phase 1 (architecture.md §5.1).
    (void)src;
    (void)dst;
    (void)size;
}

void TensorCompressor::bit_unshuffle_sse(const uint8_t* src, uint8_t* dst, size_t size) {
    (void)src;
    (void)dst;
    (void)size;
}
