#pragma once 
#include "tensor.h"

class TensorCompressor {

    public:

    TensorCompressor() = default;
    ~TensorCompressor() = default;

    // lossless compression of raw tensor block 
    CompressedTensor compress(const RawTensor& tensor);

    // decompress block into raw tensor
    RawTensor decompress(const CompressedTensor& compressed);


    private:

    // low-level bit shuffling routine 
    void bit_shuffle_sse(const uint8_t* src, uint8_t* dst, size_t size);

    void bit_unshuffle_sse(const uint8_t* src, uint8_t* dst, size_t size)
};