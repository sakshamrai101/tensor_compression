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

    void bit_unshuffle_sse(const uint8_t* src, uint8_t* dst, size_t size);
};

// Next steps:
//   1. src/compressor.cpp — compress(): bit_shuffle_sse → xor_delta → rle_pack
//   2. src/compressor.cpp — decompress(): reverse the above; fill CompressedTensor metadata
//   3. CMakeLists.txt + -mavx2 — link a small round-trip test on synthetic FP32 layer weights