#pragma once 
#include <vector>
#include <cstdint>
#include <cstddef>



enum class DataType {
    FP32,
    FP16,
    INT8
};


// Aligning our structures to 64-byte cache lines to eliminate unaligned access penalties
struct alignas(64) RawTensor {

    std::vector<uint8_t> data;
    std::vector<size_t> shape;

    DataType dtype;

    size_t size_bytes() const {
        return data.size();
    }
};

struct alignas(64) CompressedTensor {
    std::vector<uint8_t> compressed_data;
    std::vector<size_t> shape;
    DataType dtype;

    size_t original_size_bytes;

    // Meta-data tracking for layer-aware compression
    uint32_t block_count;
    uint32_t compression_ratio_x100; // e.g., 180 means 1.8x compression


}