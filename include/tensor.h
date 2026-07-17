#pragma once

#include <vector>
#include <cstdint>
#include <cstddef>

enum class DataType : uint8_t {
    FP32 = 0,
    FP16 = 1,
    INT8 = 2,
};

// Aligning our structures to 64-byte cache lines to eliminate unaligned access penalties.
// Note: alignas(64) applies to the struct object itself, not to data's heap buffer.
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

    // Trailing ';' is required: in C++, `struct Name { ... };` is a type definition.
    // Without ';', the compiler expects more declarators (e.g. `} g_instance;`) and fails.
};

// Bytes per element for a dtype (FP32→4, FP16→2, INT8→1).
size_t element_size(DataType dtype);

// Number of elements = product of shape dimensions (empty shape → 0).
size_t numel(const std::vector<size_t>& shape);
