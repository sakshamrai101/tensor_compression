#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>
#include <vector>

namespace tc {

enum class DataType : uint8_t {
    FP32 = 0,
    FP16 = 1,
    INT8 = 2,
};

// alignas(64) aligns the struct object itself to a cache line. It does NOT
// align the heap buffer owned by `data`; kernels never assume aligned payloads.
struct alignas(64) RawTensor {
    std::vector<uint8_t> data;      // raw bytes, row-major, tightly packed
    std::vector<size_t> shape;
    DataType dtype = DataType::FP32;

    size_t size_bytes() const { return data.size(); }
};

struct alignas(64) CompressedTensor {
    std::vector<uint8_t> compressed_data;   // full TCF1 frame: 64-byte header + block records

    // Denormalized copies of header fields, filled by the compressor for
    // convenience. The header inside compressed_data is the source of truth.
    std::vector<size_t> shape;
    DataType dtype = DataType::FP32;
    size_t original_size_bytes = 0;
    uint32_t block_count = 0;
    uint32_t compression_ratio_x100 = 0;    // 133 means 1.33x; computed against the whole frame
};

// Bytes per element (FP32 -> 4, FP16 -> 2, INT8 -> 1). Throws on an unknown value.
size_t element_size(DataType dtype);

// Product of the dimensions. An empty shape is 0 elements (not 1), and the
// product is overflow-checked.
size_t numel(const std::vector<size_t>& shape);

// Enforces data.size() == numel(shape) * element_size(dtype).
void validate(const RawTensor& tensor);

const char* dtype_name(DataType dtype);
DataType parse_dtype(std::string_view name);    // "fp32" | "fp16" | "int8"

}  // namespace tc
