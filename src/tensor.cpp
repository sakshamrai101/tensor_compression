#include "tensor.h"

#include <limits>
#include <stdexcept>
#include <string>

namespace tc {

size_t element_size(DataType dtype) {
    switch (dtype) {
        case DataType::FP32: return 4;
        case DataType::FP16: return 2;
        case DataType::INT8: return 1;
    }
    throw std::invalid_argument("element_size: unknown DataType");
}

size_t numel(const std::vector<size_t>& shape) {
    if (shape.empty()) {
        return 0;
    }
    size_t n = 1;
    for (size_t dim : shape) {
        if (dim != 0 && n > std::numeric_limits<size_t>::max() / dim) {
            throw std::overflow_error("numel: shape product overflows size_t");
        }
        n *= dim;
    }
    return n;
}

void validate(const RawTensor& tensor) {
    const size_t elems = numel(tensor.shape);
    const size_t elem = element_size(tensor.dtype);
    if (elems > std::numeric_limits<size_t>::max() / elem) {
        throw std::overflow_error("validate: tensor byte size overflows size_t");
    }
    if (tensor.data.size() != elems * elem) {
        throw std::invalid_argument(
            "validate: data.size()=" + std::to_string(tensor.data.size()) +
            " but shape*element_size=" + std::to_string(elems * elem));
    }
}

const char* dtype_name(DataType dtype) {
    switch (dtype) {
        case DataType::FP32: return "fp32";
        case DataType::FP16: return "fp16";
        case DataType::INT8: return "int8";
    }
    return "unknown";
}

DataType parse_dtype(std::string_view name) {
    if (name == "fp32") return DataType::FP32;
    if (name == "fp16") return DataType::FP16;
    if (name == "int8") return DataType::INT8;
    throw std::invalid_argument("unknown dtype '" + std::string(name) + "' (expected fp32|fp16|int8)");
}

}  // namespace tc
