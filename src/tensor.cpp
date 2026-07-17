#include "tensor.h"

#include <stdexcept>

size_t element_size(DataType dtype) {
    // =============================================================================
    // HUMAN-TODO: element_size
    // WHY: Every tensor size check is shape × element_size(dtype). Wrong sizes
    //      break the invariant in architecture.md §3.2 and corrupt compress/decompress.
    // DO: Switch on dtype; return 4 (FP32), 2 (FP16), 1 (INT8).
    //     Throw std::invalid_argument (or similar) for an unknown value.
    // HINT: `switch (dtype) { case DataType::FP32: return 4; ... }`
    // READ: architecture.md §3.1 (DataType), learning.md §4 "Before Phase 0"
    // =============================================================================
    (void)dtype;
    throw std::logic_error("HUMAN-TODO: element_size not implemented");
}

size_t numel(const std::vector<size_t>& shape) {
    // =============================================================================
    // HUMAN-TODO: numel
    // WHY: "Row-major tightly packed" means total elements are the product of dims.
    //      You need this to verify data.size() == numel(shape) * element_size(dtype).
    // DO: Multiply all entries in shape. If shape.empty(), return 0.
    // HINT: Start with size_t n = 1; loop and multiply (watch empty → 0, not 1).
    // READ: architecture.md §3.2 (RawTensor invariant), learning.md §4 "Before Phase 0"
    // =============================================================================
    (void)shape;
    throw std::logic_error("HUMAN-TODO: numel not implemented");
}
