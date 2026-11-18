#pragma once

#include "tensor.h"

#include <cstddef>
#include <cstdint>
#include <vector>

// Entropy analysis: where in a tensor's bytes is the redundancy, and how much
// of it does the RLE pipeline capture?
//
// All entropies are order-0 (memoryless) Shannon entropies of a byte
// histogram: H = -sum p(b) log2 p(b), in bits per byte, between 0 and 8. For a
// byte plane, H/8 * plane_bytes is the size an ideal entropy coder with one
// static table for that plane would reach. It is a bound for that model class
// only; a coder with context can beat it and RLE cannot reach it unless the
// redundancy shows up as runs.

namespace tc {

struct PlaneStats {
    uint64_t bytes = 0;             // bytes in this plane (= number of elements)
    double entropy_bits = 0.0;      // H of the plane's bytes
    double xor_entropy_bits = 0.0;  // H after XOR-delta along the plane
    uint64_t rle_bytes = 0;         // size after XOR-delta + zero-run RLE of this plane alone
};

struct AnalysisReport {
    size_t elem_size = 0;
    uint64_t n_elems = 0;
    uint64_t total_bytes = 0;
    std::vector<PlaneStats> planes;         // index j = byte j of each element (j = 0 is the low byte)

    double order0_ratio = 0.0;              // total_bytes / sum_j (H_j / 8 * bytes_j)
    double xor_order0_ratio = 0.0;          // same bound on the XOR-delta'd planes
    double plane_rle_ratio = 0.0;           // total_bytes / sum_j min(rle_bytes_j, bytes_j)

    // Float dtypes only (-1 otherwise): entropy of the IEEE754 fields.
    double sign_entropy_bits = -1.0;        // of 1 bit
    double exponent_entropy_bits = -1.0;    // of 8 bits (FP32) or 5 bits (FP16)
    int exponent_field_bits = 0;
};

// Order-0 entropy in bits per byte of data[0], data[stride], ... (count bytes).
double shannon_entropy_bits(const uint8_t* data, size_t count, size_t stride = 1);

AnalysisReport analyze(const RawTensor& tensor);

}  // namespace tc
