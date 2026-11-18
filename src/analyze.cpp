#include "analyze.h"

#include "codec/byte_shuffle.h"
#include "codec/rle.h"
#include "codec/xor_delta.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace tc {

namespace {

double entropy_from_histogram(const uint64_t* counts, size_t buckets, uint64_t total) {
    if (total == 0) return 0.0;
    double h = 0.0;
    for (size_t i = 0; i < buckets; ++i) {
        if (counts[i] == 0) continue;
        const double p = static_cast<double>(counts[i]) / static_cast<double>(total);
        h -= p * std::log2(p);
    }
    return h;
}

}  // namespace

double shannon_entropy_bits(const uint8_t* data, size_t count, size_t stride) {
    uint64_t counts[256] = {};
    for (size_t i = 0; i < count; ++i) ++counts[data[i * stride]];
    return entropy_from_histogram(counts, 256, count);
}

AnalysisReport analyze(const RawTensor& tensor) {
    validate(tensor);

    AnalysisReport report;
    report.elem_size = element_size(tensor.dtype);
    report.n_elems = tensor.data.size() / report.elem_size;
    report.total_bytes = tensor.data.size();
    report.planes.resize(report.elem_size);

    const size_t n = report.n_elems;
    if (n == 0) return report;

    std::vector<uint8_t> shuffled(tensor.data.size());
    codec::byte_shuffle_scalar(tensor.data.data(), shuffled.data(), tensor.data.size(), report.elem_size);

    std::vector<uint8_t> delta(n);
    std::vector<uint8_t> packed(codec::rle_max_encoded_size(n));

    double order0_bytes = 0.0;
    double xor_order0_bytes = 0.0;
    uint64_t rle_total = 0;
    for (size_t j = 0; j < report.elem_size; ++j) {
        const uint8_t* plane = shuffled.data() + j * n;
        PlaneStats& stats = report.planes[j];
        stats.bytes = n;
        stats.entropy_bits = shannon_entropy_bits(plane, n);

        codec::xor_delta_encode(plane, delta.data(), n);
        stats.xor_entropy_bits = shannon_entropy_bits(delta.data(), n);
        stats.rle_bytes = codec::rle_encode(delta.data(), n, packed.data(), packed.size());

        order0_bytes += stats.entropy_bits / 8.0 * static_cast<double>(n);
        xor_order0_bytes += stats.xor_entropy_bits / 8.0 * static_cast<double>(n);
        rle_total += std::min<uint64_t>(stats.rle_bytes, n);
    }

    // A fully predictable tensor has a zero-byte bound: report the ratio as inf.
    const double total = static_cast<double>(report.total_bytes);
    report.order0_ratio = total / order0_bytes;
    report.xor_order0_ratio = total / xor_order0_bytes;
    report.plane_rle_ratio = total / static_cast<double>(rle_total);

    if (tensor.dtype == DataType::FP32 || tensor.dtype == DataType::FP16) {
        const bool fp32 = tensor.dtype == DataType::FP32;
        const int exponent_bits = fp32 ? 8 : 5;
        const int mantissa_bits = fp32 ? 23 : 10;
        uint64_t sign_counts[2] = {};
        uint64_t exponent_counts[256] = {};
        for (size_t i = 0; i < n; ++i) {
            // Assemble the element from little-endian bytes (the on-disk order).
            uint32_t bits = 0;
            for (size_t b = 0; b < report.elem_size; ++b) {
                bits |= static_cast<uint32_t>(tensor.data[i * report.elem_size + b]) << (8 * b);
            }
            ++exponent_counts[(bits >> mantissa_bits) & ((1u << exponent_bits) - 1)];
            ++sign_counts[(bits >> (mantissa_bits + exponent_bits)) & 1u];
        }
        report.exponent_field_bits = exponent_bits;
        report.exponent_entropy_bits = entropy_from_histogram(exponent_counts, 256, n);
        report.sign_entropy_bits = entropy_from_histogram(sign_counts, 2, n);
    }
    return report;
}

}  // namespace tc
