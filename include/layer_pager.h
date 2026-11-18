#pragma once

#include "compressor.h"
#include "tensor.h"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

// LayerPager: the residency pattern the codec is meant for, as a demo.
//
// N layers stay in memory compressed. Before layer i runs, it is decompressed
// into one reusable scratch buffer; the returned view is valid until the next
// map_layer() call. Peak resident weight memory is therefore
// sum(compressed layers) + max(raw layer), instead of sum(raw layers).
//
// This is not an inference engine: there is no model math here.

namespace tc {

class LayerPager {
public:
    LayerPager() = default;
    explicit LayerPager(const CompressOptions& options) : compressor_(options) {}

    // Compresses a raw layer and keeps only the compressed form. Returns its index.
    size_t add_layer(const RawTensor& layer);

    // Takes ownership of already-compressed layers and sizes the scratch
    // buffer for the largest one.
    void load_library(std::vector<CompressedTensor> layers);

    size_t layer_count() const { return layers_.size(); }
    const CompressedTensor& compressed_layer(size_t i) const { return layers_.at(i); }

    // Decompresses layer i into the scratch buffer and returns a view of it.
    // Allocation-free once the scratch buffer has reached the largest layer's size.
    std::span<const uint8_t> map_layer(size_t i);

    // Shape and dtype of the layer currently mapped.
    const RawTensor& mapped() const { return scratch_; }

    size_t compressed_bytes() const;    // sum of frame sizes
    size_t raw_bytes() const;           // sum of original sizes
    size_t scratch_bytes() const { return scratch_.data.capacity(); }

private:
    TensorCompressor compressor_;
    std::vector<CompressedTensor> layers_;
    RawTensor scratch_;
};

}  // namespace tc
