#include "layer_pager.h"

#include <algorithm>
#include <utility>

namespace tc {

size_t LayerPager::add_layer(const RawTensor& layer) {
    layers_.push_back(compressor_.compress(layer));
    // Reserve up front so map_layer() never has to grow the scratch buffer.
    scratch_.data.reserve(std::max(scratch_.data.capacity(), layer.data.size()));
    scratch_.shape.reserve(TCF1_MAX_DIMS);
    return layers_.size() - 1;
}

void LayerPager::load_library(std::vector<CompressedTensor> layers) {
    layers_ = std::move(layers);
    size_t largest = 0;
    for (const CompressedTensor& layer : layers_) {
        // Trust the frame header, not the convenience fields.
        largest = std::max<size_t>(largest, FrameReader(layer.compressed_data).header().uncompressed_size);
    }
    scratch_.data.reserve(largest);
    scratch_.shape.reserve(TCF1_MAX_DIMS);
}

std::span<const uint8_t> LayerPager::map_layer(size_t i) {
    compressor_.decompress_to(layers_.at(i), scratch_);
    return scratch_.data;
}

size_t LayerPager::compressed_bytes() const {
    size_t total = 0;
    for (const CompressedTensor& layer : layers_) total += layer.compressed_data.size();
    return total;
}

size_t LayerPager::raw_bytes() const {
    size_t total = 0;
    for (const CompressedTensor& layer : layers_) total += layer.original_size_bytes;
    return total;
}

}  // namespace tc
