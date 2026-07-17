#include "format.h"

#include <stdexcept>

namespace tc {

void pack_header(const Tcf1Header& hdr, std::vector<uint8_t>& out) {
    // Phase 0 stub: copy raw 64 bytes once HUMAN-TODO fields exist.
    // Real field-wise little-endian packing lands when you fill Tcf1Header.
    out.resize(TCF1_HEADER_BYTES);
    const auto* src = reinterpret_cast<const uint8_t*>(&hdr);
    for (std::size_t i = 0; i < TCF1_HEADER_BYTES; ++i) {
        out[i] = src[i];
    }
}

Tcf1Header unpack_header(const std::vector<uint8_t>& data) {
    if (data.size() < TCF1_HEADER_BYTES) {
        throw std::runtime_error("TCF1: buffer shorter than 64-byte header");
    }
    Tcf1Header hdr{};
    auto* dst = reinterpret_cast<uint8_t*>(&hdr);
    for (std::size_t i = 0; i < TCF1_HEADER_BYTES; ++i) {
        dst[i] = data[i];
    }
    // Magic / version checks belong after you define real fields (architecture.md §11).
    return hdr;
}

}  // namespace tc
