#include "codec/xor_delta.h"

namespace tc::codec {

void xor_delta_encode(const uint8_t* src, uint8_t* dst, size_t n) {
    if (n == 0) return;
    dst[0] = src[0];
    for (size_t i = 1; i < n; ++i) {
        dst[i] = static_cast<uint8_t>(src[i] ^ src[i - 1]);
    }
}

void xor_delta_decode(const uint8_t* src, uint8_t* dst, size_t n) {
    uint8_t running = 0;
    for (size_t i = 0; i < n; ++i) {
        running = static_cast<uint8_t>(running ^ src[i]);
        dst[i] = running;
    }
}

}  // namespace tc::codec
