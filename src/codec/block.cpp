#include "codec/block.h"

#include "codec/byte_shuffle.h"
#include "codec/rle.h"
#include "codec/xor_delta.h"
#include "format.h"

#include <cstring>
#include <stdexcept>

namespace tc::codec {

size_t max_encoded_block_size(size_t raw_len) { return rle_max_encoded_size(raw_len); }

size_t encode_block(const uint8_t* raw, size_t raw_len, size_t elem_size, uint16_t flags,
                    uint8_t* dst, uint8_t* scratch_a, uint8_t* scratch_b) {
    const bool shuffle = (flags & FLAG_SHUFFLE) != 0;
    const bool xor_delta = (flags & FLAG_XOR) != 0;
    const bool rle = (flags & FLAG_RLE) != 0;

    // Each stage reads `cur` and writes its own buffer; the last active stage
    // writes straight into dst so no stage output is copied twice.
    const uint8_t* cur = raw;

    if (shuffle) {
        uint8_t* target = (xor_delta || rle) ? scratch_a : dst;
        byte_shuffle(cur, target, raw_len, elem_size);
        cur = target;
    }
    if (xor_delta) {
        uint8_t* target = rle ? scratch_b : dst;
        xor_delta_encode(cur, target, raw_len);
        cur = target;
    }
    if (rle) {
        return rle_encode(cur, raw_len, dst, max_encoded_block_size(raw_len));
    }
    if (cur == raw && raw_len != 0) {
        std::memcpy(dst, raw, raw_len);     // no stages selected
    }
    return raw_len;
}

void decode_block(const uint8_t* comp, size_t comp_len, size_t elem_size, uint16_t flags,
                  uint8_t* out, size_t raw_len, uint8_t* scratch) {
    const bool shuffle = (flags & FLAG_SHUFFLE) != 0;
    const bool xor_delta = (flags & FLAG_XOR) != 0;
    const bool rle = (flags & FLAG_RLE) != 0;

    const uint8_t* cur = comp;

    if (rle) {
        uint8_t* target = (xor_delta || shuffle) ? scratch : out;
        rle_decode(comp, comp_len, target, raw_len);
        cur = target;
    } else if (comp_len != raw_len) {
        // Without RLE every stage is length-preserving.
        throw std::runtime_error("TCF1: encoded block length must equal raw_len when RLE is off");
    }
    if (xor_delta) {
        uint8_t* target = shuffle ? scratch : out;
        xor_delta_decode(cur, target, raw_len);     // in place when cur == scratch
        cur = target;
    }
    if (shuffle) {
        byte_unshuffle(cur, out, raw_len, elem_size);
    } else if (cur != out && raw_len != 0) {
        std::memcpy(out, cur, raw_len);
    }
}

}  // namespace tc::codec
