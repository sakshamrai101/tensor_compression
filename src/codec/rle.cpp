#include "codec/rle.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace tc::codec {

namespace {

inline uint64_t load_u64(const uint8_t* p) {
    uint64_t v;
    std::memcpy(&v, p, sizeof(v));
    return v;
}

// True if any of the 8 bytes in v is zero. Subtracting 1 from each byte
// borrows into its top bit only when the byte was 0 (or >= 0x81, which the
// `& ~v` term removes). Byte order does not matter for a yes/no answer.
inline bool has_zero_byte(uint64_t v) {
    return ((v - 0x0101010101010101ull) & ~v & 0x8080808080808080ull) != 0;
}

}  // namespace

size_t rle_encode(const uint8_t* src, size_t n, uint8_t* dst, size_t cap) {
    if (cap < rle_max_encoded_size(n)) {
        throw std::length_error("rle_encode: output capacity below rle_max_encoded_size(n)");
    }

    size_t op = 0;
    size_t literal_start = 0;   // src[literal_start..i) is waiting to be written as literals

    auto flush_literals = [&](size_t end) {
        size_t pos = literal_start;
        while (pos < end) {
            const size_t len = std::min(end - pos, kRleMaxRun);
            dst[op++] = kRleTokenLiteral;
            dst[op++] = static_cast<uint8_t>(len - 1);
            std::memcpy(dst + op, src + pos, len);
            op += len;
            pos += len;
        }
    };

    size_t i = 0;
    while (i < n) {
        if (src[i] != 0) {
            // Skip 8 bytes at a time while none of them is zero.
            while (i + 8 <= n && !has_zero_byte(load_u64(src + i))) i += 8;
            while (i < n && src[i] != 0) ++i;
            continue;
        }

        size_t run_end = i + 1;
        while (run_end + 8 <= n && load_u64(src + run_end) == 0) run_end += 8;
        while (run_end < n && src[run_end] == 0) ++run_end;

        size_t run = run_end - i;
        if (run >= kMinZeroRun) {
            flush_literals(i);
            while (run > 0) {
                const size_t len = std::min(run, kRleMaxRun);
                dst[op++] = kRleTokenZeroRun;
                dst[op++] = static_cast<uint8_t>(len - 1);
                run -= len;
            }
            literal_start = run_end;
        }
        // A short run stays pending and is written as part of the literal around it.
        i = run_end;
    }
    flush_literals(n);
    return op;
}

size_t rle_decode(const uint8_t* src, size_t n, uint8_t* dst, size_t expected) {
    size_t ip = 0;
    size_t op = 0;
    while (ip < n) {
        if (n - ip < 2) throw std::runtime_error("rle_decode: truncated token header");
        const uint8_t tag = src[ip];
        const size_t len = static_cast<size_t>(src[ip + 1]) + 1;
        ip += 2;

        if (len > expected - op) throw std::runtime_error("rle_decode: output overrun");

        if (tag == kRleTokenZeroRun) {
            std::memset(dst + op, 0, len);
        } else if (tag == kRleTokenLiteral) {
            if (len > n - ip) throw std::runtime_error("rle_decode: truncated literal");
            std::memcpy(dst + op, src + ip, len);
            ip += len;
        } else {
            throw std::runtime_error("rle_decode: unknown token");
        }
        op += len;
    }
    if (op != expected) throw std::runtime_error("rle_decode: stream ended before the block was full");
    return op;
}

}  // namespace tc::codec
