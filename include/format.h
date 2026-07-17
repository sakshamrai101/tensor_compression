#pragma once

#include "tensor.h"

#include <cstdint>
#include <cstddef>
#include <vector>


// TCF1 — on-wire / in-memory compressed frame (architecture.md §4)
// Fixed 64-byte header + blocked payload. Little-endian.

namespace tc {

// Flag bits (architecture.md §4.3) — usable once header fields exist
constexpr uint16_t FLAG_SHUFFLE = 1u << 0;
constexpr uint16_t FLAG_XOR     = 1u << 1;
constexpr uint16_t FLAG_RLE     = 1u << 2;

constexpr uint32_t TCF1_MAGIC = 0x31464354u; // 'T''C''F''1' little-endian bytes
constexpr uint16_t TCF1_VERSION = 1;
constexpr std::size_t TCF1_HEADER_BYTES = 64;

// =============================================================================
// HUMAN-TODO: Tcf1Header field layout
// WHY: The 64-byte header is the contract between compress and decompress.
//      Wrong offsets = corrupt frames. You should be able to whiteboard each field.
// DO: Fill this struct so it is exactly 64 bytes (static_assert below must pass).
//     Use the v1 simplification from architecture.md §4.1:
//       magic, version, flags, dtype, ndims, reserved0, block_size, block_count,
//       reserved1, uncompressed_size, payload_size, shape_packed[4] (uint32_t).
//     Prefer fixed-width types (uint8_t / uint16_t / uint32_t / uint64_t).
// HINT: Sum the sizes in the §4.1 table; shape_packed is uint32_t[4] for ndims ≤ 4.
// READ: architecture.md §4.1 (Header), §4.2 (Payload overview), plan.md D13
// =============================================================================

// contents for compressed file ono disk
struct Tcf1Header {
    uint32_t magic;
    uint16_t version;
    uint16_t flags;
    uint8_t dtype;
    uint8_t ndims;
    uint16_t reserved0;
    uint32_t block_size;
    uint32_t block_count;
    uint32_t reserved1;
    uint64_t uncompressed_size;
    uint64_t payload_size;

    // 4 slots, each uint32_t (4 bytes) -> 64 bytes, which is what the header has left at the end.
    uint32_t shape_packed[4];

};

struct Tcf1Header {
    // TODO: replace this placeholder so sizeof(Tcf1Header) == 64
    uint8_t bytes[TCF1_HEADER_BYTES]{};

};

// Once you fill real fields, change the placeholder and keep this assert.
static_assert(sizeof(Tcf1Header) == TCF1_HEADER_BYTES,
              "Tcf1Header must be exactly 64 bytes (architecture.md §4.1)");

// Pack header into the first 64 bytes of `out` (resizes / overwrites as needed).
void pack_header(const Tcf1Header& hdr, std::vector<uint8_t>& out);

// Unpack header from `data`; throws std::runtime_error on short buffer / bad magic.
Tcf1Header unpack_header(const std::vector<uint8_t>& data);

}  // namespace tc
