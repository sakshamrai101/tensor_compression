#pragma once

#include "tensor.h"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

// TCF1: the compressed frame layout (architecture.md section 4).
//
//   frame   = header (64 bytes) + payload
//   payload = block_count block records
//   record  = u32 raw_len | u32 comp_len | u8 mode | u8[3] reserved | comp_len bytes
//
// Every multi-byte integer is little-endian and is serialized field by field,
// so the bytes on disk do not depend on host endianness or struct padding.

namespace tc {

// Stage flags. One format covers every ablation: a frame records which stages
// produced it, and the decoder undoes exactly those.
constexpr uint16_t FLAG_SHUFFLE = 1u << 0;
constexpr uint16_t FLAG_XOR     = 1u << 1;
constexpr uint16_t FLAG_RLE     = 1u << 2;
constexpr uint16_t FLAG_ALL     = FLAG_SHUFFLE | FLAG_XOR | FLAG_RLE;

constexpr uint32_t TCF1_MAGIC = 0x31464354u;    // bytes 'T' 'C' 'F' '1' in file order
constexpr uint16_t TCF1_VERSION = 1;
constexpr size_t TCF1_HEADER_BYTES = 64;
constexpr size_t TCF1_MAX_DIMS = 6;

// 64 KiB: a block plus its scratch copy stays inside a per-core L2 cache.
constexpr uint32_t DEFAULT_BLOCK_SIZE = 64u * 1024u;
constexpr uint32_t MIN_BLOCK_SIZE = 16u;
// Upper bound so a corrupt header cannot make the decoder allocate gigabytes of scratch.
constexpr uint32_t MAX_BLOCK_SIZE = 16u * 1024u * 1024u;

constexpr size_t BLOCK_RECORD_BYTES = 12;
constexpr uint8_t BLOCK_ENCODED = 0;    // bytes went through the stages named in header.flags
constexpr uint8_t BLOCK_STORED  = 1;    // stages did not shrink the block: raw bytes, verbatim

// In-memory mirror of the 64-byte header. Field order matches the byte layout.
struct Tcf1Header {
    uint32_t magic = TCF1_MAGIC;        // offset  0
    uint16_t version = TCF1_VERSION;    // offset  4
    uint16_t flags = 0;                 // offset  6
    uint8_t dtype = 0;                  // offset  8
    uint8_t ndims = 0;                  // offset  9   (0..6)
    uint16_t reserved0 = 0;             // offset 10
    uint32_t block_size = 0;            // offset 12
    uint32_t block_count = 0;           // offset 16
    uint32_t reserved1 = 0;             // offset 20
    uint64_t uncompressed_size = 0;     // offset 24
    uint64_t payload_size = 0;          // offset 32   bytes after the header
    uint32_t shape[TCF1_MAX_DIMS] = {}; // offset 40   first ndims used, rest 0
};

static_assert(sizeof(Tcf1Header) == TCF1_HEADER_BYTES, "Tcf1Header must be exactly 64 bytes");
static_assert(offsetof(Tcf1Header, magic) == 0);
static_assert(offsetof(Tcf1Header, version) == 4);
static_assert(offsetof(Tcf1Header, flags) == 6);
static_assert(offsetof(Tcf1Header, dtype) == 8);
static_assert(offsetof(Tcf1Header, ndims) == 9);
static_assert(offsetof(Tcf1Header, reserved0) == 10);
static_assert(offsetof(Tcf1Header, block_size) == 12);
static_assert(offsetof(Tcf1Header, block_count) == 16);
static_assert(offsetof(Tcf1Header, reserved1) == 20);
static_assert(offsetof(Tcf1Header, uncompressed_size) == 24);
static_assert(offsetof(Tcf1Header, payload_size) == 32);
static_assert(offsetof(Tcf1Header, shape) == 40);

// --- little-endian primitives -------------------------------------------------

inline void write_u16(uint8_t* p, uint16_t v) {
    p[0] = static_cast<uint8_t>(v);
    p[1] = static_cast<uint8_t>(v >> 8);
}
inline void write_u32(uint8_t* p, uint32_t v) {
    for (int i = 0; i < 4; ++i) p[i] = static_cast<uint8_t>(v >> (8 * i));
}
inline void write_u64(uint8_t* p, uint64_t v) {
    for (int i = 0; i < 8; ++i) p[i] = static_cast<uint8_t>(v >> (8 * i));
}
inline uint16_t read_u16(const uint8_t* p) {
    return static_cast<uint16_t>(p[0] | (p[1] << 8));
}
inline uint32_t read_u32(const uint8_t* p) {
    uint32_t v = 0;
    for (int i = 0; i < 4; ++i) v |= static_cast<uint32_t>(p[i]) << (8 * i);
    return v;
}
inline uint64_t read_u64(const uint8_t* p) {
    uint64_t v = 0;
    for (int i = 0; i < 8; ++i) v |= static_cast<uint64_t>(p[i]) << (8 * i);
    return v;
}

// --- header -------------------------------------------------------------------

// Builds a header for a tensor of this shape/dtype. block_count and
// uncompressed_size are derived; payload_size is left 0 for the caller.
// Throws std::invalid_argument for more than 6 dims, a dim above UINT32_MAX,
// unknown flag bits, or a block_size outside [MIN_BLOCK_SIZE, MAX_BLOCK_SIZE].
Tcf1Header make_header(const std::vector<size_t>& shape, DataType dtype,
                       uint16_t flags, uint32_t block_size);

// Writes the header into out[0..64). Throws std::invalid_argument if out is shorter.
void pack_header(const Tcf1Header& hdr, std::span<uint8_t> out);

// Reads and validates a header from data[0..64). Throws std::runtime_error on a
// short buffer, bad magic/version/dtype/ndims/flags, non-zero reserved fields,
// or sizes that disagree with each other.
Tcf1Header unpack_header(std::span<const uint8_t> data);

// --- block records ------------------------------------------------------------

struct BlockView {
    uint32_t raw_len = 0;
    uint32_t comp_len = 0;
    uint8_t mode = BLOCK_ENCODED;
    const uint8_t* bytes = nullptr;     // comp_len bytes inside the frame
    uint64_t raw_offset = 0;            // where this block's bytes go in the tensor
};

void write_block_record(uint8_t* p, uint32_t raw_len, uint32_t comp_len, uint8_t mode);

// Bounds-checked forward iterator over a frame's block records. Every decoder
// goes through this class, so the checks live in one place. Does not allocate.
class FrameReader {
public:
    // Validates the header and that the frame is exactly header + payload_size bytes.
    explicit FrameReader(std::span<const uint8_t> frame);

    const Tcf1Header& header() const { return hdr_; }

    // Fills `block` and returns true, or returns false after the last block.
    // Throws std::runtime_error on any inconsistency.
    bool next(BlockView& block);

private:
    Tcf1Header hdr_;
    const uint8_t* payload_ = nullptr;
    uint64_t pos_ = 0;
    uint64_t raw_done_ = 0;
    uint32_t index_ = 0;
};

struct FrameStats {
    Tcf1Header header;
    uint64_t frame_bytes = 0;
    uint32_t encoded_blocks = 0;
    uint32_t stored_blocks = 0;
    double ratio = 0.0;     // uncompressed_size / frame_bytes (0 for an empty tensor)
};

// Walks every block record (fully validated) and summarizes the frame.
FrameStats inspect_frame(std::span<const uint8_t> frame);

std::vector<size_t> header_shape(const Tcf1Header& hdr);

}  // namespace tc
