#include "format.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>

namespace tc {

namespace {

[[noreturn]] void bad_frame(const char* what) {
    throw std::runtime_error(std::string("TCF1: ") + what);
}

// numel * element_size computed in 64 bits with overflow detection, from the
// header's own fields. Returns false on overflow.
bool header_byte_size(const Tcf1Header& hdr, uint64_t& out) {
    if (hdr.ndims == 0) {
        out = 0;
        return true;
    }
    uint64_t n = 1;
    for (size_t i = 0; i < hdr.ndims; ++i) {
        const uint64_t dim = hdr.shape[i];
        if (dim != 0 && n > std::numeric_limits<uint64_t>::max() / dim) return false;
        n *= dim;
    }
    const uint64_t elem = element_size(static_cast<DataType>(hdr.dtype));
    if (n > std::numeric_limits<uint64_t>::max() / elem) return false;
    out = n * elem;
    return true;
}

uint64_t blocks_for(uint64_t bytes, uint32_t block_size) {
    return bytes / block_size + (bytes % block_size != 0 ? 1 : 0);
}

}  // namespace

Tcf1Header make_header(const std::vector<size_t>& shape, DataType dtype,
                       uint16_t flags, uint32_t block_size) {
    if (shape.size() > TCF1_MAX_DIMS) {
        throw std::invalid_argument("TCF1 v1 supports at most 6 dimensions");
    }
    if ((flags & ~FLAG_ALL) != 0) {
        throw std::invalid_argument("TCF1: unknown stage flag bits");
    }
    if (block_size < MIN_BLOCK_SIZE || block_size > MAX_BLOCK_SIZE) {
        throw std::invalid_argument("TCF1: block_size out of range");
    }

    Tcf1Header hdr;
    hdr.flags = flags;
    hdr.dtype = static_cast<uint8_t>(dtype);
    hdr.ndims = static_cast<uint8_t>(shape.size());
    hdr.block_size = block_size;
    for (size_t i = 0; i < shape.size(); ++i) {
        if (shape[i] > std::numeric_limits<uint32_t>::max()) {
            throw std::invalid_argument("TCF1 v1 stores each dimension as u32");
        }
        hdr.shape[i] = static_cast<uint32_t>(shape[i]);
    }

    const uint64_t bytes = static_cast<uint64_t>(numel(shape)) * element_size(dtype);
    const uint64_t blocks = blocks_for(bytes, block_size);
    if (blocks > std::numeric_limits<uint32_t>::max()) {
        throw std::invalid_argument("TCF1: tensor needs more than 2^32 blocks");
    }
    hdr.uncompressed_size = bytes;
    hdr.block_count = static_cast<uint32_t>(blocks);
    return hdr;
}

void pack_header(const Tcf1Header& hdr, std::span<uint8_t> out) {
    if (out.size() < TCF1_HEADER_BYTES) {
        throw std::invalid_argument("pack_header: output shorter than 64 bytes");
    }
    uint8_t* p = out.data();
    write_u32(p + 0, hdr.magic);
    write_u16(p + 4, hdr.version);
    write_u16(p + 6, hdr.flags);
    p[8] = hdr.dtype;
    p[9] = hdr.ndims;
    write_u16(p + 10, hdr.reserved0);
    write_u32(p + 12, hdr.block_size);
    write_u32(p + 16, hdr.block_count);
    write_u32(p + 20, hdr.reserved1);
    write_u64(p + 24, hdr.uncompressed_size);
    write_u64(p + 32, hdr.payload_size);
    for (size_t i = 0; i < TCF1_MAX_DIMS; ++i) {
        write_u32(p + 40 + 4 * i, hdr.shape[i]);
    }
}

Tcf1Header unpack_header(std::span<const uint8_t> data) {
    if (data.size() < TCF1_HEADER_BYTES) bad_frame("buffer shorter than the 64-byte header");

    const uint8_t* p = data.data();
    Tcf1Header hdr;
    hdr.magic = read_u32(p + 0);
    hdr.version = read_u16(p + 4);
    hdr.flags = read_u16(p + 6);
    hdr.dtype = p[8];
    hdr.ndims = p[9];
    hdr.reserved0 = read_u16(p + 10);
    hdr.block_size = read_u32(p + 12);
    hdr.block_count = read_u32(p + 16);
    hdr.reserved1 = read_u32(p + 20);
    hdr.uncompressed_size = read_u64(p + 24);
    hdr.payload_size = read_u64(p + 32);
    for (size_t i = 0; i < TCF1_MAX_DIMS; ++i) {
        hdr.shape[i] = read_u32(p + 40 + 4 * i);
    }

    if (hdr.magic != TCF1_MAGIC) bad_frame("bad magic");
    if (hdr.version != TCF1_VERSION) bad_frame("unsupported version");
    if ((hdr.flags & ~FLAG_ALL) != 0) bad_frame("unknown flag bits");
    if (hdr.dtype > static_cast<uint8_t>(DataType::INT8)) bad_frame("unknown dtype");
    if (hdr.ndims > TCF1_MAX_DIMS) bad_frame("ndims > 6");
    if (hdr.reserved0 != 0 || hdr.reserved1 != 0) bad_frame("reserved fields must be zero");
    if (hdr.block_size < MIN_BLOCK_SIZE || hdr.block_size > MAX_BLOCK_SIZE) {
        bad_frame("block_size out of range");
    }
    for (size_t i = hdr.ndims; i < TCF1_MAX_DIMS; ++i) {
        if (hdr.shape[i] != 0) bad_frame("unused shape slots must be zero");
    }

    uint64_t expected_bytes = 0;
    if (!header_byte_size(hdr, expected_bytes)) bad_frame("shape overflows");
    if (hdr.uncompressed_size != expected_bytes) {
        bad_frame("uncompressed_size does not match shape * element size");
    }
    if (hdr.block_count != blocks_for(hdr.uncompressed_size, hdr.block_size)) {
        bad_frame("block_count does not match uncompressed_size / block_size");
    }
    // Each block costs at least its 12-byte record.
    if (hdr.payload_size < static_cast<uint64_t>(hdr.block_count) * BLOCK_RECORD_BYTES) {
        bad_frame("payload_size too small for block_count");
    }
    return hdr;
}

void write_block_record(uint8_t* p, uint32_t raw_len, uint32_t comp_len, uint8_t mode) {
    write_u32(p + 0, raw_len);
    write_u32(p + 4, comp_len);
    p[8] = mode;
    p[9] = 0;
    p[10] = 0;
    p[11] = 0;
}

FrameReader::FrameReader(std::span<const uint8_t> frame) : hdr_(unpack_header(frame)) {
    if (frame.size() - TCF1_HEADER_BYTES != hdr_.payload_size) {
        bad_frame("frame length does not equal header + payload_size");
    }
    // The densest token (a zero run) turns 2 bytes into 256, so no payload can
    // decode to more than 128x its size. Without RLE every stage preserves
    // length. Checking this before anyone sizes an output buffer keeps a
    // corrupt header from requesting an absurd allocation.
    const uint64_t max_expansion = (hdr_.flags & FLAG_RLE) != 0 ? 128 : 1;
    if (hdr_.uncompressed_size / max_expansion > hdr_.payload_size) {
        bad_frame("header claims more output than the payload can encode");
    }
    payload_ = frame.data() + TCF1_HEADER_BYTES;
}

bool FrameReader::next(BlockView& block) {
    if (index_ == hdr_.block_count) {
        if (pos_ != hdr_.payload_size) bad_frame("trailing bytes after the last block");
        if (raw_done_ != hdr_.uncompressed_size) bad_frame("blocks do not add up to uncompressed_size");
        return false;
    }
    if (hdr_.payload_size - pos_ < BLOCK_RECORD_BYTES) bad_frame("truncated block record");

    const uint8_t* rec = payload_ + pos_;
    const uint32_t raw_len = read_u32(rec + 0);
    const uint32_t comp_len = read_u32(rec + 4);
    const uint8_t mode = rec[8];
    pos_ += BLOCK_RECORD_BYTES;

    // Blocks are fixed-size except the last, so raw_len is fully determined by
    // the header. Checking it here means a decoder can trust raw_len as a
    // write bound into the output tensor.
    const uint64_t remaining = hdr_.uncompressed_size - raw_done_;
    const uint64_t expected_raw = std::min<uint64_t>(hdr_.block_size, remaining);
    if (raw_len != expected_raw) bad_frame("block raw_len disagrees with header");
    if (mode != BLOCK_ENCODED && mode != BLOCK_STORED) bad_frame("unknown block mode");
    if (rec[9] != 0 || rec[10] != 0 || rec[11] != 0) bad_frame("block reserved bytes must be zero");
    if (comp_len > hdr_.payload_size - pos_) bad_frame("block comp_len runs past the payload");
    if (mode == BLOCK_STORED && comp_len != raw_len) bad_frame("stored block with comp_len != raw_len");

    block.raw_len = raw_len;
    block.comp_len = comp_len;
    block.mode = mode;
    block.bytes = payload_ + pos_;
    block.raw_offset = raw_done_;

    pos_ += comp_len;
    raw_done_ += raw_len;
    ++index_;
    return true;
}

FrameStats inspect_frame(std::span<const uint8_t> frame) {
    FrameReader reader(frame);
    FrameStats stats;
    stats.header = reader.header();
    stats.frame_bytes = frame.size();

    BlockView block;
    while (reader.next(block)) {
        if (block.mode == BLOCK_STORED) {
            ++stats.stored_blocks;
        } else {
            ++stats.encoded_blocks;
        }
    }
    if (stats.header.uncompressed_size != 0) {
        stats.ratio = static_cast<double>(stats.header.uncompressed_size) /
                      static_cast<double>(stats.frame_bytes);
    }
    return stats;
}

std::vector<size_t> header_shape(const Tcf1Header& hdr) {
    return std::vector<size_t>(hdr.shape, hdr.shape + hdr.ndims);
}

}  // namespace tc
