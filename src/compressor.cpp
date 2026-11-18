#include "compressor.h"

#include "codec/block.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <utility>

namespace tc {

namespace {

uint32_t ratio_x100(uint64_t original_bytes, uint64_t frame_bytes) {
    if (original_bytes == 0 || frame_bytes == 0) return 0;
    const double ratio = static_cast<double>(original_bytes) / static_cast<double>(frame_bytes);
    return static_cast<uint32_t>(std::llround(ratio * 100.0));
}

void check_options(const CompressOptions& opts) {
    if ((opts.flags & ~FLAG_ALL) != 0) {
        throw std::invalid_argument("CompressOptions: unknown stage flag bits");
    }
    if (opts.block_size < MIN_BLOCK_SIZE || opts.block_size > MAX_BLOCK_SIZE) {
        throw std::invalid_argument("CompressOptions: block_size out of range");
    }
}

}  // namespace

TensorCompressor::TensorCompressor() : TensorCompressor(CompressOptions{}) {}

TensorCompressor::TensorCompressor(const CompressOptions& options) : opts_(options) {
    check_options(opts_);
}

void TensorCompressor::reserve_scratch(size_t block_bytes, bool for_encode) {
    // Grow-only: after the first block of a given size these are no-ops, which
    // is what keeps the decode hot path allocation-free.
    if (scratch_a_.size() < block_bytes) scratch_a_.resize(block_bytes);
    if (for_encode && scratch_b_.size() < block_bytes) scratch_b_.resize(block_bytes);
}

size_t TensorCompressor::max_frame_size(size_t raw_bytes, uint32_t block_size) {
    const size_t blocks = raw_bytes / block_size + (raw_bytes % block_size != 0 ? 1 : 0);
    // Per block: record + worst-case RLE output. With the stored fallback a
    // block never exceeds record + raw_len, but the encoder writes its attempt
    // into the frame first, so capacity must cover the attempt.
    return TCF1_HEADER_BYTES + blocks * (BLOCK_RECORD_BYTES + codec::max_encoded_block_size(block_size));
}

size_t TensorCompressor::compress_frame(std::span<const uint8_t> raw, const std::vector<size_t>& shape,
                                        DataType dtype, std::span<uint8_t> out_frame) {
    Tcf1Header hdr = make_header(shape, dtype, opts_.flags, opts_.block_size);
    if (hdr.uncompressed_size != raw.size()) {
        throw std::invalid_argument("compress_frame: raw size does not match shape * element size");
    }
    if (out_frame.size() < max_frame_size(raw.size(), opts_.block_size)) {
        throw std::invalid_argument("compress_frame: output smaller than max_frame_size()");
    }

    const size_t elem = element_size(dtype);
    reserve_scratch(std::min<size_t>(opts_.block_size, raw.size()), /*for_encode=*/true);

    uint8_t* const frame = out_frame.data();
    size_t pos = TCF1_HEADER_BYTES;
    size_t offset = 0;
    while (offset < raw.size()) {
        const size_t raw_len = std::min<size_t>(opts_.block_size, raw.size() - offset);
        const uint8_t* block = raw.data() + offset;
        uint8_t* body = frame + pos + BLOCK_RECORD_BYTES;

        size_t comp_len = codec::encode_block(block, raw_len, elem, opts_.flags, body,
                                              scratch_a_.data(), scratch_b_.data());
        uint8_t mode = BLOCK_ENCODED;
        if (opts_.stored_fallback && comp_len >= raw_len) {
            std::memcpy(body, block, raw_len);
            comp_len = raw_len;
            mode = BLOCK_STORED;
        }
        write_block_record(frame + pos, static_cast<uint32_t>(raw_len),
                           static_cast<uint32_t>(comp_len), mode);
        pos += BLOCK_RECORD_BYTES + comp_len;
        offset += raw_len;
    }

    hdr.payload_size = pos - TCF1_HEADER_BYTES;
    pack_header(hdr, out_frame);
    return pos;
}

Tcf1Header TensorCompressor::decompress_frame(std::span<const uint8_t> frame, std::span<uint8_t> out) {
    FrameReader reader(frame);
    const Tcf1Header& hdr = reader.header();
    if (out.size() != hdr.uncompressed_size) {
        throw std::invalid_argument("decompress_frame: output size != frame uncompressed_size");
    }

    const size_t elem = element_size(static_cast<DataType>(hdr.dtype));
    reserve_scratch(std::min<uint64_t>(hdr.block_size, hdr.uncompressed_size), /*for_encode=*/false);

    BlockView block;
    while (reader.next(block)) {
        // FrameReader guarantees raw_offset + raw_len <= uncompressed_size == out.size().
        uint8_t* dst = out.data() + block.raw_offset;
        if (block.mode == BLOCK_STORED) {
            std::memcpy(dst, block.bytes, block.raw_len);
        } else {
            codec::decode_block(block.bytes, block.comp_len, elem, hdr.flags, dst, block.raw_len,
                                scratch_a_.data());
        }
    }
    return hdr;
}

void TensorCompressor::compress_to(const RawTensor& in, CompressedTensor& out) {
    validate(in);
    out.compressed_data.resize(max_frame_size(in.data.size(), opts_.block_size));
    const size_t frame_bytes = compress_frame(in.data, in.shape, in.dtype, out.compressed_data);
    out.compressed_data.resize(frame_bytes);

    out.shape = in.shape;
    out.dtype = in.dtype;
    out.original_size_bytes = in.data.size();
    out.block_count = unpack_header(out.compressed_data).block_count;
    out.compression_ratio_x100 = ratio_x100(in.data.size(), frame_bytes);
}

void TensorCompressor::decompress_to(const CompressedTensor& in, RawTensor& out) {
    // FrameReader's constructor rejects a frame that claims more output than
    // its payload could encode, so a corrupt header cannot drive a huge resize.
    const Tcf1Header hdr = FrameReader(in.compressed_data).header();
    // No-ops when `out` already has this size and rank.
    out.data.resize(hdr.uncompressed_size);
    out.shape.assign(hdr.shape, hdr.shape + hdr.ndims);
    out.dtype = static_cast<DataType>(hdr.dtype);
    decompress_frame(in.compressed_data, out.data);
}

CompressedTensor TensorCompressor::compress(const RawTensor& tensor) {
    CompressedTensor out;
    compress_to(tensor, out);
    return out;
}

RawTensor TensorCompressor::decompress(const CompressedTensor& compressed) {
    RawTensor out;
    decompress_to(compressed, out);
    return out;
}

CompressedTensor compressed_from_frame(std::vector<uint8_t> frame) {
    const FrameStats stats = inspect_frame(frame);
    CompressedTensor out;
    out.shape = header_shape(stats.header);
    out.dtype = static_cast<DataType>(stats.header.dtype);
    out.original_size_bytes = stats.header.uncompressed_size;
    out.block_count = stats.header.block_count;
    out.compression_ratio_x100 = ratio_x100(stats.header.uncompressed_size, stats.frame_bytes);
    out.compressed_data = std::move(frame);
    return out;
}

}  // namespace tc
