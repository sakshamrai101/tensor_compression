#include "compressor.h"
#include "format.h"
#include "test_harness.h"
#include "test_util.h"

#include <cstring>
#include <vector>

using namespace tc;

namespace {

std::vector<uint8_t> good_frame() {
    RawTensor t;
    t.dtype = DataType::FP32;
    t.shape = {32};
    t.data = tctest::random_bytes(128, 55, 0.5);
    TensorCompressor c;
    return c.compress(t).compressed_data;
}

}  // namespace

TC_TEST(malformed, truncated_frame) {
    auto frame = good_frame();
    frame.resize(TCF1_HEADER_BYTES + 4);
    TensorCompressor c;
    RawTensor out;
    TC_CHECK_THROWS(c.decompress_to(compressed_from_frame(std::move(frame)), out), std::runtime_error);
}

TC_TEST(malformed, bad_magic) {
    auto frame = good_frame();
    frame[0] = 'X';
    TC_CHECK_THROWS(unpack_header(frame), std::runtime_error);
}

TC_TEST(malformed, bad_block_comp_len) {
    auto frame = good_frame();
    // Overwrite first block's comp_len to claim more bytes than remain.
    write_u32(frame.data() + TCF1_HEADER_BYTES + 4, 0x00FFFFFFu);
    TensorCompressor c;
    RawTensor out;
    TC_CHECK_THROWS(c.decompress_to(compressed_from_frame(std::move(frame)), out), std::runtime_error);
}

TC_TEST(malformed, corrupt_rle_token) {
    // Build a tiny encoded-only frame then poke a bad RLE tag into the payload.
    CompressOptions opts;
    opts.flags = FLAG_RLE;  // RLE alone so payload is RLE tokens
    opts.block_size = 64;
    opts.stored_fallback = false;  // force encode path even if larger
    RawTensor t;
    t.dtype = DataType::INT8;
    t.shape = {40};
    t.data.assign(40, 1);  // non-zero -> literal tokens
    TensorCompressor c(opts);
    auto frame = c.compress(t).compressed_data;
    // First byte of first block body (after 12-byte record) should be a token.
    const size_t body = TCF1_HEADER_BYTES + BLOCK_RECORD_BYTES;
    TC_REQUIRE(frame.size() > body);
    frame[body] = 0x7F;  // unknown token
    RawTensor out;
    TC_CHECK_THROWS(c.decompress_to(compressed_from_frame(std::move(frame)), out), std::runtime_error);
}

TC_TEST(malformed, payload_size_mismatch) {
    auto frame = good_frame();
    Tcf1Header hdr = unpack_header(frame);
    hdr.payload_size += 1;
    pack_header(hdr, frame);
    TC_CHECK_THROWS(inspect_frame(frame), std::runtime_error);
}
