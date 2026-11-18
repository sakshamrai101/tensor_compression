#include "format.h"
#include "test_harness.h"

#include <array>
#include <cstring>

using namespace tc;

namespace {

Tcf1Header sample_header() {
    // 3 x 5 FP32 = 60 bytes, block_size 32 -> 2 blocks.
    Tcf1Header hdr = make_header({3, 5}, DataType::FP32, FLAG_ALL, 32);
    hdr.payload_size = 2 * BLOCK_RECORD_BYTES + 60;
    return hdr;
}

std::array<uint8_t, TCF1_HEADER_BYTES> packed_sample() {
    std::array<uint8_t, TCF1_HEADER_BYTES> bytes{};
    pack_header(sample_header(), bytes);
    return bytes;
}

}  // namespace

TC_TEST(format, header_is_64_bytes) {
    TC_CHECK(sizeof(Tcf1Header) == 64);
    TC_CHECK(TCF1_HEADER_BYTES == 64);
}

TC_TEST(format, byte_layout_matches_spec) {
    const auto b = packed_sample();

    // Magic is the ASCII bytes "TCF1" in file order.
    TC_CHECK(b[0] == 'T' && b[1] == 'C' && b[2] == 'F' && b[3] == '1');
    TC_CHECK(read_u16(b.data() + 4) == 1);              // version
    TC_CHECK(read_u16(b.data() + 6) == FLAG_ALL);       // flags
    TC_CHECK(b[8] == 0);                                // dtype FP32
    TC_CHECK(b[9] == 2);                                // ndims
    TC_CHECK(read_u16(b.data() + 10) == 0);             // reserved0
    TC_CHECK(read_u32(b.data() + 12) == 32);            // block_size
    TC_CHECK(read_u32(b.data() + 16) == 2);             // block_count
    TC_CHECK(read_u32(b.data() + 20) == 0);             // reserved1
    TC_CHECK(read_u64(b.data() + 24) == 60);            // uncompressed_size
    TC_CHECK(read_u64(b.data() + 32) == 84);            // payload_size
    TC_CHECK(read_u32(b.data() + 40) == 3);             // shape[0]
    TC_CHECK(read_u32(b.data() + 44) == 5);             // shape[1]
    for (size_t i = 48; i < 64; ++i) TC_CHECK(b[i] == 0);
}

TC_TEST(format, little_endian_primitives) {
    uint8_t buf[8] = {};
    write_u32(buf, 0x11223344u);
    TC_CHECK(buf[0] == 0x44 && buf[1] == 0x33 && buf[2] == 0x22 && buf[3] == 0x11);
    write_u64(buf, 0x0102030405060708ull);
    TC_CHECK(buf[0] == 0x08 && buf[7] == 0x01);
    TC_CHECK(read_u64(buf) == 0x0102030405060708ull);
    write_u16(buf, 0xBEEF);
    TC_CHECK(buf[0] == 0xEF && buf[1] == 0xBE && read_u16(buf) == 0xBEEF);
}

TC_TEST(format, pack_unpack_roundtrip) {
    const Tcf1Header in = sample_header();
    const auto bytes = packed_sample();
    const Tcf1Header out = unpack_header(bytes);

    TC_CHECK(out.magic == in.magic);
    TC_CHECK(out.version == in.version);
    TC_CHECK(out.flags == in.flags);
    TC_CHECK(out.dtype == in.dtype);
    TC_CHECK(out.ndims == in.ndims);
    TC_CHECK(out.block_size == in.block_size);
    TC_CHECK(out.block_count == in.block_count);
    TC_CHECK(out.uncompressed_size == in.uncompressed_size);
    TC_CHECK(out.payload_size == in.payload_size);
    for (size_t i = 0; i < TCF1_MAX_DIMS; ++i) TC_CHECK(out.shape[i] == in.shape[i]);

    const std::vector<size_t> shape = header_shape(out);
    TC_CHECK(shape.size() == 2 && shape[0] == 3 && shape[1] == 5);
}

TC_TEST(format, six_dims_and_all_dtypes_roundtrip) {
    for (DataType dtype : {DataType::FP32, DataType::FP16, DataType::INT8}) {
        Tcf1Header hdr = make_header({2, 3, 4, 5, 6, 7}, dtype, FLAG_SHUFFLE, DEFAULT_BLOCK_SIZE);
        hdr.payload_size = static_cast<uint64_t>(hdr.block_count) * BLOCK_RECORD_BYTES;
        std::array<uint8_t, 64> bytes{};
        pack_header(hdr, bytes);
        const Tcf1Header out = unpack_header(bytes);
        TC_CHECK(out.ndims == 6);
        TC_CHECK(out.uncompressed_size == 5040 * element_size(dtype));
        TC_CHECK(out.dtype == static_cast<uint8_t>(dtype));
    }
}

TC_TEST(format, empty_tensor_header) {
    for (const std::vector<size_t>& shape : {std::vector<size_t>{}, std::vector<size_t>{0}, std::vector<size_t>{4, 0}}) {
        const Tcf1Header hdr = make_header(shape, DataType::FP32, FLAG_ALL, DEFAULT_BLOCK_SIZE);
        TC_CHECK(hdr.uncompressed_size == 0);
        TC_CHECK(hdr.block_count == 0);
        std::array<uint8_t, 64> bytes{};
        pack_header(hdr, bytes);
        TC_CHECK(unpack_header(bytes).uncompressed_size == 0);
    }
}

TC_TEST(format, rejects_short_buffer) {
    const auto bytes = packed_sample();
    for (size_t len : {size_t{0}, size_t{1}, size_t{63}}) {
        TC_CHECK_THROWS(unpack_header(std::span<const uint8_t>(bytes.data(), len)), std::runtime_error);
    }
    std::array<uint8_t, 63> small{};
    TC_CHECK_THROWS(pack_header(sample_header(), small), std::invalid_argument);
}

TC_TEST(format, rejects_bad_fields) {
    struct Mutation {
        const char* what;
        size_t offset;
        uint8_t value;
    };
    const Mutation mutations[] = {
        {"bad magic", 0, 'X'},
        {"bad version", 4, 2},
        {"unknown flag bit", 6, 0x08},
        {"unknown flag bit (high byte)", 7, 0x01},
        {"unknown dtype", 8, 3},
        {"ndims > 6", 9, 7},
        {"reserved0 non-zero", 10, 1},
        {"block_size too small", 12, 1},
        {"block_count mismatch", 16, 3},
        {"reserved1 non-zero", 20, 1},
        {"uncompressed_size mismatch", 24, 61},
        {"payload too small for blocks", 32, 5},
        {"shape disagrees with size", 40, 4},
        {"unused shape slot non-zero", 60, 1},
    };
    for (const Mutation& m : mutations) {
        auto bytes = packed_sample();
        bytes[m.offset] = m.value;
        bool threw = false;
        try {
            unpack_header(bytes);
        } catch (const std::runtime_error&) {
            threw = true;
        }
        TC_CHECK_MSG(threw, m.what);
    }
}

TC_TEST(format, make_header_rejects_bad_arguments) {
    TC_CHECK_THROWS(make_header({1, 1, 1, 1, 1, 1, 1}, DataType::FP32, FLAG_ALL, DEFAULT_BLOCK_SIZE),
                    std::invalid_argument);
    TC_CHECK_THROWS(make_header({size_t{1} << 33}, DataType::INT8, FLAG_ALL, DEFAULT_BLOCK_SIZE),
                    std::invalid_argument);
    TC_CHECK_THROWS(make_header({4}, DataType::FP32, 0x10, DEFAULT_BLOCK_SIZE), std::invalid_argument);
    TC_CHECK_THROWS(make_header({4}, DataType::FP32, FLAG_ALL, MIN_BLOCK_SIZE - 1), std::invalid_argument);
    TC_CHECK_THROWS(make_header({4}, DataType::FP32, FLAG_ALL, MAX_BLOCK_SIZE + 1), std::invalid_argument);
}

TC_TEST(format, block_record_layout) {
    uint8_t rec[BLOCK_RECORD_BYTES];
    std::memset(rec, 0xAA, sizeof(rec));
    write_block_record(rec, 0x00010000u, 0x00000203u, BLOCK_STORED);
    TC_CHECK(read_u32(rec) == 0x00010000u);
    TC_CHECK(read_u32(rec + 4) == 0x00000203u);
    TC_CHECK(rec[8] == BLOCK_STORED);
    TC_CHECK(rec[9] == 0 && rec[10] == 0 && rec[11] == 0);
}
