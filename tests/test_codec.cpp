#include "codec/byte_shuffle.h"
#include "codec/rle.h"
#include "codec/xor_delta.h"
#include "test_harness.h"
#include "test_util.h"

#include <cstring>
#include <vector>

using namespace tc;
using namespace tc::codec;

TC_TEST(codec, byte_shuffle_roundtrip_fp32) {
    const auto src = tctest::random_bytes(4 * 17 + 3, /*seed=*/7);  // tail of 3
    std::vector<uint8_t> shuffled(src.size()), restored(src.size());
    byte_shuffle_scalar(src.data(), shuffled.data(), src.size(), 4);
    byte_unshuffle_scalar(shuffled.data(), restored.data(), src.size(), 4);
    TC_CHECK(restored == src);
    // Tail bytes are copied through unchanged.
    TC_CHECK(std::memcmp(shuffled.data() + 4 * 17, src.data() + 4 * 17, 3) == 0);
}

TC_TEST(codec, byte_shuffle_identity_for_int8) {
    const auto src = tctest::random_bytes(64, 11);
    std::vector<uint8_t> out(src.size());
    byte_shuffle_scalar(src.data(), out.data(), src.size(), 1);
    TC_CHECK(out == src);
}

TC_TEST(codec, xor_delta_roundtrip) {
    const auto src = tctest::random_bytes(1000, 13, /*zero_fraction=*/0.4);
    std::vector<uint8_t> enc(src.size()), dec(src.size());
    xor_delta_encode(src.data(), enc.data(), src.size());
    xor_delta_decode(enc.data(), dec.data(), src.size());
    TC_CHECK(dec == src);

    // In-place decode.
    std::vector<uint8_t> inplace = enc;
    xor_delta_decode(inplace.data(), inplace.data(), inplace.size());
    TC_CHECK(inplace == src);
}

TC_TEST(codec, xor_equal_bytes_become_zeros) {
    std::vector<uint8_t> src(32, 0xAB);
    std::vector<uint8_t> enc(32);
    xor_delta_encode(src.data(), enc.data(), src.size());
    TC_CHECK(enc[0] == 0xAB);
    for (size_t i = 1; i < enc.size(); ++i) TC_CHECK(enc[i] == 0);
}

TC_TEST(codec, rle_zeros_and_literals) {
    std::vector<uint8_t> src(300, 0);
    for (size_t i = 50; i < 80; ++i) src[i] = static_cast<uint8_t>(i);
    std::vector<uint8_t> enc(rle_max_encoded_size(src.size()));
    const size_t n = rle_encode(src.data(), src.size(), enc.data(), enc.size());
    TC_CHECK(n < src.size());  // zeros should compress
    std::vector<uint8_t> dec(src.size());
    TC_CHECK(rle_decode(enc.data(), n, dec.data(), src.size()) == src.size());
    TC_CHECK(dec == src);
}

TC_TEST(codec, rle_empty) {
    std::vector<uint8_t> enc(rle_max_encoded_size(0));
    const size_t n = rle_encode(nullptr, 0, enc.data(), enc.size());
    TC_CHECK(n == 0);
    TC_CHECK(rle_decode(enc.data(), 0, nullptr, 0) == 0);
}

TC_TEST(codec, rle_rejects_truncated) {
    const uint8_t bad[] = {kRleTokenLiteral, 5};  // claims 6 literal bytes, none present
    uint8_t out[16] = {};
    TC_CHECK_THROWS(rle_decode(bad, 2, out, 16), std::runtime_error);
}

TC_TEST(codec, rle_rejects_unknown_token) {
    const uint8_t bad[] = {0x02, 0};
    uint8_t out[1] = {};
    TC_CHECK_THROWS(rle_decode(bad, 2, out, 1), std::runtime_error);
}

TC_TEST(codec, rle_worst_case_bound) {
    // Incompressible: every byte non-zero and unique-ish.
    const auto src = tctest::random_bytes(4096, 99, /*zero_fraction=*/0.0);
    // Force no zeros so everything is literals.
    std::vector<uint8_t> nozero = src;
    for (uint8_t& b : nozero) if (b == 0) b = 1;
    std::vector<uint8_t> enc(rle_max_encoded_size(nozero.size()));
    const size_t n = rle_encode(nozero.data(), nozero.size(), enc.data(), enc.size());
    TC_CHECK(n <= rle_max_encoded_size(nozero.size()));
    TC_CHECK(n <= nozero.size() + nozero.size() / 128 + 16);
    std::vector<uint8_t> dec(nozero.size());
    TC_CHECK(rle_decode(enc.data(), n, dec.data(), nozero.size()) == nozero.size());
    TC_CHECK(dec == nozero);
}
