#include "compressor.h"
#include "synthetic.h"
#include "test_harness.h"
#include "test_util.h"

#include <cstring>
#include <string>
#include <vector>

using namespace tc;

namespace {

RawTensor make_tensor(std::vector<uint8_t> data, std::vector<size_t> shape, DataType dtype) {
    RawTensor t;
    t.data = std::move(data);
    t.shape = std::move(shape);
    t.dtype = dtype;
    validate(t);
    return t;
}

bool roundtrip_ok(const RawTensor& in, const CompressOptions& opts) {
    TensorCompressor c(opts);
    CompressedTensor mid = c.compress(in);
    RawTensor out = c.decompress(mid);
    return out.dtype == in.dtype && out.shape == in.shape && out.data == in.data;
}

const uint16_t kAllFlagCombos[] = {
    0,
    FLAG_SHUFFLE,
    FLAG_XOR,
    FLAG_RLE,
    FLAG_SHUFFLE | FLAG_XOR,
    FLAG_SHUFFLE | FLAG_RLE,
    FLAG_XOR | FLAG_RLE,
    FLAG_ALL,
};

}  // namespace

TC_TEST(roundtrip, empty_tensor) {
    CompressOptions opts;
    opts.block_size = 64;
    TC_CHECK(roundtrip_ok(make_tensor({}, {}, DataType::FP32), opts));
}

TC_TEST(roundtrip, flag_dtype_size_matrix) {
    const size_t sizes[] = {0, 1, 3, 4, 5, 63, 64, 65, 65535, 65536, 65537, 3 * 65536 + 7};
    const DataType dtypes[] = {DataType::FP32, DataType::FP16, DataType::INT8};
    CompressOptions opts;
    opts.block_size = 65536;
    opts.stored_fallback = true;

    for (DataType dtype : dtypes) {
        const size_t elem = element_size(dtype);
        for (size_t nbytes : sizes) {
            // Build a flat shape that matches nbytes (pad with a trailing dim if needed).
            if (nbytes % elem != 0) continue;  // skip impossible sizes for this dtype
            const size_t n = nbytes / elem;
            auto bytes = tctest::random_bytes(nbytes, 1000 + nbytes + static_cast<size_t>(dtype), 0.25);
            RawTensor in = make_tensor(std::move(bytes), {n}, dtype);
            for (uint16_t flags : kAllFlagCombos) {
                opts.flags = flags;
                TC_CHECK_MSG(roundtrip_ok(in, opts),
                             "dtype=" + std::string(dtype_name(dtype)) +
                                 " n=" + std::to_string(nbytes) +
                                 " flags=" + std::to_string(flags));
            }
        }
    }
}

TC_TEST(roundtrip, synthetic_distributions) {
    const Dist dists[] = {Dist::Zeros, Dist::RandomBits, Dist::Gaussian, Dist::Ar1Gaussian,
                          Dist::SharedExponent, Dist::QuantizedInt8};
    CompressOptions opts;
    opts.flags = FLAG_ALL;
    opts.block_size = 4096;

    for (Dist dist : dists) {
        SynthSpec spec;
        spec.dist = dist;
        spec.seed = 42;
        spec.shape = {128, 64};
        if (dist == Dist::QuantizedInt8) spec.dtype = DataType::INT8;
        RawTensor in = generate(spec);
        TC_CHECK_MSG(roundtrip_ok(in, opts), dist_name(dist));
    }
}

TC_TEST(roundtrip, seeded_random_loop) {
    CompressOptions opts;
    opts.flags = FLAG_ALL;
    opts.block_size = 1024;
    for (uint64_t seed = 1; seed <= 500; ++seed) {
        const size_t n = 16 + static_cast<size_t>(seed % 2000);
        auto bytes = tctest::random_bytes(n * 4, seed, 0.3);
        RawTensor in = make_tensor(std::move(bytes), {n}, DataType::FP32);
        TC_REQUIRE(roundtrip_ok(in, opts));
    }
}

TC_TEST(roundtrip, zeros_and_ar1_ratio_above_one) {
    CompressOptions opts;
    opts.flags = FLAG_ALL;
    opts.block_size = 65536;

    SynthSpec zeros;
    zeros.dist = Dist::Zeros;
    zeros.shape = {256, 256};
    zeros.seed = 1;
    TensorCompressor c(opts);
    CompressedTensor z = c.compress(generate(zeros));
    TC_CHECK(inspect_frame(z.compressed_data).ratio > 1.0);

    SynthSpec ar1;
    ar1.dist = Dist::Ar1Gaussian;
    ar1.shape = {256, 256};
    ar1.seed = 2;
    ar1.rho = 0.99;
    CompressedTensor a = c.compress(generate(ar1));
    TC_CHECK(inspect_frame(a.compressed_data).ratio > 1.0);
}

TC_TEST(roundtrip, random_bits_ratio_floor) {
    CompressOptions opts;
    opts.flags = FLAG_ALL;
    opts.block_size = 65536;
    SynthSpec spec;
    spec.dist = Dist::RandomBits;
    spec.shape = {64, 1024};
    spec.seed = 7;
    TensorCompressor c(opts);
    CompressedTensor mid = c.compress(generate(spec));
    // Stored fallback + header overhead: ratio should stay near 1.0, never ~0.5.
    TC_CHECK(inspect_frame(mid.compressed_data).ratio >= 0.98);
}

TC_TEST(roundtrip, decompress_to_reuse) {
    SynthSpec spec;
    spec.dist = Dist::Gaussian;
    spec.shape = {64, 64};
    spec.seed = 3;
    RawTensor in = generate(spec);
    TensorCompressor c;
    CompressedTensor mid = c.compress(in);
    RawTensor out;
    c.decompress_to(mid, out);
    TC_CHECK(out.data == in.data);
    // Second call on same-sized frame should still round-trip.
    c.decompress_to(mid, out);
    TC_CHECK(out.data == in.data);
}
