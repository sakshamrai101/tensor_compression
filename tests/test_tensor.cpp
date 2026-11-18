#include "synthetic.h"
#include "tensor.h"
#include "test_harness.h"

#include <cmath>
#include <cstring>
#include <limits>

using namespace tc;

namespace {

std::vector<float> as_floats(const RawTensor& t) {
    std::vector<float> values(t.data.size() / sizeof(float));
    if (!values.empty()) std::memcpy(values.data(), t.data.data(), t.data.size());
    return values;
}

}  // namespace

TC_TEST(tensor, element_size) {
    TC_CHECK(element_size(DataType::FP32) == 4);
    TC_CHECK(element_size(DataType::FP16) == 2);
    TC_CHECK(element_size(DataType::INT8) == 1);
    TC_CHECK_THROWS(element_size(static_cast<DataType>(9)), std::invalid_argument);
}

TC_TEST(tensor, numel_edge_cases) {
    TC_CHECK(numel({}) == 0);               // empty shape is "no tensor", not a scalar
    TC_CHECK(numel({0}) == 0);
    TC_CHECK(numel({7}) == 7);
    TC_CHECK(numel({3, 4}) == 12);
    TC_CHECK(numel({5, 0, 9}) == 0);
    TC_CHECK(numel({2, 3, 4, 5, 6, 7}) == 5040);
    const size_t big = std::numeric_limits<size_t>::max() / 2 + 1;
    TC_CHECK_THROWS(numel({big, 2}), std::overflow_error);
    TC_CHECK(numel({big, 0}) == 0);
}

TC_TEST(tensor, validate) {
    RawTensor t;
    t.dtype = DataType::FP32;
    t.shape = {2, 3};
    t.data.assign(24, 0);
    validate(t);    // must not throw

    t.data.assign(23, 0);
    TC_CHECK_THROWS(validate(t), std::invalid_argument);

    t.dtype = DataType::INT8;
    t.data.assign(6, 0);
    validate(t);

    RawTensor empty;
    validate(empty);
}

TC_TEST(tensor, dtype_names) {
    for (DataType d : {DataType::FP32, DataType::FP16, DataType::INT8}) {
        TC_CHECK(parse_dtype(dtype_name(d)) == d);
    }
    TC_CHECK_THROWS(parse_dtype("float64"), std::invalid_argument);
}

TC_TEST(tensor, generators_are_deterministic) {
    for (Dist dist : {Dist::Zeros, Dist::RandomBits, Dist::Gaussian, Dist::Ar1Gaussian,
                      Dist::SharedExponent, Dist::QuantizedInt8}) {
        SynthSpec spec;
        spec.dist = dist;
        spec.shape = {37, 91};
        spec.seed = 42;
        spec.dtype = dist == Dist::QuantizedInt8 ? DataType::INT8 : DataType::FP32;

        const RawTensor a = generate(spec);
        const RawTensor b = generate(spec);
        validate(a);
        TC_CHECK_MSG(a.data == b.data, dist_name(dist));

        if (dist != Dist::Zeros) {
            spec.seed = 43;
            TC_CHECK_MSG(generate(spec).data != a.data, dist_name(dist));
        }
        TC_CHECK(parse_dist(dist_name(dist)) == dist);
    }
}

TC_TEST(tensor, gaussian_has_requested_moments) {
    SynthSpec spec;
    spec.dist = Dist::Gaussian;
    spec.shape = {400, 500};
    spec.sigma = 0.02;
    const std::vector<float> x = as_floats(generate(spec));

    double sum = 0.0, sum_sq = 0.0;
    for (float v : x) {
        sum += v;
        sum_sq += static_cast<double>(v) * v;
    }
    const double mean = sum / static_cast<double>(x.size());
    const double stddev = std::sqrt(sum_sq / static_cast<double>(x.size()) - mean * mean);
    TC_CHECK(std::fabs(mean) < 2e-4);                   // ~4.5 standard errors
    TC_CHECK(std::fabs(stddev - 0.02) < 2e-4);
}

TC_TEST(tensor, ar1_has_requested_correlation) {
    for (double rho : {0.5, 0.9, 0.99}) {
        SynthSpec spec;
        spec.dist = Dist::Ar1Gaussian;
        spec.shape = {1000, 1000};
        spec.rho = rho;
        const std::vector<float> x = as_floats(generate(spec));

        double sxx = 0.0, sxy = 0.0;
        for (size_t i = 0; i + 1 < x.size(); ++i) {
            sxx += static_cast<double>(x[i]) * x[i];
            sxy += static_cast<double>(x[i]) * x[i + 1];
        }
        TC_CHECK(std::fabs(sxy / sxx - rho) < 0.02);
    }
}

TC_TEST(tensor, shared_exponent_is_one_binade) {
    SynthSpec spec;
    spec.dist = Dist::SharedExponent;
    spec.shape = {10000};
    spec.sigma = 0.02;                                  // snaps down to 2^-6 = 0.015625
    const std::vector<float> x = as_floats(generate(spec));
    for (float v : x) {
        TC_CHECK(v >= 0.015625f && v < 0.03125f);
    }
}

TC_TEST(tensor, quantized_int8_is_clipped_and_centred) {
    SynthSpec spec;
    spec.dist = Dist::QuantizedInt8;
    spec.dtype = DataType::INT8;
    spec.shape = {100000};
    const RawTensor t = generate(spec);
    long long sum = 0;
    for (uint8_t byte : t.data) {
        const int v = static_cast<int8_t>(byte);
        TC_CHECK(v >= -127 && v <= 127);
        sum += v;
    }
    TC_CHECK(std::llabs(sum) < 100000 / 2);             // |mean| < 0.5 quantization steps
}

TC_TEST(tensor, generators_enforce_dtype) {
    SynthSpec spec;
    spec.shape = {8};
    spec.dist = Dist::Gaussian;
    spec.dtype = DataType::INT8;
    TC_CHECK_THROWS(generate(spec), std::invalid_argument);
    spec.dist = Dist::QuantizedInt8;
    spec.dtype = DataType::FP32;
    TC_CHECK_THROWS(generate(spec), std::invalid_argument);
    spec.dist = Dist::Ar1Gaussian;
    spec.rho = 1.0;
    TC_CHECK_THROWS(generate(spec), std::invalid_argument);
}
