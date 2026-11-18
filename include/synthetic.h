#pragma once

#include "tensor.h"

#include <cstdint>
#include <string_view>
#include <vector>

// Deterministic synthetic tensors for tests and benchmarks.
//
// Same spec -> same bytes on a given platform. Normal variates come from an
// in-repo Box-Muller transform over std::mt19937_64 rather than
// std::normal_distribution, whose output differs between standard libraries.
// (The transform still calls libm log/sqrt/cos/sin, so bytes can differ in the
// last bit between platforms with different libm implementations.)

namespace tc {

enum class Dist {
    Zeros,           // all bytes 0: sanity upper bound for any codec
    RandomBits,      // uniform random bytes: incompressible lower bound
    Gaussian,        // iid N(0, sigma^2) FP32: the honest "weight-like" case
    Ar1Gaussian,     // x[t] = rho*x[t-1] + sqrt(1-rho^2)*sigma*z[t]: neighbours correlated.
                     // Illustrative only; it favours delta coding by construction.
    SharedExponent,  // FP32 in [sigma, 2*sigma): one sign, one exponent, random mantissa
    QuantizedInt8,   // INT8: iid Gaussian quantized symmetrically to [-127, 127]
};

struct SynthSpec {
    Dist dist = Dist::Gaussian;
    std::vector<size_t> shape;
    uint64_t seed = 1;
    double sigma = 0.02;    // weight-init scale
    double rho = 0.9;       // Ar1Gaussian only
    // RandomBits and Zeros honour this; the float distributions require FP32
    // and QuantizedInt8 requires INT8.
    DataType dtype = DataType::FP32;
};

RawTensor generate(const SynthSpec& spec);

const char* dist_name(Dist dist);
Dist parse_dist(std::string_view name);   // zeros|random_bits|gaussian|ar1_gaussian|shared_exponent|quantized_int8

}  // namespace tc
