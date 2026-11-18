#include "synthetic.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>

namespace tc {

namespace {

constexpr double kTwoPi = 6.283185307179586476925286766559;

// Standard normal variates via Box-Muller. Each transform yields two values;
// the second is cached.
class NormalSource {
public:
    explicit NormalSource(uint64_t seed) : rng_(seed) {}

    double next() {
        if (has_spare_) {
            has_spare_ = false;
            return spare_;
        }
        // 53 random bits -> u1 in (0, 1] (so log is finite), u2 in [0, 1).
        const double u1 = (static_cast<double>(rng_() >> 11) + 1.0) * 0x1.0p-53;
        const double u2 = static_cast<double>(rng_() >> 11) * 0x1.0p-53;
        const double radius = std::sqrt(-2.0 * std::log(u1));
        spare_ = radius * std::sin(kTwoPi * u2);
        has_spare_ = true;
        return radius * std::cos(kTwoPi * u2);
    }

    std::mt19937_64& bits() { return rng_; }

private:
    std::mt19937_64 rng_;
    double spare_ = 0.0;
    bool has_spare_ = false;
};

void store_f32(std::vector<uint8_t>& data, size_t index, float value) {
    std::memcpy(data.data() + index * sizeof(float), &value, sizeof(float));
}

void require_dtype(const SynthSpec& spec, DataType want) {
    if (spec.dtype != want) {
        throw std::invalid_argument(std::string("generate: dist '") + dist_name(spec.dist) +
                                    "' requires dtype " + dtype_name(want));
    }
}

}  // namespace

RawTensor generate(const SynthSpec& spec) {
    RawTensor t;
    t.shape = spec.shape;
    t.dtype = spec.dtype;
    const size_t n = numel(spec.shape);
    t.data.assign(n * element_size(spec.dtype), 0);

    switch (spec.dist) {
        case Dist::Zeros:
            break;

        case Dist::RandomBits: {
            std::mt19937_64 rng(spec.seed);
            size_t i = 0;
            for (; i + 8 <= t.data.size(); i += 8) {
                const uint64_t word = rng();
                for (size_t b = 0; b < 8; ++b) t.data[i + b] = static_cast<uint8_t>(word >> (8 * b));
            }
            if (i < t.data.size()) {
                const uint64_t word = rng();
                for (size_t b = 0; i < t.data.size(); ++i, ++b) {
                    t.data[i] = static_cast<uint8_t>(word >> (8 * b));
                }
            }
            break;
        }

        case Dist::Gaussian: {
            require_dtype(spec, DataType::FP32);
            NormalSource normal(spec.seed);
            for (size_t i = 0; i < n; ++i) {
                store_f32(t.data, i, static_cast<float>(spec.sigma * normal.next()));
            }
            break;
        }

        case Dist::Ar1Gaussian: {
            require_dtype(spec, DataType::FP32);
            if (!(spec.rho > -1.0 && spec.rho < 1.0)) {
                throw std::invalid_argument("generate: ar1_gaussian needs -1 < rho < 1");
            }
            NormalSource normal(spec.seed);
            // Innovation scale chosen so the stationary std-dev is exactly sigma.
            const double innovation = std::sqrt(1.0 - spec.rho * spec.rho) * spec.sigma;
            double x = spec.sigma * normal.next();
            for (size_t i = 0; i < n; ++i) {
                store_f32(t.data, i, static_cast<float>(x));
                x = spec.rho * x + innovation * normal.next();
            }
            break;
        }

        case Dist::SharedExponent: {
            require_dtype(spec, DataType::FP32);
            std::mt19937_64 rng(spec.seed);
            // Snap the scale to a power of two so [scale, 2*scale) is exactly one binade.
            const double scale = std::exp2(std::floor(std::log2(spec.sigma)));
            for (size_t i = 0; i < n; ++i) {
                // 23 random mantissa bits -> uniform in [1, 2).
                const double unit = 1.0 + static_cast<double>(rng() >> 41) * 0x1.0p-23;
                store_f32(t.data, i, static_cast<float>(scale * unit));
            }
            break;
        }

        case Dist::QuantizedInt8: {
            require_dtype(spec, DataType::INT8);
            NormalSource normal(spec.seed);
            // Symmetric linear quantization with the clip point at 4 sigma.
            const double step = 4.0 / 127.0;
            for (size_t i = 0; i < n; ++i) {
                const double q = std::nearbyint(normal.next() / step);
                const int clipped = static_cast<int>(std::clamp(q, -127.0, 127.0));
                t.data[i] = static_cast<uint8_t>(static_cast<int8_t>(clipped));
            }
            break;
        }
    }
    return t;
}

const char* dist_name(Dist dist) {
    switch (dist) {
        case Dist::Zeros: return "zeros";
        case Dist::RandomBits: return "random_bits";
        case Dist::Gaussian: return "gaussian";
        case Dist::Ar1Gaussian: return "ar1_gaussian";
        case Dist::SharedExponent: return "shared_exponent";
        case Dist::QuantizedInt8: return "quantized_int8";
    }
    return "unknown";
}

Dist parse_dist(std::string_view name) {
    for (Dist d : {Dist::Zeros, Dist::RandomBits, Dist::Gaussian, Dist::Ar1Gaussian,
                   Dist::SharedExponent, Dist::QuantizedInt8}) {
        if (name == dist_name(d)) return d;
    }
    throw std::invalid_argument("unknown dist '" + std::string(name) + "'");
}

}  // namespace tc
