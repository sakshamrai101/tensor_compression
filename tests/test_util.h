#pragma once

#include <cstdint>
#include <random>
#include <vector>

namespace tctest {

// Random bytes where each byte is zero with probability `zero_fraction`, in
// bursts, so the result mixes long zero runs, short zero runs and literals.
inline std::vector<uint8_t> random_bytes(size_t n, uint64_t seed, double zero_fraction = 0.0) {
    std::mt19937_64 rng(seed);
    std::vector<uint8_t> bytes(n);
    std::uniform_real_distribution<double> unit(0.0, 1.0);
    size_t i = 0;
    while (i < n) {
        if (unit(rng) < zero_fraction) {
            // Geometric-ish burst of zeros, 1..~600 bytes.
            size_t run = 1 + static_cast<size_t>(rng() % 8);
            if (rng() % 4 == 0) run += static_cast<size_t>(rng() % 600);
            for (; run > 0 && i < n; --run) bytes[i++] = 0;
        } else {
            bytes[i++] = static_cast<uint8_t>(rng());
        }
    }
    return bytes;
}

}  // namespace tctest
