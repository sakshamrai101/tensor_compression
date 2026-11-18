#include "analyze.h"
#include "synthetic.h"
#include "test_harness.h"

using namespace tc;

TC_TEST(analyze, zeros_near_zero_entropy) {
    SynthSpec spec;
    spec.dist = Dist::Zeros;
    spec.shape = {128, 128};
    AnalysisReport r = analyze(generate(spec));
    TC_CHECK(r.elem_size == 4);
    TC_CHECK(r.n_elems == 128ull * 128ull);
    for (const PlaneStats& p : r.planes) {
        TC_CHECK(p.entropy_bits < 0.01);
    }
    TC_CHECK(r.order0_ratio > 10.0);  // ideal coder would crush zeros
}

TC_TEST(analyze, random_bits_near_eight) {
    SynthSpec spec;
    spec.dist = Dist::RandomBits;
    spec.shape = {64, 256};
    spec.seed = 9;
    AnalysisReport r = analyze(generate(spec));
    for (const PlaneStats& p : r.planes) {
        TC_CHECK(p.entropy_bits > 7.5);
    }
    TC_CHECK(r.order0_ratio < 1.05);
}

TC_TEST(analyze, gaussian_has_exponent_redundancy) {
    SynthSpec spec;
    spec.dist = Dist::Gaussian;
    spec.shape = {256, 256};
    spec.sigma = 0.02;
    spec.seed = 1;
    AnalysisReport r = analyze(generate(spec));
    TC_CHECK(r.exponent_field_bits == 8);
    // Exponent field for small-sigma Gaussians is far from full entropy.
    TC_CHECK(r.exponent_entropy_bits > 0.0);
    TC_CHECK(r.exponent_entropy_bits < 7.0);
    // Low mantissa plane should be closer to random than the high byte.
    TC_CHECK(r.planes[0].entropy_bits > r.planes[3].entropy_bits);
}
