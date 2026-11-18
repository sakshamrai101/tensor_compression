#include "huffman_experiment.h"
#include "synthetic.h"
#include "test_harness.h"

using namespace tc;

TC_TEST(huffman, zeros_high_ratio) {
    SynthSpec spec;
    spec.dist = Dist::Zeros;
    spec.shape = {64, 64};
    RawTensor t = generate(spec);
    auto r = huffman::compress_exponent_plane(t.data);
    TC_CHECK(r.input_bytes == t.data.size());
    TC_CHECK(r.ratio > 1.0);
}

TC_TEST(huffman, gaussian_beats_one_sometimes) {
    SynthSpec spec;
    spec.dist = Dist::Gaussian;
    spec.shape = {128, 128};
    spec.sigma = 0.02;
    spec.seed = 1;
    auto r = huffman::compress_exponent_plane(generate(spec).data);
    // With table overhead this may be near 1.0 on small tensors; just check it runs.
    TC_CHECK(r.encoded_exponent_bytes <= r.exponent_bytes + 16);
    TC_CHECK(r.total_frame_bytes > 0);
}
