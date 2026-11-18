#include "layer_pager.h"
#include "synthetic.h"
#include "test_harness.h"

#include <algorithm>

using namespace tc;

TC_TEST(pager, map_layers_roundtrip) {
    LayerPager pager;
    SynthSpec a;
    a.dist = Dist::Gaussian;
    a.shape = {64, 64};
    a.seed = 1;
    SynthSpec b;
    b.dist = Dist::Ar1Gaussian;
    b.shape = {32, 128};
    b.seed = 2;
    RawTensor la = generate(a);
    RawTensor lb = generate(b);
    const size_t i0 = pager.add_layer(la);
    const size_t i1 = pager.add_layer(lb);
    TC_CHECK(i0 == 0 && i1 == 1);
    TC_CHECK(pager.layer_count() == 2);
    TC_CHECK(pager.compressed_bytes() < pager.raw_bytes() || pager.compressed_bytes() > 0);

    auto v0 = pager.map_layer(0);
    TC_CHECK(v0.size() == la.data.size());
    TC_CHECK(std::equal(v0.begin(), v0.end(), la.data.begin()));

    auto v1 = pager.map_layer(1);
    TC_CHECK(v1.size() == lb.data.size());
    TC_CHECK(std::equal(v1.begin(), v1.end(), lb.data.begin()));

    // Remap first layer; previous view is invalidated by design but data is correct.
    auto v0b = pager.map_layer(0);
    TC_CHECK(std::equal(v0b.begin(), v0b.end(), la.data.begin()));
}

TC_TEST(pager, load_library) {
    CompressOptions opts;
    TensorCompressor c(opts);
    SynthSpec spec;
    spec.dist = Dist::Zeros;
    spec.shape = {100, 100};
    std::vector<CompressedTensor> lib;
    lib.push_back(c.compress(generate(spec)));
    LayerPager pager;
    pager.load_library(std::move(lib));
    TC_CHECK(pager.layer_count() == 1);
    auto view = pager.map_layer(0);
    TC_CHECK(view.size() == 100ull * 100ull * 4ull);
    for (uint8_t b : view) TC_CHECK(b == 0);
}
