// Separate binary: replaces global operator new to count heap allocations on
// the decompress_to hot path after warm-up.

#include "compressor.h"
#include "synthetic.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <new>

namespace {
std::atomic<long long> g_alloc_count{0};
bool g_counting = false;
}  // namespace

void* operator new(std::size_t n) {
    if (g_counting) g_alloc_count.fetch_add(1, std::memory_order_relaxed);
    if (void* p = std::malloc(n)) return p;
    throw std::bad_alloc{};
}
void* operator new(std::size_t n, const std::nothrow_t&) noexcept {
    if (g_counting) g_alloc_count.fetch_add(1, std::memory_order_relaxed);
    return std::malloc(n);
}
void* operator new[](std::size_t n) {
    return ::operator new(n);
}
void* operator new[](std::size_t n, const std::nothrow_t&) noexcept {
    return ::operator new(n, std::nothrow);
}
void operator delete(void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t) noexcept { std::free(p); }

int main() {
    using namespace tc;
    SynthSpec spec;
    spec.dist = Dist::Gaussian;
    spec.shape = {128, 128};
    spec.seed = 1;
    RawTensor in = generate(spec);
    TensorCompressor c;
    CompressedTensor mid = c.compress(in);

    RawTensor out;
    c.decompress_to(mid, out);  // warm-up: sizes scratch + out
    if (out.data != in.data) {
        std::fprintf(stderr, "alloc test: warm-up round-trip failed\n");
        return 1;
    }

    g_alloc_count.store(0);
    g_counting = true;
    for (int i = 0; i < 50; ++i) {
        c.decompress_to(mid, out);
    }
    g_counting = false;

    const long long allocs = g_alloc_count.load();
    if (allocs != 0) {
        std::fprintf(stderr, "alloc test: expected 0 allocations after warm-up, got %lld\n", allocs);
        return 1;
    }
    if (out.data != in.data) {
        std::fprintf(stderr, "alloc test: data mismatch after counted decompress\n");
        return 1;
    }
    std::printf("alloc test: ok (0 allocations in 50 warmed decompress_to calls)\n");
    return 0;
}
