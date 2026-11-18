#include "baselines.h"

#include "codec/byte_shuffle.h"

#include <cstring>
#include <vector>

#if defined(TC_BENCH_HAVE_ZLIB)
#include <zlib.h>
#endif
#if defined(TC_BENCH_HAVE_LZ4)
#include <lz4.h>
#endif
#if defined(TC_BENCH_HAVE_ZSTD)
#include <zstd.h>
#endif

namespace tc::bench {

BaselineResult MemcpyBaseline::compress(std::span<const uint8_t> in) {
    BaselineResult r;
    r.compressed.assign(in.begin(), in.end());
    r.ok = true;
    return r;
}

bool MemcpyBaseline::decompress(std::span<const uint8_t> compressed, std::span<uint8_t> out) {
    if (compressed.size() != out.size()) return false;
    if (!out.empty()) std::memcpy(out.data(), compressed.data(), out.size());
    return true;
}

#if defined(TC_BENCH_HAVE_ZLIB)
ZlibBaseline::ZlibBaseline(int level) : level_(level), name_("zlib-" + std::to_string(level)) {}

BaselineResult ZlibBaseline::compress(std::span<const uint8_t> in) {
    BaselineResult r;
    uLongf dest_len = compressBound(static_cast<uLong>(in.size()));
    r.compressed.resize(dest_len);
    const int rc = ::compress2(r.compressed.data(), &dest_len, in.data(), static_cast<uLong>(in.size()),
                               level_);
    if (rc != Z_OK) {
        r.error = "zlib compress2 failed";
        r.compressed.clear();
        return r;
    }
    r.compressed.resize(dest_len);
    r.ok = true;
    return r;
}

bool ZlibBaseline::decompress(std::span<const uint8_t> compressed, std::span<uint8_t> out) {
    uLongf dest_len = static_cast<uLongf>(out.size());
    return ::uncompress(out.data(), &dest_len, compressed.data(),
                        static_cast<uLong>(compressed.size())) == Z_OK &&
           dest_len == out.size();
}
#endif

#if defined(TC_BENCH_HAVE_LZ4)
BaselineResult Lz4Baseline::compress(std::span<const uint8_t> in) {
    BaselineResult r;
    const int max_dst = LZ4_compressBound(static_cast<int>(in.size()));
    r.compressed.resize(static_cast<size_t>(max_dst));
    const int n = LZ4_compress_default(reinterpret_cast<const char*>(in.data()),
                                       reinterpret_cast<char*>(r.compressed.data()),
                                       static_cast<int>(in.size()), max_dst);
    if (n <= 0) {
        r.error = "lz4 compress failed";
        r.compressed.clear();
        return r;
    }
    r.compressed.resize(static_cast<size_t>(n));
    r.ok = true;
    return r;
}

bool Lz4Baseline::decompress(std::span<const uint8_t> compressed, std::span<uint8_t> out) {
    const int n = LZ4_decompress_safe(reinterpret_cast<const char*>(compressed.data()),
                                      reinterpret_cast<char*>(out.data()),
                                      static_cast<int>(compressed.size()),
                                      static_cast<int>(out.size()));
    return n == static_cast<int>(out.size());
}

BaselineResult ShuffleLz4Baseline::compress(std::span<const uint8_t> in) {
    BaselineResult r;
    if (in.size() % 4 != 0) {
        r.error = "shuffle+lz4 expects a multiple of 4 bytes";
        return r;
    }
    std::vector<uint8_t> shuffled(in.size());
    codec::byte_shuffle_scalar(in.data(), shuffled.data(), in.size(), 4);
    const int max_dst = LZ4_compressBound(static_cast<int>(in.size()));
    r.compressed.resize(static_cast<size_t>(max_dst));
    const int n = LZ4_compress_default(reinterpret_cast<const char*>(shuffled.data()),
                                       reinterpret_cast<char*>(r.compressed.data()),
                                       static_cast<int>(in.size()), max_dst);
    if (n <= 0) {
        r.error = "shuffle+lz4 compress failed";
        r.compressed.clear();
        return r;
    }
    r.compressed.resize(static_cast<size_t>(n));
    r.ok = true;
    return r;
}

bool ShuffleLz4Baseline::decompress(std::span<const uint8_t> compressed, std::span<uint8_t> out) {
    std::vector<uint8_t> shuffled(out.size());
    const int n = LZ4_decompress_safe(reinterpret_cast<const char*>(compressed.data()),
                                      reinterpret_cast<char*>(shuffled.data()),
                                      static_cast<int>(compressed.size()),
                                      static_cast<int>(out.size()));
    if (n != static_cast<int>(out.size())) return false;
    codec::byte_unshuffle_scalar(shuffled.data(), out.data(), out.size(), 4);
    return true;
}
#endif

#if defined(TC_BENCH_HAVE_ZSTD)
ZstdBaseline::ZstdBaseline(int level) : level_(level), name_("zstd-" + std::to_string(level)) {}

BaselineResult ZstdBaseline::compress(std::span<const uint8_t> in) {
    BaselineResult r;
    const size_t bound = ZSTD_compressBound(in.size());
    r.compressed.resize(bound);
    const size_t n = ZSTD_compress(r.compressed.data(), bound, in.data(), in.size(), level_);
    if (ZSTD_isError(n)) {
        r.error = ZSTD_getErrorName(n);
        r.compressed.clear();
        return r;
    }
    r.compressed.resize(n);
    r.ok = true;
    return r;
}

bool ZstdBaseline::decompress(std::span<const uint8_t> compressed, std::span<uint8_t> out) {
    const size_t n = ZSTD_decompress(out.data(), out.size(), compressed.data(), compressed.size());
    return !ZSTD_isError(n) && n == out.size();
}
#endif

std::vector<Baseline*> all_baselines() {
    static MemcpyBaseline memcpy_b;
#if defined(TC_BENCH_HAVE_ZLIB)
    static ZlibBaseline zlib1(1);
    static ZlibBaseline zlib6(6);
#endif
#if defined(TC_BENCH_HAVE_LZ4)
    static Lz4Baseline lz4_b;
    static ShuffleLz4Baseline shuffle_lz4;
#endif
#if defined(TC_BENCH_HAVE_ZSTD)
    static ZstdBaseline zstd1(1);
    static ZstdBaseline zstd3(3);
#endif
    std::vector<Baseline*> out;
    out.push_back(&memcpy_b);
#if defined(TC_BENCH_HAVE_ZLIB)
    out.push_back(&zlib1);
    out.push_back(&zlib6);
#endif
#if defined(TC_BENCH_HAVE_LZ4)
    out.push_back(&lz4_b);
    out.push_back(&shuffle_lz4);
#endif
#if defined(TC_BENCH_HAVE_ZSTD)
    out.push_back(&zstd1);
    out.push_back(&zstd3);
#endif
    return out;
}

}  // namespace tc::bench
