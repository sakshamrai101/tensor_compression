#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

// Compress/decompress adapters used by tc_bench. A missing library is reported
// as available() == false rather than silently dropped from the table.

namespace tc::bench {

struct BaselineResult {
    std::vector<uint8_t> compressed;
    bool ok = false;
    std::string error;

    size_t compressed_bytes() const { return compressed.size(); }
};

struct Baseline {
    virtual ~Baseline() = default;
    virtual const char* name() const = 0;
    virtual bool available() const = 0;
    virtual BaselineResult compress(std::span<const uint8_t> in) = 0;
    virtual bool decompress(std::span<const uint8_t> compressed, std::span<uint8_t> out) = 0;
};

std::vector<Baseline*> all_baselines();

class MemcpyBaseline : public Baseline {
public:
    const char* name() const override { return "memcpy"; }
    bool available() const override { return true; }
    BaselineResult compress(std::span<const uint8_t> in) override;
    bool decompress(std::span<const uint8_t> compressed, std::span<uint8_t> out) override;
};

#if defined(TC_BENCH_HAVE_ZLIB)
class ZlibBaseline : public Baseline {
public:
    explicit ZlibBaseline(int level);
    const char* name() const override { return name_.c_str(); }
    bool available() const override { return true; }
    BaselineResult compress(std::span<const uint8_t> in) override;
    bool decompress(std::span<const uint8_t> compressed, std::span<uint8_t> out) override;

private:
    int level_;
    std::string name_;
};
#endif

#if defined(TC_BENCH_HAVE_LZ4)
class Lz4Baseline : public Baseline {
public:
    const char* name() const override { return "lz4"; }
    bool available() const override { return true; }
    BaselineResult compress(std::span<const uint8_t> in) override;
    bool decompress(std::span<const uint8_t> compressed, std::span<uint8_t> out) override;
};

class ShuffleLz4Baseline : public Baseline {
public:
    const char* name() const override { return "shuffle+lz4"; }
    bool available() const override { return true; }
    BaselineResult compress(std::span<const uint8_t> in) override;
    bool decompress(std::span<const uint8_t> compressed, std::span<uint8_t> out) override;
};
#endif

#if defined(TC_BENCH_HAVE_ZSTD)
class ZstdBaseline : public Baseline {
public:
    explicit ZstdBaseline(int level);
    const char* name() const override { return name_.c_str(); }
    bool available() const override { return true; }
    BaselineResult compress(std::span<const uint8_t> in) override;
    bool decompress(std::span<const uint8_t> compressed, std::span<uint8_t> out) override;

private:
    int level_;
    std::string name_;
};
#endif

}  // namespace tc::bench
