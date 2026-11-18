# TensorCompression (TC-Runtime)

Lossless compression for tensor weight blobs, aimed at the **memory wall** in edge / local AI inference.

Inference at small batch sizes is often **memory-bandwidth bound**, not TFLOPS-bound. Weights stream from DRAM into cache; when the bus saturates, cores stall. Quantization reduces bytes by changing values. TC-Runtime asks a complementary question: how much of an FP32 weight stream can be removed **losslessly**, with a decode fast enough that the savings can matter.

**Status:** implemented through Phase 3 (codec, NEON path, benches, writeup, tools). Results below come from `tc_bench` on this machine — never invented.

---

## Pipeline

Per 64 KiB block, stages toggle via TCF1 flags:

1. **Byte-shuffle** — transpose AoS element bytes into SoA byte planes (Blosc/HDF5-style shuffle filter).
2. **XOR-delta** — `out[i] = in[i] ^ in[i-1]` so runs of equal bytes become zeros (Gorilla-style idea on bytes).
3. **Zero-run + literal RLE** — pack zero runs; incompressible bytes ride in literal tokens.

If a block does not shrink, it is **stored raw** (worst-case ratio ≈ 0.99×, never ~0.5×).

Decode is the exact reverse. Hot path: `decompress_to` into a caller-sized buffer with member scratch (zero heap alloc after warm-up).

Related engineering (not novelty claims): Bitshuffle, Blosc, lz4/zstd, Gorilla, fpzip, ZipNN, DFloat11.

---

## Build

```bash
# CMake ≥ 3.20, C++20. On this repo a venv cmake is fine:
#   python3 -m venv .venv && .venv/bin/pip install cmake
export PATH="$PWD/.venv/bin:$PATH"   # if using the venv

cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Optional: `-DTC_SANITIZE=ON` (Debug), `-DTC_WITH_LZ4=OFF`, `-DTC_WITH_ZSTD=OFF`.

**ISA:** scalar always works. On Apple Silicon, NEON shuffle/unshuffle is compiled in (`TC_HAVE_NEON`). AVX2 is compiled only on x86 hosts and is never required.

---

## CLI

```bash
./build/tc gen --dist gaussian --shape 4096,4096 --seed 1 -o x.bin
./build/tc compress -i x.bin -o x.tcf --dtype fp32 --shape 4096,4096
./build/tc decompress -i x.tcf -o y.bin
./build/tc info -i x.tcf
./build/tc stats -i x.tcf
./build/tc analyze -i x.bin --dtype fp32
```

Distributions: `zeros`, `random_bits`, `gaussian`, `ar1_gaussian`, `shared_exponent`, `quantized_int8`.

---

## Results (measured)

Source: [`results/arm64_874c29d_quick.csv`](results/arm64_874c29d_quick.csv) from

```bash
./build/tc_bench --quick --csv results/arm64_<sha>_quick.csv
```

Host (that run): Apple Silicon, Darwin 25.6, AppleClang 17, Release `-O3`, SIMD=`neon`, single-thread memcpy ≈ **56–63 GB/s** (varies by thermal/load).

| dataset (64 KiB hot) | codec | ratio | enc GB/s | dec GB/s |
| :--- | :--- | ---: | ---: | ---: |
| zeros | tc-full | 111.5 | 13.7 | 3.7 |
| zeros | lz4 | 245.5 | 26.2 | 20.4 |
| zeros | shuffle+lz4 | 245.5 | 3.4 | 3.2 |
| gaussian σ=0.02 | tc-full | 0.999 | 4.1 | 71.5† |
| gaussian | shuffle+lz4 | 1.063 | 1.7 | 3.0 |
| gaussian | zstd-1 | 1.080 | 1.5 | 1.7 |
| ar1 ρ=0.99 | tc-full | 1.130 | 4.0 | 2.3 |
| ar1 | shuffle+lz4 | 1.175 | 2.2 | 3.4 |
| shared_exponent | tc-full | 1.317 | 11.1 | 3.8 |
| random_bits | tc-full | 0.999 | 8.6 | 71.6† |

† Decode ≈ memcpy because the **stored-raw fallback** kept every block uncompressed (ratio ≈ 1). That is correct behavior, not a speed claim for the RLE path.

**Break-even (serial model \(T > B_1 R/(R-1)\)):** on this host, `tc-full` loses for ar1/shared (required hundreds of GB/s; measured ~2–4 GB/s on the encoded path). High DRAM bandwidth makes modest lossless ratios hard to amortize — documented in [`docs/writeup.md`](docs/writeup.md). Re-run `tc_bench` after code changes; replace this table from the new CSV.

---

## Layout

| Path | Role |
|------|------|
| `include/`, `src/` | Library: format, codec stages, compressor, analyze, layer pager |
| `src/simd/` | NEON / AVX2 shuffle kernels |
| `tests/` | Bit-exact + malformed + alloc-count suite |
| `benchmarks/` | `tc_bench` + zlib/lz4/zstd/shuffle+lz4 baselines |
| `docs/writeup.md` | Portfolio writeup |
| `tools/` | Weight export helpers |

---

## Limitations / next steps

- Not a llama.cpp / ONNX plugin; optional `LayerPager` demo only.
- RLE captures runs, not general entropy; exponent Huffman is a stretch experiment only.
- XOR-delta decode is serial (prefix dependency); format v2 could add restart blocks.
- Primary eval dtype is FP32; FP16/INT8 paths exist for round-trip and INT8 “post-quant” demos.

See [`architecture.md`](architecture.md), [`plan.md`](plan.md), [`docs/writeup.md`](docs/writeup.md).
