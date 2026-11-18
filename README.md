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

Primary source: [`results/arm64_63a8ced_full.csv`](results/arm64_63a8ced_full.csv) (`./build/tc_bench --csv …`). Quick smoke: [`results/arm64_874c29d_quick.csv`](results/arm64_874c29d_quick.csv).

Host: Apple Silicon, AppleClang 17, Release `-O3`, SIMD=`neon`, single-thread memcpy \(B_1\) ≈ **66 GB/s**.

### Hot (64 KiB) vs cold (32 MiB)

| dataset | codec | regime | ratio | enc GB/s | dec GB/s |
| :--- | :--- | :--- | ---: | ---: | ---: |
| zeros | tc-full | hot | 111.5 | 13.3 | 4.1 |
| zeros | tc-full | cold | 125.0 | 14.9 | 4.0 |
| zeros | shuffle+lz4 | cold | 255.0 | 3.8 | 3.8 |
| gaussian σ=0.02 | tc-full | hot | 0.999 | 4.1 | 68.4† |
| gaussian | tc-full | cold | 1.000 | 1.9 | 72.1† |
| gaussian | shuffle+lz4 | cold | 1.083 | 1.4 | 3.3 |
| gaussian | zstd-1 | cold | 1.080 | 1.5 | 1.7 |
| ar1 ρ=0.99 | tc-full | hot | 1.130 | 4.0 | 3.0 |
| ar1 | tc-full | cold | 1.135 | 1.7 | 2.2 |
| ar1 | shuffle+lz4 | cold | 1.196 | 1.9 | 3.0 |
| shared_exponent | tc-full | hot | 1.317 | 10.0 | 3.7 |
| random_bits | tc-full | cold | 1.000 | 5.6 | 73.3† |

† Decode ≈ memcpy: **stored-raw fallback** kept blocks uncompressed (ratio ≈ 1). Not an RLE-path speed claim.

**Cold vs hot:** ratios are stable; encode slows on cold working sets; decode on the encoded path stays ~2–4 GB/s. `shuffle+lz4` still wins ratio on ar1/gaussian; TC wins only when the stream is highly zero/run structured.

**Break-even (serial \(T > B_1 R/(R-1)\)):** `tc-full` is LOSS on ar1/shared/zeros-encoded on this host (required ≫ measured decode). Documented in [`docs/writeup.md`](docs/writeup.md).

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
