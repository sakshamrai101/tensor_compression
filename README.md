# TC-Runtime (TensorCompression)

**Lossless** compression for tensor weight blobs, aimed at the **memory wall** in edge / local AI inference.

Small-batch inference is often **memory-bandwidth bound**, not TFLOPS-bound. Weights stream from DRAM into cache; when the bus saturates, cores stall. Quantization cuts bytes by changing values. This project asks a complementary, measurable question: how much of an FP32 weight stream can be removed **bit-exactly**, with decode fast enough that the savings can matter on a real machine?

**v1.0** — C++20 codec + NEON path + bit-exact tests + honest benches vs zlib/lz4/zstd/shuffle+lz4. All numbers below come from `tc_bench` CSVs in [`results/`](results/). Complementary to quantization; not a replacement. Related engineering (not novelty claims): Bitshuffle, Blosc, lz4/zstd, Gorilla, fpzip, ZipNN, DFloat11.

Portfolio writeup: [`docs/writeup.md`](docs/writeup.md)

---

## Pipeline

Per 64 KiB block (stages toggled by TCF1 flags):

1. **Byte-shuffle** — AoS → SoA byte planes (Blosc/HDF5-style shuffle filter)
2. **XOR-delta** — `out[i] = in[i] ^ in[i-1]` (Gorilla-style idea on bytes)
3. **Zero-run + literal RLE** — zero runs packed; incompressible bytes in literal tokens

If a block does not shrink, it is **stored raw** (worst-case ratio ≈ 0.99×). Decode is the exact reverse. Hot path: `decompress_to` into a caller-sized buffer with member scratch (**zero heap alloc** after warm-up; enforced by `tc_alloc_test`).

---

## Build

```bash
# CMake ≥ 3.20, C++20. Optional: python3 -m venv .venv && .venv/bin/pip install cmake
export PATH="$PWD/.venv/bin:$PATH"

cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
./build/tc_bench --quick --csv results/local.csv
```

- **ISA:** scalar always works. Apple Silicon builds NEON (`TC_HAVE_NEON`). AVX2 is optional on x86 only.
- **Sanitize:** `-DTC_SANITIZE=ON` → UBSan on macOS (ASan+UBSan on Linux; ASan can deadlock in dyld init on some AppleClang/macOS combos).

---

## CLI

```bash
./build/tc gen --dist gaussian --shape 4096,4096 --seed 1 -o x.bin
./build/tc compress -i x.bin -o x.tcf --dtype fp32 --shape 4096,4096
./build/tc decompress -i x.tcf -o y.bin   # bit-identical to x.bin
./build/tc stats -i x.tcf
./build/tc analyze -i x.bin --dtype fp32
```

---

## Results (measured)

Source of truth: [`results/arm64_63a8ced_full.csv`](results/arm64_63a8ced_full.csv)  
Host: Apple Silicon · AppleClang 17 · Release `-O3` · SIMD=`neon` · single-thread memcpy \(B_1\) ≈ **66 GB/s**

| dataset | codec | regime | ratio | enc GB/s | dec GB/s |
| :--- | :--- | :--- | ---: | ---: | ---: |
| zeros | tc-full | hot (64 KiB) | 111.5 | 13.3 | 4.1 |
| zeros | tc-full | cold (32 MiB) | 125.0 | 14.9 | 4.0 |
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

† Decode ≈ memcpy because the **stored-raw fallback** left blocks uncompressed (ratio ≈ 1). Not an RLE-path speed claim.

**GPT-2 small (native FP32, 40 tensors):** every large matrix ≥100 KB is ~**0.9998–1.0001×**; only tiny LayerNorm vectors sometimes exceed ~1.2×. See [`results/gpt2_ratios.txt`](results/gpt2_ratios.txt).

**Break-even:** for serial load, decode must exceed \(B_1 \cdot R/(R-1)\). On this host `tc-full` is a **LOSS** for ar1/shared (required hundreds of GB/s; measured ~2–4 GB/s on the encoded path). High DRAM bandwidth makes modest lossless ratios hard to amortize — that finding is part of the project. Full narrative: [`docs/writeup.md`](docs/writeup.md).

---

## What this is / is not

| Is | Is not |
|----|--------|
| Lossless tensor block codec + measurement study | llama.cpp / ONNX / CUDA plugin |
| Complementary to quantization | A replacement for INT8/INT4 |
| Scalar oracle + NEON + honest baselines | A claim of beating Blosc/ZipNN on ratio |

**Limitations (by design):** RLE captures runs, not general entropy (exponent field still has headroom — see `tc analyze`). XOR-delta decode is serial. Primary eval dtype is FP32.

---

## Repo map

| Path | Role |
|------|------|
| `include/`, `src/codec/`, `src/simd/` | Format, stages, NEON/AVX2 |
| `tests/` | Round-trip, malformed frames, alloc-count |
| `benchmarks/` | `tc_bench`, baselines, fused-stream microbench |
| `docs/writeup.md` | Motivation → method → results → failures |
| `tools/` | GPT-2 / safetensors export helpers |
| `results/` | Measured CSVs (only published numbers) |

```bash
ctest --test-dir build --output-on-failure
./build/tc_bench --csv results/run.csv
```
