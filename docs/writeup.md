# TC-Runtime: Lossless Tensor Compression for the Memory Wall

Portfolio writeup for MS CS applications. Companion to the code in this repository.

## 1. Motivation

Modern LLM and edge inference at batch size 1 is often limited by **memory bandwidth**, not peak FLOPS. Each token (or each layer evaluation) streams weight tensors from DRAM through a finite bus into the cache hierarchy. When the bus saturates, cores stall — the classic **memory wall**.

The industry default mitigation is **lossy quantization** (FP16 → INT8/INT4, GPTQ/AWQ, etc.): fewer bytes on the wire, but **changed values** and a quality trade-off. TC-Runtime asks a complementary question:

> How much of an FP32 weight byte-stream is losslessly removable by a cheap, cache-local, block-wise transform, and at what decode throughput does that removal pay for itself on a real memory system?

The goal is not “beat the memory wall forever.” It is a **measured** answer: ratio, decode GB/s, entropy headroom, and a break-even model against measured DRAM bandwidth on the evaluation host.

## 2. Background and related work

**Memory hierarchy.** Effective bandwidth is limited by DRAM, then by last-level cache capacity. A codec that shrinks the resident set and decodes inside L2 can matter even when peak DRAM GB/s looks high on a laptop SoC.

**Filters vs entropy coders.** HDF5/Blosc’s **shuffle** filter and **Bitshuffle** rearrange bytes/bits so subsequent compressors see longer runs. **lz4** and **zstd** are general-purpose compressors; DEFLATE/zlib is slower for the same idea. Facebook’s **Gorilla** XOR-deltas floating-point time series; **fpzip** targets scientific floats. Recent NN-weight work (**ZipNN**, **DFloat11**) entropy-codes the low-entropy **exponent** field of IEEE floats — a stronger compressor than RLE on mantissa-heavy data.

TC-Runtime does **not** claim novelty over these. It is an engineering study: a small C++20 codec, a versioned frame format, a scalar oracle + NEON fast path, honest baselines (including shuffle+lz4), and an admissions-grade evaluation methodology.

**Quantization stance.** Complementary, not competitive. An INT8 tensor that has already been quantized can still be passed through the same lossless pipeline (see `quantized_int8` synthetic and tools).

## 3. Method

### 3.1 Pipeline

Per block (default 64 KiB):

1. **Byte-shuffle** (element size 4/2/1): AoS → SoA byte planes. Clusters sign/exponent-ish bytes for FP32.
2. **XOR-delta**: `out[i] = in[i] ^ in[i-1]`. Equal-byte runs become zeros.
3. **Zero-run + literal RLE**: `0x00,k` → k+1 zeros; `0x01,k,bytes` → k+1 literals.

If encoded size ≥ raw size, the block is **stored raw** (`mode=1`). Stages are selected by TCF1 flags so ablations share one format.

### 3.2 Format (TCF1)

64-byte little-endian header (`magic, version, flags, dtype, ndims, …, shape[6]`) plus block records `(raw_len, comp_len, mode, reserved[3], bytes)`. Explicit field serialization — no host-endian `memcpy` of the struct. See `architecture.md` §4 and `include/format.h`.

### 3.3 Hot path

`TensorCompressor::decompress_to` writes into a caller-sized `RawTensor` using member scratch buffers. After warm-up, an allocation-counting test requires **zero** heap allocations on that path.

## 4. Implementation

- **Language / build:** C++20, CMake ≥ 3.20. Targets: `tc_runtime`, `tc`, `tc_tests`, `tc_bench`, `tc_alloc_test`.
- **Modules:** `format`, `codec/{byte_shuffle,xor_delta,rle,block}`, `compressor`, `synthetic`, `analyze`, `runtime/layer_pager`, `simd/byte_shuffle_{neon,avx2}`.
- **SIMD:** scalar is the oracle. Apple Silicon builds NEON (`vld4q_u8` / `vst4q_u8` for 4-byte unshuffle). AVX2 is compiled only on x86. Equality tests compare SIMD to scalar.
- **Tests:** in-repo harness (no GoogleTest). Round-trip matrix over flags × dtypes × sizes; malformed frames; sanitizer (UBSan on Apple; ASan+UBSan elsewhere — ASan currently deadlocks in dyld init on this macOS).
- **CLI:** `tc gen|compress|decompress|info|stats|analyze`.

## 5. Experiments

### 5.1 Hardware / software

Recorded automatically by `tc_bench` into CSV headers: chip, OS, compiler, build type, git SHA, SIMD kernel, single-thread and all-thread memcpy bandwidth over 256 MiB.

### 5.2 Datasets

| Name | Role |
|------|------|
| `zeros` | Sanity upper bound |
| `random_bits` | Incompressible lower bound / stored-fallback check |
| `gaussian(σ=0.02)` | Honest “weight-like” iid FP32 |
| `ar1_gaussian` | Illustrative (favours delta by construction) |
| `shared_exponent` | Narrow dynamic range |
| `quantized_int8` | Post-quant complementary case |
| GPT-2 small FP32 dumps | Real weights via `tools/export_safetensors.py` (gitignored under `data/`) |

### 5.3 Protocol

Release `-O3`, warm-up 3, 21 timed iterations (7 in `--quick`), report median encode/decode GB/s (uncompressed bytes / wall time), min and p90. Hot (≤64 KiB) and cold (≥ several MiB) regimes. Baselines: memcpy, zlib-1/6, lz4, zstd-1/3, **shuffle+lz4**.

### 5.4 Break-even model

Compression helps a bandwidth-bound load (serial, rough) when:

\[
\frac{C}{B} + t_{\mathrm{decode}} < \frac{N}{B}
\quad\Leftrightarrow\quad
T_{\mathrm{dec}} > B \cdot \frac{R}{R-1}
\]

with \(R = N/C\). `tc_bench` prints required vs measured decode GB/s using measured \(B_1\).

## 6. Results

**Source of truth:** `results/*.csv` produced by `./build/tc_bench --csv …`. Do not copy numbers by hand into this section without a matching CSV row.

Qualitative findings observed during development (confirm against your latest CSV):

- **Zeros** compress extremely well (RLE of zero planes).
- **iid Gaussian FP32** often sits near **1.0×** with the stored fallback — mantissa planes are near-random; this is a finding, not a bug.
- **AR(1)** exceeds 1× and shows the pipeline can win when correlation exists — disclose that the distribution favours XOR-delta.
- **Random bits** stay ≥ ~0.98× thanks to the stored fallback (classic `[value][run_len]` would have expanded them ~2×).
- **NEON** matches scalar byte-for-byte; speedup appears mainly on shuffle/unshuffle.
- **Break-even** on Apple Silicon’s high \(B_1\) is harsh: modest \(R\) demands very high decode GB/s. The honest story includes when the codec does **not** win.

Fill the tables below after running the harness:

### Main results

From `results/arm64_63a8ced_full.csv` (Apple Silicon, AppleClang 17, Release, NEON, \(B_1 \approx 66\) GB/s). Hot = 64 KiB; cold = 32 MiB.

| dataset | codec | regime | ratio | enc GB/s | dec GB/s |
|---------|-------|--------|------:|---------:|---------:|
| zeros | tc-full | hot | 111.5 | 13.3 | 4.1 |
| zeros | tc-full | cold | 125.0 | 14.9 | 4.0 |
| gaussian | tc-full | cold | 1.000 | 1.9 | 72.1† |
| gaussian | shuffle+lz4 | cold | 1.083 | 1.4 | 3.3 |
| gaussian | zstd-1 | cold | 1.080 | 1.5 | 1.7 |
| ar1 ρ=0.99 | tc-full | hot | 1.130 | 4.0 | 3.0 |
| ar1 | tc-full | cold | 1.135 | 1.7 | 2.2 |
| ar1 | shuffle+lz4 | cold | 1.196 | 1.9 | 3.0 |
| shared_exponent | tc-full | hot | 1.317 | 10.0 | 3.7 |

† Stored-raw fallback: decode is memcpy of uncompressed blocks.

Cold confirms hot: ratios hold; encode pays a cold-cache tax; encoded-path decode stays ~2–4 GB/s.

### Ablations (ratio on ar1_hot)

| tc-full | tc-shuffle | tc-sx | tc-xr | tc-rle |
|--------:|-----------:|------:|------:|-------:|
| 1.130 | 0.999 | 0.999 | 0.999 | 0.999 |

Full pipeline needed for the 1.13× win; shuffle alone does not shrink (stored fallback).

### Break-even

| dataset | R | required GB/s | measured GB/s | verdict |
|---------|--:|--------------:|--------------:|---------|
| zeros_hot tc-full | 111.5 | 66.5 | 4.1 | LOSS |
| zeros_hot tc-rle | 111.5 | 66.5 | 143 | WIN (RLE-only decode) |
| ar1_hot tc-full | 1.13 | 574 | 3.0 | LOSS |
| ar1_cold tc-full | 1.14 | 554 | 2.2 | LOSS |
| shared_hot tc-full | 1.32 | 273 | 3.7 | LOSS |

Fused stream microbench (`tc_fused_stream` 4 MiB AR1, 4 threads): raw touch ~90 GB/s; serial decompress+touch ~1.7 GB/s; parallel aggregate ~7 GB/s — still below \(B_1\) on this host.

### GPT-2 small (native FP32)

Exported with `tools/export_safetensors.py --model gpt2 --max-tensors 40` (layers 0–3 + embeddings; no BF16 upcast). Full table: `results/gpt2_ratios.txt`.

| class | n | min R | median R | max R |
|-------|--:|------:|---------:|------:|
| matrices (≥100 KB) | 14 | 0.9998 | 0.9998 | 1.0001 |
| vectors (<100 KB) | 26 | 0.9759 | 0.9910 | 1.2637 |

Every large weight matrix sits on the stored-fallback floor (~1.0×). Small LayerNorm weight vectors sometimes exceed 1.2× (structured scales), which does not move the model-size needle. Entropy on a bias vector (`tc analyze`): byte plane 3 ≈ 3.3 bits/byte vs ≈ 7.9 for mantissa planes; order-0 bound ≈ 1.19× while plane XOR+RLE ≈ 1.00× — RLE misses the remaining exponent redundancy (Huffman stretch ~1.17× on gaussian synthetics).

## 7. What didn’t work

1. **Classic `[value][run_len]` RLE** doubles incompressible bytes. Replaced by zero-run + literal tokens before any README claim.
2. **iid Gaussian / real FP32 mantissa entropy** leaves little for RLE after shuffle+XOR. Entropy analysis (`tc analyze`) shows the exponent byte holds most order-0 redundancy; ZipNN/DFloat11-style coding is the known next step (stretch Huffman experiment in `benchmarks/huffman_experiment.cpp`).
3. **Serial XOR-delta decode** limits SIMD gains on the undo path. Format v2 restart blocks are future work.
4. **ASan on this macOS** deadlocks in `AsanInitInternal` during dyld init. CI / local sanitize uses UBSan on Apple; ASan remains enabled on Linux.
5. **BF16-upcast trap:** casting BF16 checkpoints to FP32 creates 16 zero low bits and inflates ratio. Export tools should prefer natively FP32 checkpoints (GPT-2 small) or report upcast separately.

## 8. Next steps

- Restartable XOR sub-blocks for parallel decode.
- Stronger exponent entropy coding in-format (v2), still lossless w.r.t. FP32 bits.
- Optional plugin shim for a real runtime (out of portfolio scope unless explicitly requested).
- Broader real-weight corpus (not only GPT-2).

## References (engineering)

- Bitshuffle / Blosc shuffle filters; lz4; zstd; zlib/DEFLATE.
- Pelkonen et al., Gorilla (VLDB 2015) — XOR of floating-point values.
- fpzip — lossless floating-point compression.
- ZipNN (2024); DFloat11 (2025) — NN weight / exponent-oriented lossless compression.
