# TC-Runtime — Technical Architecture

Companion to [`plan.md`](plan.md) and [`cursorrules.md`](cursorrules.md).  
This document is the source of truth for **format, algorithms, modules, and data flow**.

---

## 1. System overview

TC-Runtime is a **lossless tensor block codec** plus a thin optional **layer paging** demo. It is not a full inference engine.

```mermaid
flowchart TB
  subgraph api [Public API]
    RT[RawTensor]
    CT[CompressedTensor]
    TC[TensorCompressor]
  end

  subgraph pipeline [Codec pipeline encode]
    BS[BitShuffle]
    XD[XorDelta]
    RLE[RLE Pack]
  end

  subgraph decode [Codec pipeline decode]
    URLE[RLE Unpack]
    UXD[XorDelta Undo]
    UBS[BitUnshuffle]
  end

  RT --> TC
  TC --> BS --> XD --> RLE --> CT
  CT --> URLE --> UXD --> UBS --> RT
```

**Design intent:** Weights reside in DRAM as `CompressedTensor` payloads. Before a layer runs, decompress into a reusable scratch `RawTensor` (or raw buffer) using SIMD-friendly kernels, then consume with existing compute code (outside this repo for now).

---

## 2. Problem model (technical)

### 2.1 Quantities

- Let weight tensor \(W\) have byte size \(N = \mathrm{numel}(W) \cdot \mathrm{sizeof(dtype)}\).
- Uncompressed traffic to load \(W\) once ≈ \(N\) bytes from DRAM.
- Compressed size \(C = |payload| + |header|\).
- **Compression ratio** \(R = N / C\) (report as e.g. `1.78x`; store as `compression_ratio_x100 = round(R * 100)`).
- **Decode throughput** \(T_{dec} = N / t_{decode}\) (bytes of uncompressed output per second).

### 2.2 When this helps

Compression helps a bandwidth-bound load if:

\[
\frac{C}{B_{dram}} + t_{decode} < \frac{N}{B_{dram}}
\]

(rough model; ignores cache effects). So decode must be fast relative to the bytes saved. This is why **decode GB/s** matters as much as ratio.

### 2.3 Relationship to quantization

| Approach | Bytes | Values |
|----------|-------|--------|
| Quantization | Fewer | Changed (lossy) |
| TC-Runtime | Fewer on the bus / in resident set | Unchanged (lossless) |
| Both (future) | Pack already-quantized INT8/INT4 further | Still lossless w.r.t. quantized values |

Portfolio framing: **complementary**, not competitive, with quantization.

---

## 3. Public data structures

Evolve existing headers in `include/tensor.h` and `include/compressor.h`.

### 3.1 `DataType`

```cpp
enum class DataType : uint8_t {
    FP32 = 0,
    FP16 = 1,
    INT8 = 2,
};
```

Element sizes: FP32=4, FP16=2, INT8=1.

### 3.2 `RawTensor`

- `alignas(64)` structure (cache-line friendly metadata; payload may be heap-allocated via `std::vector`)
- `std::vector<uint8_t> data` — raw bytes, row-major, tightly packed
- `std::vector<size_t> shape` — dimensions
- `DataType dtype`
- `size_bytes()` → `data.size()`
- Invariant: `data.size() == product(shape) * element_size(dtype)` (enforce in helpers)

### 3.3 `CompressedTensor`

- `alignas(64)`
- `compressed_data` — full frame: **header + payload** (see format)
- `shape`, `dtype` — denormalized for convenience (must match header)
- `original_size_bytes`
- `block_count` — number of codec blocks
- `compression_ratio_x100`
- **Must end with semicolon** (current stub is missing `;`)

### 3.4 `TensorCompressor`

```cpp
CompressedTensor compress(const RawTensor& tensor);
RawTensor decompress(const CompressedTensor& compressed);

// Preferred internal / advanced API (add):
void compress_to(const RawTensor& in, CompressedTensor& out);
void decompress_to(const CompressedTensor& in, RawTensor& out); // pre-reserve out.data
```

Hot path: prefer `decompress_to` with pre-sized `out.data` to avoid allocation churn.

Rename/clarify private methods:

- `bit_shuffle_*` / `bit_unshuffle_*` — keep scalar + simd variants
- Existing names `bit_shuffle_sse` may remain as thin wrappers or be renamed to `bit_shuffle_simd` with comments; do not leave misleading SSE-only naming if implementing AVX2

---

## 4. On-wire / in-memory frame format (`TCF1`)

Define in `include/format.h`. Little-endian. Versioned.

### 4.1 Header (fixed, 64 bytes)

Locked layout (little-endian, field-by-field serialization — never `memcpy` the struct):

| Offset | Size | Field | Notes |
|--------|------|-------|-------|
| 0 | 4 | magic | `'T' 'C' 'F' '1'` (`0x31464354` as u32 LE) |
| 4 | 2 | version | `1` |
| 6 | 2 | flags | bit0 shuffle; bit1 xor; bit2 rle |
| 8 | 1 | dtype | `DataType` |
| 9 | 1 | ndims | 0–6 |
| 10 | 2 | reserved0 | must be 0 |
| 12 | 4 | block_size | default 65536 |
| 16 | 4 | block_count | |
| 20 | 4 | reserved1 | must be 0 |
| 24 | 8 | uncompressed_size | |
| 32 | 8 | payload_size | bytes after the header |
| 40 | 24 | shape | `uint32_t[6]`; first `ndims` used, rest 0 |

`static_assert(sizeof(Tcf1Header) == 64)` plus `offsetof` checks live in `include/format.h`. CRC is not used in v1.

### 4.2 Payload

Sequence of **block records**:

```
for each block:
    u32 raw_len
    u32 comp_len
    u8  mode          # 0 = encoded (stages in flags), 1 = stored raw
    u8  reserved[3]   # must be 0
    u8  bytes[comp_len]
```

Last block may be shorter than `block_size`. Each block independently runs shuffle → xor → rle (when flags say so). If the encoded form is not smaller than `raw_len`, the compressor stores the block raw (`mode = 1`) so a frame never expands by more than header + record overhead.

### 4.3 Flags and ablations

Flags select which stages are active so ablations share one format:

- `FLAG_SHUFFLE = 1 << 0`
- `FLAG_XOR     = 1 << 1`
- `FLAG_RLE     = 1 << 2`

Full pipeline: all three set. Bench can emit shuffle-only etc.

---

## 5. Algorithm specifications

### 5.1 Bit-shuffle (FP32)

Treat the block as `n` consecutive FP32 elements (`n = block_bytes / 4`).  
Each element is 4 bytes `b0,b1,b2,b3` (IEEE754 little-endian memory order).

**Byte-shuffle (v1 default — implement first):**  
Transpose from AoS bytes to SoA byte planes:

```
input:  [e0b0 e0b1 e0b2 e0b3 | e1b0 e1b1 ...]
output: [e0b0 e1b0 e2b0 ... | e0b1 e1b1 ... | e0b2 ... | e0b3 ...]
```

This clusters sign/exponent-ish bits without bit-level ops and matches common “shuffle filter” practice.

**Bit-plane shuffle (optional v1.1):**  
Transpose to 32 bit-planes. Higher complexity; only add if byte-shuffle ratios disappoint and schedule allows.

Document which variant is active in header flags or a `shuffle_mode` nibble.

**INT8 / FP16:**  
- INT8: shuffle is no-op or identity at element size 1; still may XOR+RLE  
- FP16: 2-byte plane shuffle analogous to FP32

### 5.2 XOR-delta

On the shuffled byte stream `s[0..L)`:

```
out[0] = s[0]
out[i] = s[i] XOR s[i-1]   for i > 0
```

Decode:

```
s[0] = out[0]
s[i] = out[i] XOR s[i-1]
```

Widening: process with `uint64_t` or SIMD chunks carefully (dependency chain on decode — document; may use blocked XOR with restart markers later if needed). **v1:** simple sequential XOR-delta; SIMD can still help encode and other stages.

### 5.3 RLE variant (v1: zero-run + literal)

Goal: exploit zero-dense streams after XOR without doubling incompressible mantissa bytes.

**Locked token format:**

```
0x00, k              => emit (k+1) zero bytes     (1..256 zeros)
0x01, k, <k+1 bytes> => emit (k+1) literal bytes  (1..256 literals)
```

Any other tag is invalid. The encoder only emits a zero token for runs of length ≥ 4 (shorter zeros stay inside the surrounding literal). Worst-case expansion is under ~0.8%.

Classic `[value][run_len]` is **not** used in the pipeline (it doubles random bytes). It may appear only as a bench-side ablation to document why it failed.

Decode must reject truncated streams, unknown tokens, and length mismatches (`std::runtime_error`).

### 5.4 Encode / decode orchestration

**Encode block:**

1. Take up to `block_size` bytes from raw tensor  
2. If SHUFFLE: bit/byte shuffle into temp  
3. If XOR: xor-delta into temp2  
4. If RLE: rle into output buffer  
5. Write block lengths + bytes  

**Decode block:** reverse order with inverse stages.

**Temps:** use thread-local or member scratch buffers sized to `block_size`; no per-block heap alloc on decode hot path after warm-up.

---

## 6. SIMD architecture

### 6.1 Principles

1. Scalar path is the **correctness oracle** — SIMD must match scalar byte-for-byte  
2. Feature-gate with CMake (`TC_HAVE_AVX2`, `TC_HAVE_NEON`)  
3. Dispatch: function pointers or `if constexpr` / separate TUs  
4. Alignment: prefer 32-byte aligned loads when possible; handle tails with scalar  

### 6.2 What to vectorize first

Priority for portfolio impact:

1. **Byte-shuffle / unshuffle** (high arithmetic regularity)  
2. RLE decode scan (harder; optional)  
3. XOR-delta encode (easy); decode is serial — low priority  

### 6.3 AVX2 notes

- Use `<immintrin.h>`  
- Compile SIMD TU with `-mavx2` (GCC/Clang) or `/arch:AVX2` (MSVC)  
- Do not compile SIMD TU on hosts without support; auto-detect in CMake  

### 6.4 Apple Silicon / ARM

If builder host is `arm64`:

- Phase 1 scalar still required  
- Phase 2: implement NEON shuffle helpers **or** document SIMD deferred and tune scalar  
- Do not pretend AVX2 exists  

---

## 7. Module responsibilities

| Module | Responsibility | Must not do |
|--------|----------------|-------------|
| `format` | Header pack/unpack, magic/version checks | Codec math |
| `bitshuffle` | Shuffle/unshuffle | RLE |
| `xor_delta` | Delta XOR encode/decode | Allocate tensors |
| `rle` | Pack/unpack | Know about dtypes |
| `compressor` | Pipeline glue, block loop, metadata | SIMD intrinsics inline spaghetti — call into `simd/` |
| `simd/*` | Vector kernels | Format policy |
| `runtime/layer_pager` | Demo residency pattern | Model math |
| `benchmarks` | Timing + baselines | Change codec semantics |
| `tests` | Correctness | Performance claims |

---

## 8. CLI surface (`src/main.cpp`)

```
tc compress  -i raw.bin -o out.tcf --dtype fp32 --shape 4096,4096
tc decompress -i out.tcf -o raw.out.bin
tc info      -i out.tcf
tc bench     --synthetic mlp|random|zeros --size ...
```

Phase 0 may implement `info` + synthetic bench only.

---

## 9. Benchmark methodology

### 9.1 Metrics

- Compression ratio \(R = N/C\)  
- Encode throughput, decode throughput  
- Optional: peak RSS during decompress (should stay flat with scratch reuse)  

### 9.2 Datasets

1. **zeros** — sanity upper bound  
2. **random FP32** — lower bound / stress lossless  
3. **correlated synthetic** — shared exponents + smooth noise (primary “weight-like”)  
4. **optional real dumps** — named files under `data/` (gitignored if large) with docs on provenance  

### 9.3 Baselines

- **raw**: memcpy timing as bandwidth reference  
- **zlib**: level 6 default (document)  
- **lz4**: default frame or block API (document)  

Link zlib (system or FetchContent); lz4 via FetchContent if needed.

### 9.4 Timing protocol

- Release builds only for published numbers  
- Warm-up ≥ 3 iters; measure ≥ 20 iters; report median  
- Record: CPU brand, OS, compiler version, ISA flags  

---

## 10. Optional layer pager (Phase 3)

```mermaid
sequenceDiagram
  participant DRAM as DRAM_CompressedLayers
  participant Pager as LayerPager
  participant Scratch as ScratchRawBuffer
  participant Compute as FakeMatmul

  Pager->>DRAM: select layer i compressed
  Pager->>Scratch: decompress_to(scratch)
  Scratch->>Compute: consume bytes
  Note over Scratch: reuse same buffer for i+1
```

API sketch:

```cpp
class LayerPager {
  void load_library(std::vector<CompressedTensor> layers);
  std::span<const uint8_t> map_layer(size_t i); // decompress into scratch, return view
};
```

No real GEMM required; `FakeMatmul` can `std::accumulate` or volatile touch bytes to prevent DCE.

---

## 11. Error handling

- Wrong magic / version → throw `std::runtime_error` or return `std::expected` (C++20: prefer exception for v1 simplicity in library, or error codes — **locked: exceptions for v1**)  
- Size mismatch / truncated payload → throw  
- Tests cover happy path + a few corruptions  

---

## 12. Security / robustness note

This is a **trusted local codec** for a portfolio project. Do not claim hardened untrusted-input parsing. Basic bounds checks are still required to avoid buffer overruns during decode (good engineering hygiene).

---

## 13. Mapping from old README claims

| Old README claim | Treatment now |
|------------------|---------------|
| 1.78× ratio, 4.2 GB/s, −18.4% L3 | Removed. Numbers come only from `tc_bench` CSVs under `results/` |
| AVX2 required | Scalar always; NEON on Apple Silicon; AVX2 optional on x86 |
| No malloc on decompress | Enforced via `decompress_to` + `tc_alloc_test` |
| Layer-aware | `LayerPager` residency demo; not ML-layer semantics magic |

---

## 14. Extension points (post-portfolio)

- Format v2: restartable XOR blocks for parallel decode  
- NEON / AVX-512 kernels  
- INT4 nibbles packing after external quantizer  
- Plugin ABI for real runtimes  
- Activation / KV cache experiments  

Do not build these until Phases 0–2 are done.
