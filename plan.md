# TC-Runtime — Build Plan

**Project:** TensorCompression (TC-Runtime)  
**Audience:** Builder agent + human implementer  
**Goal type:** Portfolio deep-work project (GitHub README + LinkedIn post + personal writeup)  
**Not a goal:** Conference publication, production inference engine, llama.cpp replacement

Read this file together with [`architecture.md`](architecture.md) and [`cursorrules.md`](cursorrules.md) before writing code.

---

## 1. Scope of the problem

### 1.1 Problem statement

Edge and local AI inference is often **memory-bandwidth bound**. Large weight tensors live in DRAM; each layer must stream those bytes across the memory bus into the CPU cache hierarchy before compute can proceed. When the bus saturates, cores stall — the **memory wall**.

Industry default mitigation is **lossy quantization** (e.g. FP16 → INT8/INT4), which reduces bytes transferred but **changes weight values** and can affect model quality.

**TC-Runtime addresses a complementary question:**

> Can we keep weights **bit-exact** (lossless), store them **compressed in DRAM**, and **decompress fast enough with SIMD** (ideally cache-locally, without hot-path heap allocation) that effective bandwidth improves for edge-style workloads?

### 1.2 In scope

| Item | Detail |
|------|--------|
| Lossless codec for tensor blobs | Primary: FP32; secondary: FP16, INT8 byte streams |
| Pipeline | Bit-shuffle → XOR-delta → RLE (and exact reverse) |
| Library + CLI | Compress / decompress / bench entrypoints |
| Scalar reference implementation | Correctness oracle |
| SIMD fast path | Host-appropriate (AVX2 on x86; NEON later if on ARM) |
| Tests | Bit-exact round-trips, edge cases, format version |
| Benchmarks | Ratio + decode/encode throughput vs raw, zlib, lz4 |
| Synthetic tensors | Layer-shaped matrices with tunable correlation |
| Optional Phase 3 | Real weight dumps; mini layer-paging demo |
| Docs | Honest README, `docs/writeup.md` portfolio paper |

### 1.3 Out of scope (explicit non-goals)

- Full integration into llama.cpp / ONNX Runtime / MLX
- Training, fine-tuning, or accuracy recovery after quantization
- Lossy compression, FP8/MX formats, GPTQ/AWQ-style quantizers
- CUDA / Metal / NPU kernels
- KV-cache or activation compression (future work only)
- Fabricating benchmark numbers in README
- Claiming novelty over Bitshuffle/Blosc; we **cite** them and position as NN-weight / edge-runtime oriented engineering

### 1.4 Success criteria (portfolio bar)

A phase is “done” only when the checklist below is met with **measured** evidence:

1. `cmake` + build succeeds on the developer machine  
2. Round-trip decompress(compress(x)) == x (byte-identical) for tested dtypes  
3. At least one synthetic distribution shows ratio **> 1.0×** with the full pipeline  
4. Decode throughput is reported in MB/s or GB/s with methodology notes  
5. Baselines include at least **raw**, **zlib**, and **lz4** (if lz4 is unavailable, document and use zlib + optional zstd)  
6. README metrics table contains **only numbers from this repo’s harness**  
7. `docs/writeup.md` has motivation → method → experiments → failures → next steps  

**Stretch success (nice, not required):** ~1.5–2× on some weight-like data with decode fast enough to discuss “hot path” plausibility; mini paging demo.

---

## 2. Decisions (locked)

These are settled for the builder. Do not reopen unless the human explicitly changes them.

| ID | Decision | Choice | Rationale |
|----|----------|--------|-----------|
| D1 | Project intent | Portfolio deep-work, not conference paper | Learning + LinkedIn/GitHub proof |
| D2 | Compression class | **Lossless only** | Complements quantization; simpler correctness story |
| D3 | Pipeline | Shuffle → XOR-delta → RLE | Matches existing README/API intent |
| D4 | Language / standard | **C++20** | Already chosen; low-level control |
| D5 | Build system | **CMake ≥ 3.20** | Fill empty `CMakeLists.txt` |
| D6 | Primary dtype for Phase 0–2 | **FP32** | Easiest IEEE754 teaching + existing stubs |
| D7 | Secondary dtypes | FP16, INT8 (byte-wise path) | Declared in `DataType`; implement after FP32 works |
| D8 | SIMD strategy | Scalar first, then host SIMD | Correctness before speed; detect ISA at configure/compile time |
| D9 | x86 SIMD | **AVX2** when available | README target; fallback scalar always |
| D10 | ARM | Deferred after scalar works | Mention in writeup; NEON optional Phase 2/4 |
| D11 | Hot-path alloc | Decode into **caller-provided / pre-sized buffers** | Avoid `malloc` on decompress hot path |
| D12 | Storage in API | May use `std::vector` at API boundary | Internals use raw pointers + sizes for kernels |
| D13 | Format | **Versioned binary header** + payload | Forward compatible; see `architecture.md` |
| D14 | Baselines | raw, zlib, lz4 | Honest comparison; no zlib-only strawman |
| D15 | README numbers | **Measured only** | Delete/replace aspirational table until harness exists |
| D16 | Runtime integration | Optional thin paging demo only | Not a full inference engine |
| D17 | Quantization stance | **Complementary**, not replacement | Writeup/README must say this |
| D18 | Code style | Clear > clever; readable for learning | Human is learning these concepts |
| D19 | Existing stubs | Evolve `include/*.h`; implement empty `src/*` | Fix `CompressedTensor` missing `;` |
| D20 | Phasing | Ship Phase 0→1 before SIMD/benches polish | Builder default start: Phase 0–1 |

---

## 3. Current repo state (as of plan authoring)

```
tensor-compression/
├── CMakeLists.txt          # empty
├── README.md               # vision + aspirational metrics (replace when measured)
├── include/tensor.h        # RawTensor / CompressedTensor / DataType (fix trailing ;)
├── include/compressor.h    # TensorCompressor API + TODOs
├── src/{compressor,tensor,main}.cpp  # empty
├── benchmarks/             # empty dir
└── tests/                  # empty dir
```

Nothing builds yet. Treat README performance table as **targets/hypotheses**, not facts.

---

## 4. Target repo layout

```
tensor-compression/
├── CMakeLists.txt
├── plan.md                 # this file
├── architecture.md
├── cursorrules.md
├── README.md
├── include/
│   ├── tensor.h
│   ├── compressor.h
│   └── format.h            # on-disk / in-memory compressed frame layout
├── src/
│   ├── main.cpp            # CLI: compress | decompress | info
│   ├── tensor.cpp
│   ├── compressor.cpp      # orchestration
│   ├── codec/
│   │   ├── bitshuffle.cpp / .h
│   │   ├── xor_delta.cpp / .h
│   │   └── rle.cpp / .h
│   ├── simd/
│   │   ├── bitshuffle_scalar.cpp
│   │   └── bitshuffle_avx2.cpp   # ifdef’d / optional target
│   └── runtime/            # Phase 3 optional
│       └── layer_pager.cpp
├── tests/
│   ├── test_roundtrip.cpp
│   └── test_format.cpp
├── benchmarks/
│   └── bench_codec.cpp
├── tools/                  # optional
│   └── gen_synthetic_tensor.cpp
└── docs/
    └── writeup.md
```

Builder may adjust file names slightly but must preserve **module boundaries**: codec stages separable for ablations.

---

## 5. Phased implementation plan

### Phase 0 — Foundations (build + format + synthetic data)

**Objectives**
- Project compiles and links a trivial binary
- Headers compile; public types are consistent
- Compressed frame format specified and encoded/decoded (even if payload is raw/uncompressed passthrough initially)

**Tasks**
1. Implement `CMakeLists.txt`: C++20, `tc_runtime` library, `tc` CLI, `tc_tests`, `tc_bench`
2. Fix `CompressedTensor` semicolon and any header hygiene
3. Add `include/format.h` with magic, version, flags, dtype, ndims/shape, uncompressed size, payload size
4. Implement synthetic FP32 generator: shapes like `[4096, 4096]`, `[11008, 4096]` (MLP-ish), controllable exponent correlation / noise
5. CLI stub: `tc info` prints tensor/format metadata

**Exit criteria**
- [ ] `cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build`
- [ ] At least one test executable runs
- [ ] Format round-trip for header fields

**Builder prompt hint:** “Implement Phase 0 only from plan.md.”

---

### Phase 1 — Correct scalar codec (the core)

**Objectives**
- Full lossless pipeline works in scalar C++
- Bit-exact round-trip on synthetic FP32

**Tasks**
1. `bit_shuffle` / `bit_unshuffle` for FP32 (byte/bit plane transpose as specified in architecture)
2. `xor_delta_encode` / `xor_delta_decode` on shuffled stream
3. `rle_encode` / `rle_decode` (document exact RLE variant in architecture; keep simple)
4. `TensorCompressor::compress` / `decompress` orchestration + metadata (`block_count`, `compression_ratio_x100`)
5. Tests: empty, tiny, odd sizes, non-multiple-of-32, all-zeros, high-entropy random, correlated synthetic weights
6. First ratio report printed by bench or test logger (not written into README until Phase 2 methodology is set)

**Exit criteria**
- [ ] All round-trip tests pass
- [ ] Correlated synthetic data compresses to ratio > 1.0×
- [ ] Random FP32 may compress poorly — assert only losslessness, not ratio

**Checkpoint narrative:** “I built a lossless float tensor codec.”

---

### Phase 2 — Speed + honest benchmarks

**Objectives**
- Fast decode path where host allows
- Publishable (portfolio) numbers with clear methodology

**Tasks**
1. Detect AVX2 (or NEON) via CMake checks; compile SIMD TU only when supported
2. SIMD accelerate the hottest stage(s) — prefer **unshuffle + XOR undo** on decode path first
3. Bench harness: warm-up, many iters, report encode MB/s, decode MB/s, ratio
4. Baseline adapters: memcpy (raw), zlib, lz4
5. Rewrite README results section with measured table + CPU model + compiler flags
6. Ablation flags: shuffle-only, shuffle+xor, full pipeline

**Exit criteria**
- [ ] SIMD or documented “scalar-only on this host” with optimized scalar
- [ ] README table fully measured
- [ ] Ablation numbers recorded in `docs/writeup.md` or bench output checked in as `docs/results.md`

**Checkpoint narrative:** “Decode is fast enough to discuss memory-wall relevance.”

---

### Phase 3 — Credibility extras

**Objectives**
- Make the story feel like edge runtime engineering, not only a toy codec

**Tasks**
1. Optional: script/notes to export a few tensors from a small open model (safetensors → raw dump)
2. Optional: `runtime/layer_pager` — N compressed layers in memory; decompress layer i into reusable scratch; time load+decompress vs raw load
3. Expand writeup: failures, surprises, what didn’t compress
4. Draft LinkedIn outline in `docs/linkedin_draft.md`

**Exit criteria**
- [ ] Writeup sections 1–8 present (see below)
- [ ] At least one “what failed” subsection with real evidence

---

### Phase 4 — Polish

**Tasks**
1. README: problem diagram, build instructions, measured results, limitations, next steps
2. API cleanup; remove dead stubs; consistent naming (SSE-named methods → `bit_shuffle_simd` or keep SSE as subset with clear comments)
3. Next-steps list: ARM NEON, post-quant INT8 packing, real runtime plugin
4. Ensure `cursorrules.md` still matches code reality

---

## 6. Portfolio writeup structure (`docs/writeup.md`)

Create in Phase 1–3; do not wait until the end for an empty stub.

1. Motivation — memory wall, edge constraints  
2. Background — quantization vs lossless; SIMD; caches; related work (Bitshuffle, Blosc, lz4/zstd)  
3. Method — pipeline + format  
4. Implementation — C++20, module map, design choices  
5. Experiments — hardware, datasets, methodology  
6. Results — tables/figures from harness  
7. What didn’t work — trial and error  
8. Next steps  

---

## 7. Testing & measurement rules

1. **Losslessness is sacred.** Any failure to round-trip is a P0 bug.  
2. Never commit benchmark numbers you did not produce from `tc_bench` (or documented command).  
3. Report: CPU, ISA flags, `CMAKE_BUILD_TYPE=Release`, dataset name/shape, iterations.  
4. Prefer median of N runs over single-shot.  
5. Compare decode throughput **and** ratio; a slow high-ratio codec loses the plot.

---

## 8. Suggested builder session order

| Session | Work |
|---------|------|
| 1 | Phase 0 |
| 2 | Phase 1 codec stages + round-trip tests |
| 3 | Phase 2 SIMD + benches + README honesty pass |
| 4 | Phase 3–4 writeup / paging demo / polish |

Human learning parallel track: see **[`learning.md`](learning.md)** (what to read before each phase, what to code yourself vs agent, interview depth map, 4-week cadence).

**Short version:** read 30–60 min → paper sketch → you implement one small function → agent reviews/unblocks → you explain it out loud. Do not let the agent own all of Phase 1 silently.

---

## 9. Risks & mitigations

| Risk | Mitigation |
|------|------------|
| Random weights don’t compress | Use correlated synthetics + real weights; report both |
| SIMD host mismatch (e.g. Apple Silicon) | Scalar always works; NEON path or document deferral |
| Scope creep into full runtime | Hard stop at optional pager demo |
| Overclaiming vs Blosc | Cite prior art; emphasize NN tensor framing + measured edge-oriented eval |
| Aspirational README left in place | Phase 2 exit criterion forces replacement |

---

## 10. Definition of “project complete” for portfolio

- Phases 0–2 done  
- Phase 3 writeup done (paging demo optional)  
- README and writeup consistent  
- Human can demo: build → round-trip → show bench table → explain pipeline in 2 minutes  

Phase 4 is polish, not a blocker for the LinkedIn post.
