# Cursor / Builder Rules — TC-Runtime

**Always apply** when working in this repository.  
Also read [`plan.md`](plan.md) and [`architecture.md`](architecture.md) before implementing.

If your environment supports project rules, mirror this file into `.cursor/rules/tc-runtime.mdc` with `alwaysApply: true`.

---

## 1. Who the human is

- Learning low-level systems, SIMD, and memory-hierarchy concepts while building this.
- **Rusty C++** — expect slow progress; explain syntax; prefer small diffs; quiz rather than dump.
- Wants **deep work**, not conference-novelty theater.
- Deliverables that matter: working code, honest benchmarks, GitHub README, LinkedIn-ready story, portfolio writeup.
- Prefer teaching-quality clarity over clever abstractions.
- Follow [`learning.md`](learning.md): human should own scalar codec stages; do not silently implement all of Phase 1 unless they explicitly ask for a full draft.

---

## 2. Mission

Build **TC-Runtime**: a lossless tensor compression library (bit-shuffle → XOR-delta → RLE) aimed at edge-AI **memory bandwidth** problems, with optional thin layer-paging demo.

Complement quantization; do not market this as replacing it.

---

## 3. Hard constraints

1. **Lossless only.** `decompress(compress(x))` must be byte-identical to `x` for supported inputs.  
2. **No fabricated metrics.** Never invent README/benchmark numbers. If unknown, write “TBD — run `tc_bench`”.  
3. **Replace aspirational README table** once the harness exists; until then, label claims as targets/hypotheses.  
4. **Follow phases in `plan.md`.** Default: implement Phase 0 → Phase 1 before SIMD heroics.  
5. **Do not expand into** llama.cpp/ONNX integration, CUDA/Metal/NPU, KV-cache compression, or training code unless the human explicitly asks.  
6. **Do not delete** `plan.md`, `architecture.md`, `learning.md`, or this file.  
7. **Match `architecture.md` format (`TCF1`)** when implementing the frame layout. If you must deviate, update `architecture.md` in the same change.  
8. **Scalar oracle first.** Any SIMD path must match scalar output exactly (test it).  
9. **Hot-path decode:** avoid allocations inside per-block decompress after buffers are warmed; provide `decompress_to`.  
10. **C++20 + CMake** only for the core library (no Python rewrite of the codec). Python helper scripts for data export are OK under `tools/` if needed.  
11. **Collab mode:** default in `learning.md` is scaffold + `HUMAN-TODO` for impactful logic. **Suspended for the Phase 0–3 full-implementation Builder chunk** at the human’s request (implement stages fully; keep code teachable). Re-enable collab mode when the human asks to learn by filling stubs again.

---

## 4. Engineering standards

### Code

- Clear names: `bit_shuffle`, `xor_delta_encode`, `rle_encode`, …
- Small modules under `src/codec/` and `src/simd/` — one concern per file.
- Prefer `stdint` / explicit sizes in format code.
- Check bounds when reading compressed streams.
- Fix known stub bug: `CompressedTensor` needs a trailing `;` in `include/tensor.h`.
- Use `alignas(64)` as in existing headers; keep public API ergonomic.
- Comments explain **why** (especially IEEE754 shuffle rationale, RLE token layout), not narrate obvious code.
- No dead empty files left without a note; either implement or remove from CMake targets.

### Build

- CMake ≥ 3.20, `CMAKE_CXX_STANDARD 20`
- Separate targets: library, CLI, tests, bench
- Feature-detect AVX2/NEON; never hard-require AVX2 on ARM hosts
- Release flags for benches (`-O3`); debug OK for tests

### Tests

- Bit-exact round-trips are mandatory before claiming a stage “done”
- Include: zeros, random, correlated synthetic, odd sizes, multi-block
- Fast enough to run locally every change

### Benchmarks

- Report methodology (CPU, flags, iters, median)
- Baselines: raw/memcpy, zlib, lz4 (document if a baseline is missing)
- Ablations via format flags when possible

### Docs

- Keep README build instructions accurate
- Put long-form narrative in `docs/writeup.md`
- Cite Bitshuffle/Blosc/lz4 as related engineering prior art; don’t overclaim novelty

---

## 5. Decision authority

Locked decisions live in `plan.md` §2. Do not reopen them casually.

If stuck between two designs:

1. Prefer what `architecture.md` says  
2. Prefer simplicity and teachability  
3. Ask the human only when the choice changes portfolio scope (e.g. adding CUDA)

---

## 6. Session workflow for agents

At session start:

1. **Read first, code second:** `plan.md`, `architecture.md`, `cursorrules.md`, `learning.md`  
2. In the first reply, summarize the mission + this session’s phase/exit criteria + what is out of scope (from the docs)  
3. Treat the `.md` files as source of truth — do not invent a different pipeline, format, or scope  
4. `git status` + skim existing `include/`  
5. Implement **one phase (or a clear sub-slice)** end-to-end using collab mode (`HUMAN-TODO` for impactful logic)  
6. Build + run tests (or tell the human the exact commands)  
7. List remaining `HUMAN-TODO`s and next Phase tasks  

At session end:

- List files touched  
- Note next Phase tasks  
- Never leave README with fake performance numbers  
- Never “improve” the design in a way that contradicts the docs without asking  

---

## 7. Suggested commit style (only when human asks to commit)

- `build: cmake skeleton and format header`
- `feat: scalar bitshuffle xor rle roundtrip`
- `feat: avx2 unshuffle decode path`
- `bench: add zlib lz4 baselines`
- `docs: measured results and writeup`

Do not commit unless the human explicitly requests it.

---

## 8. Tone in generated docs / README

- Direct, technical, honest about limitations  
- No hype tables without data  
- OK to sound ambitious in motivation; **results must be earned**  
- Explain memory wall + lossless vs quant in accessible language for LinkedIn/README audiences  

---

## 9. Definition of done (per phase)

Use checkboxes in `plan.md`. A phase is not done when “code exists”; it is done when **exit criteria** are met (build, tests, and any required measurements).

---

## 10. Quick command cheatsheet (target)

```bash
export PATH="$PWD/.venv/bin:$PATH"   # if cmake lives in the project venv
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
./build/tc_bench --quick --csv results/local.csv
./build/tc info -i example.tcf
```

---

## 11. Anti-patterns (do not do)

- Gigantic single `compressor.cpp` with all stages inlined and unnamed constants for the format  
- Skipping tests because “SIMD is more interesting”  
- Adding dependencies unrelated to codec/bench (Boost, Eigen, CUDA toolkits, etc.)  
- Quietly changing the pipeline order without updating docs  
- Leaving the old aspirational metrics table presented as measured fact  
- Implementing a full transformer runtime “while you’re at it”
