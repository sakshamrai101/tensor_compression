# TC-Runtime — Learning & Ownership Guide

How to actually **understand** this project deeply (for interviews and for yourself), given rusty C++, while still shipping with a builder agent.

Read with [`plan.md`](plan.md) and [`architecture.md`](architecture.md).

---

## 0. START HERE — your first session (do this tonight)

You do **not** start by writing the full compressor. Step one is: **make the project build**, then touch one tiny piece of C++ yourself.

### Right now (no builder yet) — ~20 minutes

1. Open [`architecture.md`](architecture.md) §3 (`RawTensor` / `CompressedTensor`) and skim it once. Don’t memorize.
2. Open [`include/tensor.h`](include/tensor.h).
3. **You fix this yourself:** `CompressedTensor` is missing a `;` after the closing `}`. Add it. Save.
4. Create `docs/notes.md` with today’s date and one line: `Session 0: fixed semicolon, starting Phase 0.`

That is already real progress.

### Then open a **new** Cursor chat as Builder — paste this exact prompt

```text
You are the BUILDER for TC-Runtime. You are NOT free to invent architecture.

MANDATORY READ (do this BEFORE writing any code):
1. Read plan.md fully — scope, locked decisions (D1–D20), current phase exit criteria
2. Read architecture.md fully — TCF1 format, algorithms, module boundaries, what is deferred
3. Read cursorrules.md fully — hard constraints, collab mode, anti-patterns
4. Read learning.md — especially §0 START HERE and the HUMAN-TODO / human-vs-agent split

After reading, in your first reply (before big code dumps):
- Summarize in 5–8 bullets what we are building and what this session will NOT do
- Name the phase you are implementing and quote the relevant exit criteria from plan.md
- Confirm you will follow architecture.md (not redesign the pipeline/format on your own)

If anything in those docs conflicts with an impulse to "just build something nicer," the .md files win. Update the docs only if I explicitly ask; otherwise implement what they say.

COLLAB MODE (mandatory):
- You write ALL boilerplate: CMake, file skeletons, includes, stubs that compile, tests/CLI shells, comments.
- Leave clearly marked HUMAN-TODO regions for the impactful logic I must write myself.
- Use this exact comment style:

  // =============================================================================
  // HUMAN-TODO: <short title>
  // WHY: <1-2 lines — concept I should understand>
  // DO: <exact steps / signature to implement>
  // HINT: <optional tiny hint, not the full solution>
  // READ: <architecture.md section or learning.md topic>
  // =============================================================================

- Do NOT silently fill HUMAN-TODO bodies with the real algorithm unless I ask.
- You MAY implement trivial glue around my TODOs so the project builds.
- HUMAN-TODO "READ:" lines must point at real sections in our .md docs (not vague "google it").
- After finishing, list every HUMAN-TODO you left (file + title) so I know what to code next.
- Do NOT invent benchmark numbers. Do NOT start later phases.
- Do NOT ignore or contradict plan.md / architecture.md / cursorrules.md / learning.md.

THIS SESSION = Phase 0 slice A only (per plan.md Phase 0):
1. CMakeLists.txt: C++20, library `tc_runtime`, executable `tc`
2. src/main.cpp prints "tc-runtime phase0" and returns 0 (or calls into a stub)
3. Minimal stubs so empty/nearly-empty src/compressor.cpp and src/tensor.cpp compile
4. Fix compile blockers in headers if needed (e.g. CompressedTensor trailing `;`) — if you fix the semicolon, leave a short comment that I should understand why it was required
5. Add include/format.h skeleton for TCF1 with HUMAN-TODO for me to fill the struct fields from architecture.md §4 (you write the file shell + pack/unpack function declarations; I fill the struct layout)
6. Add element_size / numel declarations with HUMAN-TODO in tensor.h or tensor.cpp for me to implement
7. Tell me exact build/run commands for my machine

Out of scope this session (plan.md / architecture.md): shuffle, XOR, RLE, SIMD, benches, README metric rewrites, llama.cpp/runtime integration, redesigning the codec.
```

### After builder finishes — you verify (~10 minutes)

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build -j
./build/tc
```

You want to see `tc-runtime phase0` printed. If it fails, paste the error to the builder.

### Same evening or next day — fill the HUMAN-TODOs (~30–45 min)

Builder should have left TODOs like `element_size` / `numel` and TCF1 struct fields.  
Open those files, fill **one** TODO at a time, then use the review prompt in §9.

If a TODO is unclear, use the “stuck mid-TODO” prompt — don’t ask the builder to fill all TODOs for you.

### What you are NOT doing in session 0

- No bit-shuffle, XOR, RLE  
- No SIMD  
- No benchmarks  
- No README performance table  

**Definition of done for tonight:** `./build/tc` runs + you wrote `element_size` / `numel` + a note in `docs/notes.md`.

---

## 1. How to use the builder without becoming a passenger

**Rule of thumb:** the agent can scaffold, unblock syntax, and write boilerplate. **You** must own the ideas you would explain on a whiteboard.

| You own (do yourself, or type with agent coaching) | Agent can draft first, you must rewrite/explain | Agent can fully own |
|----------------------------------------------------|--------------------------------------------------|---------------------|
| Explaining memory wall, lossless vs quant | CMakeLists plumbing | Fetching lz4/zlib wiring once you approve |
| Drawing shuffle → XOR → RLE on paper | Format header pack/unpack | CI / file renames |
| Implementing **scalar** shuffle *or* XOR *or* RLE at least once by hand | AVX2 intrinsics (hard; learn by reading diffs) | Benchmark harness skeleton |
| Walking a failing round-trip in a debugger | Multi-file refactors | Docs formatting |
| Interpreting bench numbers and writing README results | NEON port later | |

**Interview test:** if you cannot explain a file’s job in 60 seconds without looking, you didn’t own it — go back and re-implement or annotate it.

---

## 2. Cadence that works when you’re slow at C++

Do **not** try to “finish Phase 1 in one weekend.” Use this loop:

1. **Read 30–60 min** (concept for the phase — links below)  
2. **Paper 10 min** — draw the data layout / pipeline for *your* tensors  
3. **Builder scaffold** — agent writes boilerplate + `HUMAN-TODO` stubs (see §0 prompt)  
4. **You fill HUMAN-TODOs** — 45–90 min on one stub  
5. **Ask when stuck** — “explain this TODO”, “why won’t this compile”, “review my fill-in”  
6. **Explain out loud** — 2 min voice note or rubber-duck paragraph in `docs/notes.md`  
7. Stop. Next day continue.

### Default collab pattern (scaffold → you fill → review)

```
Builder: CMake, files, declarations, HUMAN-TODO blocks, tests that call your functions
You:     implement the TODO bodies (helpers, then XOR, RLE, shuffle, …)
Builder: review, fix compile errors, wire next TODO, quiz you once
```

### Prompt patterns that keep you learning

- “Don’t write the codec yet. Quiz me on IEEE754 and how byte-shuffle helps. Then I’ll try shuffle.”  
- “Scaffold Phase 1 XOR with HUMAN-TODO only — I’ll fill the loop.”  
- “Here’s my fill-in for the HUMAN-TODO in xor_delta.cpp. Review; don’t rewrite unless broken.”  
- “Explain this AVX2 intrinsic line-by-line like I’m intermediate.”

### Prompt patterns that make you a passenger (avoid)

- “Implement Phase 0–2 fully.” (unless you explicitly want a full draft later)  
- “Just make it work and update the README with good numbers.”  
- “Write the whole compressor while I watch.”  
- “Fill in all the HUMAN-TODOs for me.”

---

## 3. What to implement yourself (by phase)

### Phase 0 — Foundations
**You should do:**
- Fix the `CompressedTensor` `;` yourself (trivial win)
- Write `element_size(DataType)` and a `product(shape)` helper yourself
- Sketch the 64-byte TCF1 header on paper from `architecture.md`, then type the struct

**Agent can do:**
- First working `CMakeLists.txt` (then you read every target and explain them)
- CLI argument stub

**You must be able to say:** what `RawTensor` stores; what “row-major tightly packed” means; why `alignas(64)`.

### Phase 1 — Scalar codec (your core interview meat)
**You should implement at least one full stage by hand:**

**Recommended order for *your* hands:**
1. **XOR-delta encode/decode** (shortest, builds confidence)  
2. **Classic RLE** `[value][run_len]`  
3. **Byte-shuffle / unshuffle** for FP32 (hardest conceptually — worth owning)

**Agent role:** write tests *around* your functions; help with C++ iterators/spans; review edge cases (odd sizes, empty, non-multiples of 4).

**Do not** let the agent own all three stages silently. Minimum bar: you typed the core loop of each stage at least once, even if the agent cleaned it up later.

### Phase 2 — SIMD + benches
**You should do:**
- Read the scalar shuffle and mark which loops are vectorizable  
- Run benches yourself; fill README with *your* machine’s numbers  
- Explain why decode throughput matters (inequality in `architecture.md` §2.2)

**Agent can do:**
- First AVX2/NEON draft  
- zlib/lz4 baseline wiring  

**You must then:** sit with the SIMD file and comment each intrinsic in your own words (`docs/notes_simd.md`).

### Phase 3–4 — Story
**You own 100%:** writeup narrative, “what failed”, LinkedIn draft. Agent can edit for clarity, not invent your insights.

### HUMAN-TODO comment contract (builder must use)

```cpp
// =============================================================================
// HUMAN-TODO: <short title>
// WHY: <concept>
// DO: <what to implement>
// HINT: <optional>
// READ: <doc section>
// =============================================================================
```

Builders write boilerplate + these stubs. Humans fill bodies. Later phases reuse the same pattern (e.g. `HUMAN-TODO: xor_delta_encode loop`).

---

## 4. Concept curriculum (what to read before each phase)

Resources are mostly free. Depth over volume.

### Before Phase 0 — “What is a tensor in memory?”

| Concept | Learn from | Stop when you can… |
|---------|------------|-------------------|
| Arrays vs tensors as contiguous bytes | Any C++ array/buffer tutorial; [CppReference `std::vector`](https://en.cppreference.com/w/cpp/container/vector) | Compute byte size from shape × dtype |
| Cache lines (64B intuition) | “What every programmer should know about memory” (PDF, Ulrich Drepper) — **skim ch. 1–3 only** | Say why aligning structs matters at a high level |
| CMake basics | Official CMake tutorial “A Basic Starting Point” | Add one library + one executable |

**Optional video vibe:** search “CPU cache explained” (any short systems intro). Don’t go down OS rabbit holes yet.

### Before Phase 1 — “Why floats compress poorly / how our pipeline works”

| Concept | Learn from | Stop when you can… |
|---------|------------|-------------------|
| IEEE754 FP32 layout (sign/exp/mantissa) | [Wikipedia IEEE 754](https://en.wikipedia.org/wiki/IEEE_754) dual diagram; or [float.exposed](https://float.exposed) | Draw one float’s bits; explain exponent bias roughly |
| Why raw floats defeat zlib | Intuition: adjacent floats share exponents but byte streams look high-entropy — write this in your notes | Explain why shuffle helps entropy coders / RLE |
| Bitshuffle / Blosc idea | Skim [Bitshuffle README](https://github.com/kiyo-masui/bitshuffle) + Blosc “shuffle filter” docs | Contrast their general arrays vs our NN-weight framing |
| Delta / XOR encoding | Any data-compression notes on delta coding; XOR as reversible delta on bits | Hand-compute XOR-delta on 8 bytes |
| RLE | Wikipedia Run-length encoding — enough | Encode `0000aa00` by hand with `[val][len]` |

**Paper/article (skim, don’t memorize):**  
- “Floating-point compression” survey blogs are fine; ZFP/SZ are *lossy* — know they exist so you don’t confuse them with us.

### Before Phase 2 — “SIMD and measurement”

| Concept | Learn from | Stop when you can… |
|---------|------------|-------------------|
| SIMD mental model | [Agner Fog](https://www.agner.org/optimize/) microarchitecture *intro* pages, or “SIMD introduction” for AVX2 | Explain “4× float lanes” without jargon spiral |
| Intrinsics | Intel Intrinsics Guide (lookup `_mm256_*` as needed) | Say what one load/store/shuffle intrinsic does |
| Benchmark honesty | “Pitfalls of microbenchmarks” any short article; always Release mode | List 3 ways to lie with benches |

### Before Phase 3 — “Edge runtime story”

| Concept | Learn from | Stop when you can… |
|---------|------------|-------------------|
| Memory wall / bandwidth vs compute | Classic roofline intro (any short blog) | Explain compute-bound vs bandwidth-bound |
| Why LLMs page weights | Skim llama.cpp / “weight streaming” discussions at high level | Describe compressed-resident + scratch decompress |
| Quantization basics | HF or PyTorch quant primer (INT8) | Explain why we *complement* quant |

---

## 5. Interview depth map (what “deep” means here)

You are not expected to invent new compression theory. You **are** expected to go deep on *your* system.

### Must-crush (prepare stories + whiteboard)

1. **Problem:** memory wall for edge inference; bytes on the bus  
2. **Design:** shuffle → XOR → RLE; why that order  
3. **Lossless guarantee:** reverse pipeline; tests  
4. **Format:** TCF1 header fields; why blocked  
5. **Tradeoff:** ratio vs decode speed; when compression *hurts*  
6. **Measurement:** how you benched; one real number from your machine  
7. **Prior art:** Bitshuffle/Blosc/lz4 — what you reused conceptually  

### Should-know (if they poke)

- IEEE754 byte planes vs bit planes  
- Why XOR-decode is sequentially dependent  
- `alignas(64)`, scratch reuse, no malloc on hot path  
- Ablations: what each stage contributed on *your* data  

### Nice-to-have

- AVX2 unshuffle details  
- Roofline sketch  
- How you’d plug into a real runtime  

### Practice prompt (weekly)

> “Explain TC-Runtime to a senior engineer in 5 minutes, then defend why you didn’t just use zstd.”

Record yourself. Painful = useful.

---

## 6. Concrete 4-week starter schedule (adjust to your pace)

| Week | Focus | Hands-on | Agent assist |
|------|-------|----------|--------------|
| 1 | Phase 0 + C++ warm-up | Header struct, helpers, build locally | CMake |
| 2 | XOR + RLE yourself | Both stages + unit tests | Test harness polish |
| 3 | Byte-shuffle yourself | Shuffle/unshuffle + full round-trip | Debugging help |
| 4 | Benches + write notes | Run benches, write `docs/writeup.md` §1–4 | Baseline adapters |

If a week slips, **slip**. Depth > schedule.

---

## 7. Keep a learning lab notebook

Create `docs/notes.md` and after each session add:

```markdown
## YYYY-MM-DD
- Built / broke:
- Concept I learned:
- C++ thing that confused me:
- Question for next session:
- Could I explain this in an interview? (yes/no)
```

This notebook becomes LinkedIn gold (“what I struggled with”) and interview recall.

---

## 8. C++ rustiness — minimum revival kit

Before Phase 1 coding days, refresh only this:

- types: `uint8_t`, `size_t`, `std::vector`, references vs pointers  
- loops, `span` (C++20) or pointer+size  
- headers / `#pragma once`  
- how to compile one file with CMake  

Skip templates metaprogramming, inheritance gardens, Boost.

**Practice repo drill:** write `xor_delta` in a single `main.cpp` before integrating into the library.

---

## 9. What to tell the builder agent (copy-paste)

**Phase start / scaffold** — use the big prompt in **§0 START HERE**.

**After scaffold, when you want review:**

```text
Read plan.md, architecture.md, cursorrules.md, and learning.md.

I filled the HUMAN-TODO titled "<title>" in <file>.
Review my implementation only. Do not rewrite it unless it is wrong.
If wrong, explain the bug and give a hint before pasting a fix.
Quiz me once on the WHY of that TODO.
Do not start the next phase or fill other HUMAN-TODOs.
```

**When stuck mid-TODO:**

```text
I'm working on HUMAN-TODO: <title> in <file>.
Here's what I tried / my current code.
Don't complete it for me yet — explain the concept and the next 3 lines I should write.
```

---

## 10. Definition of “I understand this project”

You’re ready for interview deep-dives when you can:

1. Whiteboard the pipeline and invert it  
2. Compute by hand shuffle/XOR/RLE on ~8–16 bytes  
3. Open your `compressor.cpp` and narrate the block loop  
4. Show a bench command and interpret the table  
5. Honestly say what still confuses you (SIMD lanes / RLE edge cases / etc.)

That last point is strength, not weakness — seniors do it too.
