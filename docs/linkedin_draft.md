# LinkedIn draft (fill measured numbers from results CSV)

Hook: Small-batch LLM inference is often memory-bandwidth bound — cores wait on DRAM, not TFLOPS.

What I built: TC-Runtime, a C++20 lossless tensor codec (byte-shuffle → XOR-delta → zero-run RLE) with a versioned binary format, scalar oracle, NEON fast path, and bit-exact tests including malformed frames and a zero-allocation decode hot path.

What I measured: On Apple Silicon (~66 GB/s single-thread memcpy), iid Gaussian and real GPT-2 FP32 matrices sit at ~1.0× (stored-raw fallback); correlated synthetics reach ~1.13–1.32×; zeros crush. Cold 32 MiB runs match hot ratios. Encoded-path decode is ~2–4 GB/s, so break-even against DRAM is a LOSS here — the writeup says exactly when this would help (lower-bandwidth edge) and when it doesn’t.

Positioning: complementary to quantization; cites Bitshuffle/Blosc/lz4/ZipNN/DFloat11 rather than claiming novelty.

Repo: [link] · Writeup: docs/writeup.md
