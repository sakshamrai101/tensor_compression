# LinkedIn draft (fill measured numbers from results CSV)

Hook: Small-batch LLM inference is often memory-bandwidth bound — cores wait on DRAM, not TFLOPS.

What I built: TC-Runtime, a C++20 lossless tensor codec (byte-shuffle → XOR-delta → zero-run RLE) with a versioned binary format, scalar oracle, NEON fast path, and bit-exact tests including malformed frames and a zero-allocation decode hot path.

What I measured: On Apple Silicon, iid Gaussian FP32 barely compresses (~1.0× with a stored-raw fallback); correlated synthetics reach ~1.1–1.3×; zeros crush. Decode on the encoded path is a few GB/s; break-even against ~56 GB/s DRAM is harsh — so the writeup says exactly when this helps and when it doesn’t.

Positioning: complementary to quantization; cites Bitshuffle/Blosc/lz4/ZipNN/DFloat11 rather than claiming novelty.

Repo: [link] · Writeup: docs/writeup.md
