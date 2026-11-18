# LinkedIn draft (ready to paste)

Hook: Small-batch LLM inference is often memory-bandwidth bound — cores wait on DRAM, not TFLOPS.

What I built: **TC-Runtime** — a C++20 lossless tensor codec (byte-shuffle → XOR-delta → zero-run RLE) with a versioned binary format (TCF1), a scalar correctness oracle, a NEON fast path, bit-exact tests (including malformed frames), and a zero-allocation decode hot path.

What I measured (Apple Silicon, ~66 GB/s single-thread memcpy):
- iid Gaussian FP32 and real **GPT-2** weight matrices ≈ **1.0×** (stored-raw fallback)
- Correlated synthetics ≈ **1.13–1.32×**; zeros crush
- Cold 32 MiB runs match hot ratios
- Encoded-path decode ≈ **2–4 GB/s** → serial break-even vs DRAM is a **LOSS** here

The point isn’t “we beat the memory wall.” It’s an honest systems study: complementary to quantization, cited against Bitshuffle/Blosc/lz4/ZipNN/DFloat11, with methodology and failure analysis in the writeup.

Repo: https://github.com/sakshamrai101/tensor_compression  
Writeup: https://github.com/sakshamrai101/tensor_compression/blob/main/docs/writeup.md
