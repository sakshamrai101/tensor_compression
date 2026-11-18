# Bench results index

CSV files under [`../results/`](../results/) are the only published numbers.

| File | What |
|------|------|
| `arm64_63a8ced_full.csv` | Full hot+cold harness (primary) |
| `arm64_63a8ced_full.log` | Full stdout including break-even table |
| `arm64_874c29d_quick.csv` | Earlier `--quick` smoke |
| `gpt2_ratios.txt` | Real GPT-2 FP32 layer ratios |

Produce a new file:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
mkdir -p results
SHA=$(git rev-parse --short HEAD)
./build/tc_bench --csv "results/$(uname -m)_${SHA}_full.csv"
./build/tc_bench --quick --csv "results/$(uname -m)_${SHA}_quick.csv"
```

Each CSV header records chip, compiler, git SHA, SIMD kernel, and measured memcpy bandwidth. Ablation codecs: `tc-full`, `tc-shuffle`, `tc-sx`, `tc-xr`, `tc-rle`. Baselines: `memcpy`, `zlib-*`, `lz4`, `shuffle+lz4`, `zstd-*` when linked.
