# Bench results index

CSV files under [`../results/`](../results/) are the only published numbers.

Produce a new file:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
mkdir -p results
SHA=$(git rev-parse --short HEAD)
./build/tc_bench --csv "results/$(uname -m)_${SHA}.csv"
# faster laptop iteration:
./build/tc_bench --quick --csv "results/$(uname -m)_${SHA}_quick.csv"
```

Each CSV header records chip, compiler, git SHA, SIMD kernel, and measured memcpy bandwidth. Ablation codecs are named `tc-full`, `tc-shuffle`, `tc-sx`, `tc-xr`, `tc-rle`. Baselines include `memcpy`, `zlib-*`, `lz4`, `shuffle+lz4`, `zstd-*` when linked.

After a run, paste the markdown table from stdout into `writeup.md` §6 or keep the CSV as the artifact reviewers open.
