#!/usr/bin/env python3
"""Export GPT-2 (or other) safetensors / PyTorch weights to raw .bin dumps for TC-Runtime.

Prefer natively FP32 checkpoints. BF16/FP16 upcast to FP32 leaves zero low bits and
inflates lossless ratios — those dumps are written under data/upcast_bf16/ if requested.

Usage:
  python3 tools/export_safetensors.py --model gpt2 --out-dir data/gpt2
  python3 tools/export_safetensors.py --model gpt2 --out-dir data/gpt2 --max-tensors 8

Requires: pip install numpy torch (transformers optional for Hub download).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gpt2", help="Hugging Face model id (default: gpt2)")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max-tensors", type=int, default=0, help="0 = all")
    parser.add_argument(
        "--allow-upcast",
        action="store_true",
        help="Also dump tensors that were upcast from non-FP32 (flagged in the manifest)",
    )
    args = parser.parse_args()

    try:
        import numpy as np
        import torch
    except ImportError:
        print("Need numpy and torch: pip install numpy torch", file=sys.stderr)
        return 1

    try:
        from transformers import AutoModel
    except ImportError:
        print("Need transformers: pip install transformers", file=sys.stderr)
        return 1

    print(f"loading {args.model} …", flush=True)
    model = AutoModel.from_pretrained(args.model)
    model.eval()

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "model": args.model,
        "note": "natively_fp32 preferred; upcast tensors inflate lossless ratio",
        "tensors": [],
    }

    count = 0
    for name, param in model.named_parameters():
        t = param.detach().cpu()
        native = str(t.dtype)
        upcast = False
        if t.dtype != torch.float32:
            if not args.allow_upcast:
                continue
            t = t.float()
            upcast = True

        arr = t.contiguous().numpy().astype(np.float32, copy=False)
        safe = name.replace(".", "__")
        path = out_dir / f"{safe}.f32.bin"
        arr.tofile(path)
        entry = {
            "name": name,
            "file": path.name,
            "shape": list(arr.shape),
            "dtype": "fp32",
            "nbytes": int(arr.nbytes),
            "native_dtype": native,
            "upcast": upcast,
        }
        manifest["tensors"].append(entry)
        print(f"  wrote {path.name} shape={arr.shape} upcast={upcast}")
        count += 1
        if args.max_tensors and count >= args.max_tensors:
            break

    man_path = out_dir / "manifest.json"
    man_path.write_text(json.dumps(manifest, indent=2))
    print(f"wrote {man_path} ({count} tensors)")
    if count == 0:
        print("No tensors exported. Try --allow-upcast or a different model.", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
