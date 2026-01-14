"""
Simple "test file" runner:

- By default runs the curated benchmark set.
- You can run a single model via --model.

This is intentionally a lightweight wrapper around benchmark.py so you can
quickly validate the end-to-end pipeline.
"""

import argparse
import subprocess
import sys


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="all", choices=["all", "lenet", "lenet-medium", "vit", "repvgg", "nano"])
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--outdir", default="results")
    p.add_argument("--split-onnx", action=argparse.BooleanOptionalAction, default=True)
    args = p.parse_args()

    cmd = [
        sys.executable,
        "benchmark.py",
        "--model",
        args.model,
        "--repeats",
        str(args.repeats),
        "--outdir",
        args.outdir,
        "--split-onnx" if args.split_onnx else "--no-split-onnx",
    ]
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
