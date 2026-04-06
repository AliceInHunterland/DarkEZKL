#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ezkl_bench.vision_adapter_plotting import plot_vision_adapter_summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot full-circuit vs vision-adapter benchmark figures.")
    parser.add_argument("--summary", default="results/vision_adapter_bench/vision_adapter_summary.json")
    parser.add_argument("--out-dir", default="results/vision_adapter_bench/plots")
    args = parser.parse_args()

    plot_vision_adapter_summary(summary_path=Path(args.summary), output_dir=Path(args.out_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
