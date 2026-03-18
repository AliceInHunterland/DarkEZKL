import argparse
import logging
from pathlib import Path

from .bench import run_single_model
from .models import get_model_specs
from .utils import ensure_dir, setup_logging, write_json


def main():
    p = argparse.ArgumentParser(description="Worker process: run a single model benchmark and write result JSON.")
    # Removed strict choices=["..."] to allow dynamic model keys (fixes 'nano' crash and allows 'lenet-medium')
    p.add_argument("--model", required=True, help="Model key (e.g. lenet, lenet-medium, vit, repvgg)")
    p.add_argument("--artifacts", default="artifacts")
    p.add_argument("--cache", default=".cache")
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--out", required=True, help="Path to write model_result.json")
    p.add_argument("--log-level", default="INFO")

    # ONNX splitting controls
    p.add_argument("--split-onnx", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--split-min-params", type=int, default=50_000)
    p.add_argument("--skip-verify", action="store_true")
    p.add_argument("--skip-mock", action="store_true")

    args = p.parse_args()

    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    artifacts_root = ensure_dir(Path(args.artifacts))
    cache_root = ensure_dir(Path(args.cache))
    out_path = Path(args.out)
    ensure_dir(out_path.parent)

    specs = get_model_specs(cache_root)
    if args.model not in specs:
        logger.error(f"Invalid model key: {args.model}. Available: {list(specs.keys())}")
        exit(1)

    spec = specs[args.model]

    logger.info(
        "Worker starting. model=%s repeats=%s warmup=%s artifacts=%s cache=%s out=%s split_onnx=%s split_min_params=%s",
        args.model,
        args.repeats,
        args.warmup,
        artifacts_root,
        cache_root,
        out_path,
        args.split_onnx,
        args.split_min_params,
    )

    r = run_single_model(
        spec=spec,
        artifacts_root=artifacts_root,
        cache_root=cache_root,
        repeats=args.repeats,
        warmup=args.warmup,
        split_onnx=args.split_onnx,
        split_min_params=args.split_min_params,
        skip_verify=args.skip_verify,
        skip_mock=args.skip_mock,
    )

    write_json(out_path, r.__dict__)
    logger.info("Worker wrote result: %s", out_path)


if __name__ == "__main__":
    main()
