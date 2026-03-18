import argparse
import logging
import os
from pathlib import Path

from .bench import run_benchmark
from .models import get_model_specs
from .plotting import plot as plot_from_json
from .utils import ensure_dir, env_flag, setup_logging


def _set_env_if_not_none(key: str, value) -> None:
    if value is None:
        return
    os.environ[key] = str(value)


def main():
    p = argparse.ArgumentParser(description="EZKL GPU benchmarking (prove/verify).")
    p.add_argument("--outdir", default="results", help="Output directory for JSON + plots")
    p.add_argument("--artifacts", default="artifacts", help="Directory for ezkl artifacts (onnx/settings/keys/proofs)")
    p.add_argument("--cache", default=".cache", help="Cache dir for weights/datasets (mounted in docker-compose)")
    p.add_argument("--output", default="bench_metrics.json", help="Output JSON filename (inside --outdir)")
    p.add_argument("--repeats", type=int, default=5, help="Number of prove/verify repetitions per model")
    p.add_argument("--warmup", type=int, default=1, help="Warmup iterations (not recorded) per model")

    p.add_argument("--model", default="all", help="Which model to run. 'all' runs lenet, lenet-medium, vit, repvgg.")
    p.add_argument("--no-plot", action="store_true", help="Skip plotting step")

    p.add_argument(
        "--log-level",
        default=os.environ.get("LOG_LEVEL", "INFO"),
        help="Logging level (DEBUG, INFO, WARNING, ERROR). Can also be set via LOG_LEVEL env var.",
    )

    p.add_argument(
        "--isolate-models",
        action="store_true",
        default=env_flag("ISOLATE_MODELS", True),
        help="Run each model in a fresh subprocess (recommended in Docker). Default true if ISOLATE_MODELS is set.",
    )
    p.add_argument(
        "--fail-fast",
        action="store_true",
        default=env_flag("FAIL_FAST", False),
        help="Abort the whole benchmark on first model failure (default false).",
    )

    p.add_argument(
        "--split-onnx",
        action=argparse.BooleanOptionalAction,
        default=env_flag("SPLIT_ONNX", True),
        help="Enable ONNX model splitting for large models (recommended). Controlled by SPLIT_ONNX env var.",
    )
    p.add_argument(
        "--split-min-params-m",
        type=float,
        default=float(os.environ.get("SPLIT_MIN_PARAMS_M", "0.05")),
        help="Minimum params (in millions) per split segment, used to group small layers (default 0.05 == 50k).",
    )

    # --- Step 7: probabilistic execution benchmark knobs ---
    p.add_argument(
        "--execution-mode",
        choices=["exact", "probabilistic"],
        default=(os.environ.get("EZKL_EXECUTION_MODE") or "exact").strip().lower(),
        help="EZKL execution mode to write into settings.json (exact|probabilistic). "
        "Propagated via EZKL_EXECUTION_MODE env var so it also works in isolate/subprocess mode.",
    )
    p.add_argument(
        "--prob-k",
        type=int,
        default=int(os.environ.get("EZKL_PROB_K", "40")),
        help="Freivalds repetitions (k) for probabilistic checks. Propagated via EZKL_PROB_K env var.",
    )
    p.add_argument(
        "--prob-ops",
        default=os.environ.get("EZKL_PROB_OPS", "MatMul,Gemm,Conv"),
        help="Comma-separated ops to enable probabilistic checks for (e.g. MatMul,Gemm,Conv). "
        "Propagated via EZKL_PROB_OPS env var.",
    )
    p.add_argument(
        "--prob-seed-mode",
        default=(os.environ.get("EZKL_PROB_SEED_MODE") or "fiat_shamir").strip(),
        help="Probabilistic seed mode (e.g. fiat_shamir, public_seed). Propagated via EZKL_PROB_SEED_MODE env var.",
    )
    p.add_argument("--skip-verify", action="store_true", help="Skip verify stage.")
    p.add_argument("--skip-mock", action="store_true", help="Skip mock stage.")

    args = p.parse_args()

    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    # Propagate these knobs via env so they reliably reach worker subprocesses too.
    _set_env_if_not_none("EZKL_EXECUTION_MODE", args.execution_mode)
    _set_env_if_not_none("EZKL_PROB_K", args.prob_k)
    _set_env_if_not_none("EZKL_PROB_OPS", args.prob_ops)
    _set_env_if_not_none("EZKL_PROB_SEED_MODE", args.prob_seed_mode)

    outdir = ensure_dir(Path(args.outdir))
    artifacts_root = ensure_dir(Path(args.artifacts))
    cache_root = ensure_dir(Path(args.cache))

    logger.info(
        "Starting benchmark. model=%s repeats=%s warmup=%s outdir=%s artifacts=%s cache=%s "
        "isolate_models=%s fail_fast=%s split_onnx=%s split_min_params_m=%s "
        "execution_mode=%s prob_k=%s prob_ops=%s prob_seed_mode=%s skip_mock=%s skip_verify=%s",
        args.model,
        args.repeats,
        args.warmup,
        outdir,
        artifacts_root,
        cache_root,
        args.isolate_models,
        args.fail_fast,
        args.split_onnx,
        args.split_min_params_m,
        args.execution_mode,
        args.prob_k,
        args.prob_ops,
        args.prob_seed_mode,
        args.skip_mock,
        args.skip_verify,
    )

    specs_map = get_model_specs(cache_root)

    if args.model == "all":
        target_keys = ["lenet-5-small", "lenet-medium", "vit", "repvgg-a0"]
        specs = [specs_map[k] for k in target_keys if k in specs_map]
    elif args.model in specs_map:
        specs = [specs_map[args.model]]
    else:
        logger.error("Model '%s' not found. Available: %s", args.model, list(specs_map.keys()))
        raise SystemExit(2)

    out_json = outdir / args.output
    final = run_benchmark(
        specs=specs,
        out_json_path=out_json,
        artifacts_root=artifacts_root,
        cache_root=cache_root,
        repeats=args.repeats,
        warmup=args.warmup,
        isolate_models=args.isolate_models,
        fail_fast=args.fail_fast,
        log_level=args.log_level,
        split_onnx=args.split_onnx,
        split_min_params=int(args.split_min_params_m * 1_000_000),
        skip_verify=args.skip_verify,
        skip_mock=args.skip_mock,
    )

    logger.info("Wrote results JSON: %s", out_json)

    # If everything failed, exit non-zero so CI/CD doesn't silently "succeed".
    models = (final or {}).get("models", {}) or {}
    success = [m for m in models.values() if not (m or {}).get("error")]
    if not success:
        logger.error("All models failed (0 successful timings). Exiting with status=1.")
        if not args.no_plot:
            logger.error("Plotting skipped because there are no successful timings.")
        raise SystemExit(1)

    if not args.no_plot:
        try:
            plot_from_json(json_path=out_json, output_dir=outdir)
        except Exception as e:
            logger.exception("Plotting failed: %s", e)

    logger.info("Benchmark complete.")


if __name__ == "__main__":
    main()
