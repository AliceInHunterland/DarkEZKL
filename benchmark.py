#!/usr/bin/env python3
"""
Full benchmark suite orchestrator.

This file orchestrates multiple runs of `bench_vit.py`, aggregates results, and generates plots
using `ezkl_bench.plotting`.

Notes:
- `bench_vit.py` is the single-run executor. This file only orchestrates runs, aggregates results,
  and then produces plots.
- Plotting expects a legacy-ish JSON shape; we emit a compatible `bench_metrics.json` derived from
  our run cases and then call `ezkl_bench.plotting.plot(...)`.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# -----------------------------------------------------------------------------
# Defaults (suite is configurable via CLI)
# -----------------------------------------------------------------------------

DEFAULT_EXECUTION_MODE = "probabilistic"
DEFAULT_MODELS: List[str] = ["vit", "lenet-5-small", "repvgg-a0"]
DEFAULT_PROB_K_VALUES: List[int] = [2, 4]
DEFAULT_RUNS_PER_CASE: int = 3


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------

def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def _read_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _tail_text(s: str, max_lines: int = 200, max_chars: int = 20000) -> str:
    s = s or ""
    if len(s) > max_chars:
        s = s[-max_chars:]
    lines = s.splitlines()
    if len(lines) > max_lines:
        lines = lines[-max_lines:]
    return "\n".join(lines).strip()


def _run_subprocess(cmd: List[str], timeout_s: int) -> Dict[str, Any]:
    try:
        p = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_s,
            check=False,
        )
        return {
            "cmd": cmd,
            "returncode": p.returncode,
            "stdout": p.stdout or "",
            "stderr": p.stderr or "",
        }
    except Exception as e:  # noqa: BLE001
        return {"cmd": cmd, "error": repr(e), "returncode": None, "stdout": "", "stderr": ""}


def _parse_csv_list(s: str) -> List[str]:
    return [x.strip() for x in (s or "").split(",") if x.strip()]


def _parse_csv_ints(s: str) -> List[int]:
    out: List[int] = []
    for tok in _parse_csv_list(s):
        out.append(int(tok))
    return out


def _mean_std(values: List[float]) -> Tuple[Optional[float], Optional[float]]:
    if not values:
        return None, None
    mu = sum(values) / float(len(values))
    if len(values) < 2:
        return mu, 0.0
    var = sum((x - mu) ** 2 for x in values) / float(len(values) - 1)
    return mu, var**0.5


def _collect_aggregates(cases: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregate per (model, prob_k) over successful runs.
    """
    buckets: Dict[Tuple[str, int], List[Dict[str, Any]]] = {}
    for c in cases:
        key = (str(c.get("model_name")), int(c.get("prob_k")))
        buckets.setdefault(key, []).append(c)

    agg: Dict[str, Any] = {}
    for (model, k), runs in sorted(buckets.items(), key=lambda x: (x[0][0], x[0][1])):
        ok_runs = [r for r in runs if r.get("ok") is True and isinstance(r.get("metrics"), dict)]
        timings_by_key: Dict[str, List[float]] = {}

        for r in ok_runs:
            metrics = r["metrics"]
            timings = metrics.get("timings_s") or {}
            if isinstance(timings, dict):
                for tkey, tval in timings.items():
                    try:
                        timings_by_key.setdefault(str(tkey), []).append(float(tval))
                    except Exception:
                        continue

        timings_stats: Dict[str, Any] = {}
        for tkey, vals in sorted(timings_by_key.items()):
            mu, sd = _mean_std(vals)
            timings_stats[tkey] = {
                "n": len(vals),
                "mean_s": mu,
                "std_s": sd,
                "min_s": min(vals) if vals else None,
                "max_s": max(vals) if vals else None,
            }

        agg[f"{model}/k{k}"] = {
            "model_name": model,
            "prob_k": k,
            "runs_total": len(runs),
            "runs_ok": len(ok_runs),
            "timings_s": timings_stats,
        }

    return agg


def _gather_env_report() -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "timestamp_unix": time.time(),
        "python": {"version": sys.version, "executable": sys.executable},
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "imports": {},
        "versions": {},
        "cuda": {},
        "cli": {},
    }

    # Imports / versions
    torch = None
    ort = None
    ezkl = None

    try:
        import torch as _torch

        torch = _torch
        report["imports"]["torch"] = True
        report["versions"]["torch"] = getattr(torch, "__version__", "unknown")
    except Exception as e:  # noqa: BLE001
        report["imports"]["torch"] = False
        report["imports"]["torch_error"] = repr(e)

    try:
        import onnxruntime as _ort

        ort = _ort
        report["imports"]["onnxruntime"] = True
        report["versions"]["onnxruntime"] = getattr(ort, "__version__", "unknown")
    except Exception as e:  # noqa: BLE001
        report["imports"]["onnxruntime"] = False
        report["imports"]["onnxruntime_error"] = repr(e)

    try:
        import ezkl as _ezkl  # type: ignore

        ezkl = _ezkl
        report["imports"]["ezkl"] = True
        report["versions"]["ezkl"] = getattr(ezkl, "__version__", "unknown")
    except Exception as e:  # noqa: BLE001
        report["imports"]["ezkl"] = False
        report["imports"]["ezkl_error"] = repr(e)

    # CUDA info (if torch is present)
    if torch is not None:
        try:
            cuda_available = bool(torch.cuda.is_available())
            report["cuda"]["available"] = cuda_available
            report["cuda"]["device_count"] = int(torch.cuda.device_count()) if cuda_available else 0
            if cuda_available and torch.cuda.device_count() > 0:
                idx = int(torch.cuda.current_device())
                report["cuda"]["current_device"] = idx
                report["cuda"]["device_name"] = torch.cuda.get_device_name(idx)
        except Exception as e:  # noqa: BLE001
            report["cuda"]["error"] = repr(e)

    # CLI check (best-effort)
    report["cli"]["ezkl_version"] = _run_subprocess(
        ["bash", "-lc", "command -v ezkl && ezkl --version"],
        timeout_s=30,
    )

    # NOTE: keep this script "import safe" for tests; no heavy work here.
    _ = ort
    _ = ezkl
    return report


def _run_one_case_subprocess(
    *,
    model_name: str,
    prob_k: int,
    run_index: int,
    out_dir: Path,
    cache_dir: Optional[Path],
    prob_ops: List[str],
    prob_seed_mode: str,
    skip_verify: bool,
    skip_mock: bool,
    timeout_s: int,
    tail_lines: int,
) -> Dict[str, Any]:
    bench_vit_path = Path(__file__).with_name("bench_vit.py")
    run_dir = _ensure_dir(out_dir)

    cmd = [
        sys.executable,
        str(bench_vit_path),
        "--outdir",
        str(run_dir),
        "--model-name",
        str(model_name),
        "--prob-k",
        str(int(prob_k)),
        "--prob-ops",
        ",".join(prob_ops),
        "--prob-seed-mode",
        str(prob_seed_mode),
        # legacy args accepted by bench_vit (ignored) but keep stable CLI surface
        "--repeats",
        "1",
        "--warmup",
        "0",
    ]
    if cache_dir is not None:
        cmd += ["--cache-dir", str(cache_dir)]
    if skip_verify:
        cmd += ["--skip-verify"]
    if skip_mock:
        cmd += ["--skip-mock"]

    t0 = time.perf_counter()
    res = _run_subprocess(cmd, timeout_s=timeout_s)
    elapsed_s = time.perf_counter() - t0

    report_path = run_dir / "vit_bench_report.json"
    metrics: Optional[Dict[str, Any]] = None
    metrics_err: Optional[str] = None
    if report_path.exists():
        try:
            payload = _read_json(report_path)
            metrics = payload.get("metrics")
        except Exception as e:  # noqa: BLE001
            metrics_err = f"failed reading {report_path}: {e!r}"

    ok = (res.get("returncode") == 0) and (metrics is not None)

    return {
        "ok": bool(ok),
        "model_name": model_name,
        "prob_k": int(prob_k),
        "run_index": int(run_index),
        "elapsed_s": float(elapsed_s),
        "run_dir": str(run_dir),
        "bench_cmd": res.get("cmd"),
        "returncode": res.get("returncode"),
        "stdout_tail": _tail_text(str(res.get("stdout") or ""), max_lines=tail_lines),
        "stderr_tail": _tail_text(str(res.get("stderr") or ""), max_lines=tail_lines),
        "bench_report_path": str(report_path),
        "bench_report_exists": report_path.exists(),
        "metrics": metrics,
        "metrics_error": metrics_err,
        "subprocess_error": res.get("error"),
    }


def _run_one_case_inprocess(
    *,
    model_name: str,
    prob_k: int,
    run_index: int,
    out_dir: Path,
    cache_dir: Optional[Path],
    prob_ops: List[str],
    prob_seed_mode: str,
    skip_verify: bool,
    skip_mock: bool,
) -> Dict[str, Any]:
    t0 = time.perf_counter()
    try:
        # This directly exercises the updated probabilistic PyRunArgs path in bench_vit.
        from bench_vit import run_single_benchmark  # type: ignore

        metrics_obj = run_single_benchmark(
            model_name=str(model_name),
            prob_k=int(prob_k),
            out_dir=_ensure_dir(out_dir),
            prob_ops=list(prob_ops),
            prob_seed_mode=str(prob_seed_mode),
            cache_dir=cache_dir,
            skip_verify=bool(skip_verify),
            skip_mock=bool(skip_mock),
        )

        # bench_vit writes vit_bench_report.json too; but for aggregation we serialize the object we got.
        try:
            from dataclasses import asdict

            metrics = asdict(metrics_obj)
        except Exception:
            metrics = getattr(metrics_obj, "__dict__", {})  # fallback

        elapsed_s = time.perf_counter() - t0
        return {
            "ok": True,
            "model_name": model_name,
            "prob_k": int(prob_k),
            "run_index": int(run_index),
            "elapsed_s": float(elapsed_s),
            "run_dir": str(out_dir),
            "metrics": metrics,
        }
    except Exception as e:  # noqa: BLE001
        elapsed_s = time.perf_counter() - t0
        return {
            "ok": False,
            "model_name": model_name,
            "prob_k": int(prob_k),
            "run_index": int(run_index),
            "elapsed_s": float(elapsed_s),
            "run_dir": str(out_dir),
            "error": repr(e),
        }


def _extract_timing_s(metrics: Dict[str, Any], key: str) -> Optional[float]:
    timings = metrics.get("timings_s")
    if not isinstance(timings, dict):
        return None
    v = timings.get(key)
    if v is None:
        return None
    try:
        return float(v)
    except Exception:
        return None


def _make_plot_payload_from_cases(
    *,
    cases: List[Dict[str, Any]],
    models: List[str],
    prob_k_values: List[int],
) -> Dict[str, Any]:
    """
    Produce a JSON payload compatible with `ezkl_bench.plotting.plot`.

    plotting.py expects:
      {"models": { "<label>": {"logrows": <int>, "prove_times_s": [...], "verify_times_s": [...] } } }

    We map each (model,k) to a distinct entry key like: "vit/k2" and store `logrows = k`
    so the plot labels show (k=2) etc., matching the benchmark comparison.
    """
    by_key: Dict[str, Dict[str, Any]] = {}

    # Initialize buckets in the provided order
    for m in models:
        for k in prob_k_values:
            key = f"{m}/k{k}"
            by_key[key] = {
                "logrows": int(k),  # repurposed for plot label "(k=...)"
                "prove_times_s": [],
                "verify_times_s": [],
            }

    # Fill buckets
    for c in cases:
        if not c.get("ok"):
            continue
        m = str(c.get("model_name"))
        k = int(c.get("prob_k"))
        metrics = c.get("metrics")
        if not isinstance(metrics, dict):
            continue

        bucket_key = f"{m}/k{k}"
        if bucket_key not in by_key:
            # If inputs change mid-run or cases contain unexpected combos, keep resilient.
            by_key[bucket_key] = {"logrows": int(k), "prove_times_s": [], "verify_times_s": []}

        prove_s = _extract_timing_s(metrics, "prove_s")
        verify_s = _extract_timing_s(metrics, "verify_s")
        if prove_s is not None:
            by_key[bucket_key]["prove_times_s"].append(float(prove_s))
        if verify_s is not None:
            by_key[bucket_key]["verify_times_s"].append(float(verify_s))

    # Mark errors for buckets with no data (plotting skips errored entries)
    for key, bucket in list(by_key.items()):
        if not bucket["prove_times_s"] and not bucket["verify_times_s"]:
            bucket["error"] = "no successful timing data collected"

    return {"models": by_key}


def _generate_plots(*, plot_json_path: Path, plots_outdir: Path) -> Dict[str, Any]:
    """
    Generates plots using ezkl_bench.plotting. Returns a small report.
    """
    # Make matplotlib safe in headless environments
    os.environ.setdefault("MPLBACKEND", "Agg")

    try:
        from ezkl_bench import plotting as ezkl_plotting  # type: ignore

        ezkl_plotting.plot(json_path=plot_json_path, output_dir=plots_outdir)
        return {
            "ok": True,
            "plot_json": str(plot_json_path),
            "plots_outdir": str(plots_outdir),
        }
    except Exception as e:  # noqa: BLE001
        # As a fallback, write a diagnostic file so the run is debuggable.
        _ensure_dir(plots_outdir)
        diag = plots_outdir / "plot_error.txt"
        try:
            diag.write_text(f"Plot generation failed: {e!r}\n", encoding="utf-8")
        except Exception:
            pass
        return {
            "ok": False,
            "plot_json": str(plot_json_path),
            "plots_outdir": str(plots_outdir),
            "error": repr(e),
            "diagnostic": str(diag),
        }


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="EZKL probabilistic benchmark suite orchestrator")

    ap.add_argument("--outdir", default="/app/results", help="Directory to write results JSON + per-run artifacts")

    # Suite configuration (no canonicalization; order preserved as provided)
    ap.add_argument(
        "--models",
        default=",".join(DEFAULT_MODELS),
        help="Comma-separated model names (order preserved)",
    )
    ap.add_argument(
        "--prob-k-values",
        default=",".join(map(str, DEFAULT_PROB_K_VALUES)),
        help="Comma-separated prob_k values (order preserved)",
    )
    ap.add_argument(
        "--runs",
        type=int,
        default=DEFAULT_RUNS_PER_CASE,
        help="Runs per (model, prob_k)",
    )

    # Backwards-compat: treat --repeats as alias for --runs
    ap.add_argument("--repeats", type=int, default=None, help="Alias for --runs")

    # Execution controls
    ap.add_argument("--cache-dir", default="", help="Cache dir for weights/datasets (vit/repvgg/lenet models).")
    ap.add_argument("--prob-ops", default="MatMul,Gemm,Conv", help="Comma-separated ops to run probabilistically")
    ap.add_argument("--prob-seed-mode", default="fiat_shamir", help="Seed mode (e.g. fiat_shamir, public_seed)")

    ap.add_argument("--skip-verify", action="store_true", help="Skip verify (not recommended)")
    ap.add_argument("--skip-mock", action="store_true", help="Skip mock (not recommended)")

    ap.add_argument(
        "--isolate-models",
        action="store_true",
        help="Force subprocess-per-run execution (overrides ISOLATE_MODELS env)",
    )
    ap.add_argument(
        "--no-isolate-models",
        action="store_true",
        help="Force in-process execution (overrides ISOLATE_MODELS env)",
    )

    ap.add_argument(
        "--timeout-s",
        type=int,
        default=int(os.environ.get("MODEL_RUN_TIMEOUT_S", "7200")),
        help="Timeout per run when using subprocess isolation",
    )

    args = ap.parse_args()

    outdir = _ensure_dir(Path(args.outdir))

    # Parse suite (no validation/canonicalization)
    models = _parse_csv_list(args.models)
    if not models:
        raise SystemExit("--models must contain at least one model name")

    prob_k_values = _parse_csv_ints(args.prob_k_values)
    if not prob_k_values:
        raise SystemExit("--prob-k-values must contain at least one integer")

    runs = int(args.repeats) if args.repeats is not None else int(args.runs)
    if runs <= 0:
        raise SystemExit("--runs/--repeats must be a positive integer")

    cache_dir = Path(args.cache_dir) if str(args.cache_dir).strip() else None
    prob_ops = _parse_csv_list(args.prob_ops) or ["MatMul", "Gemm", "Conv"]
    prob_seed_mode = str(args.prob_seed_mode).strip() or "fiat_shamir"

    env_report = _gather_env_report()

    # Decide isolation behavior
    env_iso = str(os.environ.get("ISOLATE_MODELS", "false")).strip().lower() in {"1", "true", "yes", "y"}
    isolate = env_iso
    if args.isolate_models:
        isolate = True
    if args.no_isolate_models:
        isolate = False

    tail_lines = int(os.environ.get("MODEL_PROCESS_TAIL_LINES", "250"))

    suite_cfg = {
        "execution_mode": DEFAULT_EXECUTION_MODE,
        "models": models,
        "prob_k_values": prob_k_values,
        "runs_per_case": runs,
        "prob_ops": prob_ops,
        "prob_seed_mode": prob_seed_mode,
        "isolate_models": isolate,
        "skip_verify": bool(args.skip_verify),
        "skip_mock": bool(args.skip_mock),
    }

    cases: List[Dict[str, Any]] = []
    suite_t0 = time.perf_counter()

    for model_name in models:
        for prob_k in prob_k_values:
            for run_index in range(runs):
                run_dir = outdir / "runs" / str(model_name) / f"k{int(prob_k)}" / f"run{int(run_index)}"
                _ensure_dir(run_dir)

                if isolate:
                    case = _run_one_case_subprocess(
                        model_name=str(model_name),
                        prob_k=int(prob_k),
                        run_index=int(run_index),
                        out_dir=run_dir,
                        cache_dir=cache_dir,
                        prob_ops=prob_ops,
                        prob_seed_mode=prob_seed_mode,
                        skip_verify=bool(args.skip_verify),
                        skip_mock=bool(args.skip_mock),
                        timeout_s=int(args.timeout_s),
                        tail_lines=tail_lines,
                    )
                else:
                    case = _run_one_case_inprocess(
                        model_name=str(model_name),
                        prob_k=int(prob_k),
                        run_index=int(run_index),
                        out_dir=run_dir,
                        cache_dir=cache_dir,
                        prob_ops=prob_ops,
                        prob_seed_mode=prob_seed_mode,
                        skip_verify=bool(args.skip_verify),
                        skip_mock=bool(args.skip_mock),
                    )

                cases.append(case)

                # Persist incremental progress so partial failures still leave artifacts.
                _write_json(outdir / "benchmark_partial.json", {"suite": suite_cfg, "env": env_report, "cases": cases})

    suite_elapsed_s = time.perf_counter() - suite_t0
    aggregates = _collect_aggregates(cases)

    final_report: Dict[str, Any] = {
        "suite": suite_cfg,
        "env": env_report,
        "suite_elapsed_s": float(suite_elapsed_s),
        "cases": cases,
        "aggregates": aggregates,
        "ok": all(bool(c.get("ok")) for c in cases) if cases else False,
    }

    # Primary outputs
    _write_json(outdir / "benchmark.json", final_report)
    _write_json(outdir / "benchmark_suite.json", final_report)

    # Plot generation
    plot_json_path = outdir / "bench_metrics.json"
    plots_outdir = outdir / "plots"
    plot_payload = _make_plot_payload_from_cases(cases=cases, models=models, prob_k_values=prob_k_values)
    _write_json(plot_json_path, plot_payload)

    plot_report = _generate_plots(plot_json_path=plot_json_path, plots_outdir=plots_outdir)
    final_report["plots"] = plot_report

    # Re-write final reports including plot status
    _write_json(outdir / "benchmark.json", final_report)
    _write_json(outdir / "benchmark_suite.json", final_report)

    print(json.dumps(final_report, indent=2, sort_keys=True))

    # Exit codes:
    # 2: ezkl import broken (environment invalid)
    # 3: any benchmark case failed
    # 4: benchmarks OK but plot generation failed
    if not env_report.get("imports", {}).get("ezkl", False):
        return 2
    if not final_report["ok"]:
        return 3
    if not final_report.get("plots", {}).get("ok", False):
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
