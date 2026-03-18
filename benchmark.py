#!/usr/bin/env python3
"""
Full benchmark suite orchestrator.

This file orchestrates multiple single-model EZKL benchmark runs, aggregates results, and
generates plots using `ezkl_bench.plotting`.

Notes:
- The outer CLI and output JSON remain stable for `setup-gpu.sh suite`.
- The per-case executor is `ezkl_bench`, not `bench_vit.py`.
- Plotting expects a legacy-ish JSON shape; we emit a compatible `bench_metrics.json` derived from
  our run cases and then call `ezkl_bench.plotting.plot(...)`.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import platform
import selectors
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# -----------------------------------------------------------------------------
# Defaults (suite is configurable via CLI)
# -----------------------------------------------------------------------------

DEFAULT_EXECUTION_MODE = "probabilistic"
DEFAULT_MODELS: List[str] = ["lenet-5-small", "repvgg-a0", "vit"]
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


def _indent_text(s: str, prefix: str = "    ") -> str:
    s = (s or "").rstrip()
    if not s:
        return ""
    return "\n".join(f"{prefix}{line}" for line in s.splitlines())


def _print_stream_line(log_prefix: str, stream_name: str, line: str) -> None:
    prefix = f"{log_prefix}[{stream_name}] " if log_prefix else f"[{stream_name}] "
    stream = sys.stderr if stream_name == "stderr" else sys.stdout
    print(f"{prefix}{line}", end="" if line.endswith("\n") else "\n", file=stream, flush=True)


def _run_subprocess(
    cmd: List[str],
    timeout_s: int,
    *,
    stdout_log_path: Optional[Path] = None,
    stderr_log_path: Optional[Path] = None,
    log_prefix: str = "",
    env: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    stdout_chunks: List[str] = []
    stderr_chunks: List[str] = []
    start = time.perf_counter()

    try:
        stdout_log_target = stdout_log_path if stdout_log_path is not None else Path(os.devnull)
        stderr_log_target = stderr_log_path if stderr_log_path is not None else Path(os.devnull)
        stdout_log_target.parent.mkdir(parents=True, exist_ok=True)
        stderr_log_target.parent.mkdir(parents=True, exist_ok=True)

        with open(stdout_log_target, "w", encoding="utf-8") as stdout_log, open(
            stderr_log_target, "w", encoding="utf-8"
        ) as stderr_log:
            p = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env if env is not None else dict(os.environ),
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )

            sel = selectors.DefaultSelector()
            if p.stdout is not None:
                sel.register(p.stdout, selectors.EVENT_READ, ("stdout", stdout_log, stdout_chunks))
            if p.stderr is not None:
                sel.register(p.stderr, selectors.EVENT_READ, ("stderr", stderr_log, stderr_chunks))

            while sel.get_map():
                remaining = timeout_s - (time.perf_counter() - start)
                if remaining <= 0:
                    p.kill()
                    raise subprocess.TimeoutExpired(cmd, timeout_s)

                events = sel.select(timeout=min(1.0, remaining))
                if not events:
                    continue

                for key, _ in events:
                    stream_name, log_file, chunks = key.data
                    line = key.fileobj.readline()
                    if line == "":
                        try:
                            sel.unregister(key.fileobj)
                        except Exception:
                            pass
                        key.fileobj.close()
                        continue

                    chunks.append(line)
                    log_file.write(line)
                    log_file.flush()
                    _print_stream_line(log_prefix, stream_name, line)

            returncode = p.wait(timeout=5)
            return {
                "cmd": cmd,
                "returncode": returncode,
                "stdout": "".join(stdout_chunks),
                "stderr": "".join(stderr_chunks),
            }
    except subprocess.TimeoutExpired as e:
        return {
            "cmd": cmd,
            "error": f"TimeoutExpired({timeout_s}s): {e}",
            "returncode": None,
            "stdout": "".join(stdout_chunks),
            "stderr": "".join(stderr_chunks),
        }
    except Exception as e:  # noqa: BLE001
        return {
            "cmd": cmd,
            "error": repr(e),
            "returncode": None,
            "stdout": "".join(stdout_chunks),
            "stderr": "".join(stderr_chunks),
        }


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


def _mean(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return sum(values) / float(len(values))


def _env_flag(name: str, default: bool) -> bool:
    raw = (os.environ.get(name) or "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "y", "on"}


@contextlib.contextmanager
def _temporary_env(overrides: Dict[str, Optional[str]]):
    before: Dict[str, Optional[str]] = {}
    for key, value in overrides.items():
        before[key] = os.environ.get(key)
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = str(value)
    try:
        yield
    finally:
        for key, value in before.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _split_enabled_from_env() -> bool:
    return _env_flag("SPLIT_ONNX", True)


def _split_min_params_from_env() -> int:
    raw = (os.environ.get("SPLIT_MIN_PARAMS_M") or "0.05").strip()
    try:
        value = float(raw)
    except Exception:
        value = 0.05
    return max(1, int(value * 1_000_000))


def _model_env_suffix(model_name: str) -> str:
    key = (model_name or "").strip().upper()
    return "".join(ch if ch.isalnum() else "_" for ch in key)


def _read_positive_int_env(candidates: List[str]) -> Optional[int]:
    for name in candidates:
        raw = (os.environ.get(name) or "").strip()
        if not raw:
            continue
        try:
            value = int(raw)
        except Exception:
            continue
        if value > 0:
            return value
    return None


def _resolve_case_spec_defaults(model_name: str, cache_dir: Path) -> Dict[str, Optional[int]]:
    try:
        from ezkl_bench.models import get_model_specs
    except Exception:
        return {"display_name": model_name, "input_scale": None, "param_scale": None, "num_inner_cols": None}

    specs = get_model_specs(cache_dir)
    spec = specs.get(model_name)
    if spec is None:
        return {"display_name": model_name, "input_scale": None, "param_scale": None, "num_inner_cols": None}

    suffix = _model_env_suffix(spec.key)
    input_scale = _read_positive_int_env([f"EZKL_BENCH_INPUT_SCALE_{suffix}", "EZKL_BENCH_INPUT_SCALE"])
    param_scale = _read_positive_int_env([f"EZKL_BENCH_PARAM_SCALE_{suffix}", "EZKL_BENCH_PARAM_SCALE"])
    num_inner_cols = _read_positive_int_env([f"EZKL_BENCH_NUM_INNER_COLS_{suffix}", "EZKL_BENCH_NUM_INNER_COLS"])
    return {
        "display_name": spec.display_name,
        "input_scale": int(input_scale) if input_scale is not None else int(spec.input_scale),
        "param_scale": int(param_scale) if param_scale is not None else int(spec.param_scale),
        "num_inner_cols": int(num_inner_cols) if num_inner_cols is not None else int(spec.num_inner_cols),
    }


def _segment_metric_sum(diagnostics: Dict[str, Any], key: str) -> Optional[float]:
    total = 0.0
    found = False
    for seg in diagnostics.get("segments") or []:
        if not isinstance(seg, dict):
            continue
        value = seg.get(key)
        if value is None:
            continue
        try:
            total += float(value)
            found = True
        except Exception:
            continue
    return total if found else None


def _segment_path(diagnostics: Dict[str, Any], key: str) -> str:
    for seg in diagnostics.get("segments") or []:
        if not isinstance(seg, dict):
            continue
        value = seg.get(key)
        if isinstance(value, dict) and value.get("path"):
            return str(value["path"])
    return ""


def _segment_work_dir(diagnostics: Dict[str, Any]) -> str:
    for seg in diagnostics.get("segments") or []:
        if isinstance(seg, dict) and seg.get("work_dir"):
            return str(seg["work_dir"])
    return ""


def _segment_int_sum(diagnostics: Dict[str, Any], key: str) -> Optional[int]:
    total = 0
    found = False
    for seg in diagnostics.get("segments") or []:
        if not isinstance(seg, dict):
            continue
        value = seg.get(key)
        if value is None:
            continue
        try:
            total += int(value)
            found = True
        except Exception:
            continue
    return total if found else None


def _case_env_overrides(*, prob_k: int, prob_ops: List[str], prob_seed_mode: str) -> Dict[str, Optional[str]]:
    return {
        "EZKL_EXECUTION_MODE": DEFAULT_EXECUTION_MODE,
        "EZKL_PROB_K": str(int(prob_k)),
        "EZKL_PROB_OPS": ",".join(prob_ops),
        "EZKL_PROB_SEED_MODE": str(prob_seed_mode),
    }


def _adapt_model_result_to_legacy_metrics(
    *,
    model_name: str,
    prob_k: int,
    prob_ops: List[str],
    prob_seed_mode: str,
    payload: Dict[str, Any],
    spec_defaults: Dict[str, Optional[int]],
) -> Dict[str, Any]:
    diagnostics = payload.get("diagnostics") or {}
    prove_times = [float(x) for x in (payload.get("prove_times_s") or [])]
    verify_times = [float(x) for x in (payload.get("verify_times_s") or [])]

    timings: Dict[str, float] = {}
    for src_key, dst_key in (
        ("setup_time_s", "setup_s"),
        ("compile_time_s", "compile_circuit_s"),
        ("calibrate_time_s", "calibrate_settings_s"),
    ):
        value = payload.get(src_key)
        if value is None:
            continue
        try:
            timings[dst_key] = float(value)
        except Exception:
            continue

    for seg_key, dst_key in (
        ("gen_settings_time_s", "gen_settings_s"),
        ("get_srs_time_s", "get_srs_s"),
        ("gen_witness_time_s", "gen_witness_s"),
        ("mock_time_s", "mock_s"),
    ):
        total = _segment_metric_sum(diagnostics, seg_key)
        if total is not None:
            timings[dst_key] = float(total)

    prove_mean = _mean(prove_times)
    if prove_mean is not None:
        timings["prove_s"] = float(prove_mean)
    verify_mean = _mean(verify_times)
    if verify_mean is not None:
        timings["verify_s"] = float(verify_mean)

    return {
        "model_name": model_name,
        "display_name": payload.get("display_name") or spec_defaults.get("display_name") or model_name,
        "execution_mode": DEFAULT_EXECUTION_MODE,
        "prob_k": int(prob_k),
        "prob_ops": list(prob_ops),
        "prob_seed_mode": str(prob_seed_mode),
        "input_scale": spec_defaults.get("input_scale"),
        "param_scale": spec_defaults.get("param_scale"),
        "num_inner_cols": spec_defaults.get("num_inner_cols"),
        "logrows": payload.get("logrows"),
        "constraint_count": _segment_int_sum(diagnostics, "total_assignments"),
        "timings_s": timings,
        "prove_times_s": prove_times,
        "verify_times_s": verify_times,
        "settings_path": _segment_path(diagnostics, "settings"),
        "compiled_path": _segment_path(diagnostics, "compiled"),
        "witness_path": _segment_path(diagnostics, "witness"),
        "proof_path": _segment_path(diagnostics, "proof"),
        "work_dir": _segment_work_dir(diagnostics),
        "slug": payload.get("slug"),
        "diagnostics": diagnostics,
    }


def _write_legacy_case_report(
    *,
    report_path: Path,
    metrics: Dict[str, Any],
    payload: Dict[str, Any],
) -> None:
    _write_json(
        report_path,
        {
            "backend": "ezkl_bench",
            "metrics": metrics,
            "model_result": payload,
        },
    )


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
    run_dir = _ensure_dir(out_dir)
    artifacts_dir = _ensure_dir(run_dir / "artifacts")
    effective_cache_dir = _ensure_dir(cache_dir if cache_dir is not None else run_dir / ".cache")
    result_path = run_dir / "ezkl_bench_model_result.json"
    report_path = run_dir / "vit_bench_report.json"
    stdout_log_path = run_dir / "bench_stdout.log"
    stderr_log_path = run_dir / "bench_stderr.log"
    case_tag = f"[{model_name}/k{int(prob_k)}/run{int(run_index)}]"
    case_env = dict(os.environ)
    case_env.update(
        {
            k: v
            for k, v in _case_env_overrides(prob_k=int(prob_k), prob_ops=prob_ops, prob_seed_mode=prob_seed_mode).items()
            if v is not None
        }
    )

    cmd = [
        sys.executable,
        "-m",
        "ezkl_bench.worker",
        "--model",
        str(model_name),
        "--artifacts",
        str(artifacts_dir),
        "--cache",
        str(effective_cache_dir),
        "--out",
        str(result_path),
        "--repeats",
        "1",
        "--warmup",
        "0",
        "--log-level",
        os.environ.get("LOG_LEVEL", "INFO"),
        "--split-onnx" if _split_enabled_from_env() else "--no-split-onnx",
        "--split-min-params",
        str(_split_min_params_from_env()),
    ]
    if skip_verify:
        cmd.append("--skip-verify")
    if skip_mock:
        cmd.append("--skip-mock")

    print(f"  Bench command: {' '.join(cmd)}")
    print(f"  Live logs: stdout={stdout_log_path} stderr={stderr_log_path}")

    t0 = time.perf_counter()
    run_started_at = time.time()
    res = _run_subprocess(
        cmd,
        timeout_s=timeout_s,
        stdout_log_path=stdout_log_path,
        stderr_log_path=stderr_log_path,
        log_prefix=case_tag,
        env=case_env,
    )
    elapsed_s = time.perf_counter() - t0

    spec_defaults = _resolve_case_spec_defaults(model_name=str(model_name), cache_dir=effective_cache_dir)
    payload: Optional[Dict[str, Any]] = None
    metrics: Optional[Dict[str, Any]] = None
    metrics_err: Optional[str] = None
    result_is_fresh = False
    if result_path.exists():
        try:
            result_is_fresh = float(result_path.stat().st_mtime) >= (run_started_at - 1.0)
        except Exception:
            result_is_fresh = False
    if result_is_fresh:
        try:
            payload = _read_json(result_path)
            metrics = _adapt_model_result_to_legacy_metrics(
                model_name=str(model_name),
                prob_k=int(prob_k),
                prob_ops=prob_ops,
                prob_seed_mode=str(prob_seed_mode),
                payload=payload,
                spec_defaults=spec_defaults,
            )
            _write_legacy_case_report(report_path=report_path, metrics=metrics, payload=payload)
        except Exception as e:  # noqa: BLE001
            metrics_err = f"failed reading {result_path}: {e!r}"
    elif result_path.exists():
        metrics_err = f"ignoring stale result file at {result_path}"

    model_error = payload.get("error") if isinstance(payload, dict) else None
    model_traceback = payload.get("traceback") if isinstance(payload, dict) else None
    ok = (res.get("returncode") == 0) and (metrics is not None) and not model_error

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
        "model_result_path": str(result_path),
        "model_result_exists": result_path.exists(),
        "stdout_log": str(stdout_log_path),
        "stderr_log": str(stderr_log_path),
        "metrics": metrics,
        "metrics_error": metrics_err,
        "error": model_error,
        "traceback": model_traceback,
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
    run_dir = _ensure_dir(out_dir)
    artifacts_dir = _ensure_dir(run_dir / "artifacts")
    effective_cache_dir = _ensure_dir(cache_dir if cache_dir is not None else run_dir / ".cache")
    result_path = run_dir / "ezkl_bench_model_result.json"
    report_path = run_dir / "vit_bench_report.json"
    try:
        from dataclasses import asdict
        from ezkl_bench.bench import run_single_model  # type: ignore
        from ezkl_bench.models import get_model_specs  # type: ignore

        specs = get_model_specs(effective_cache_dir)
        spec = specs[str(model_name)]
        case_env = _case_env_overrides(prob_k=int(prob_k), prob_ops=prob_ops, prob_seed_mode=prob_seed_mode)
        with _temporary_env(case_env):
            result_obj = run_single_model(
                spec=spec,
                artifacts_root=artifacts_dir,
                cache_root=effective_cache_dir,
                repeats=1,
                warmup=0,
                split_onnx=_split_enabled_from_env(),
                split_min_params=_split_min_params_from_env(),
                skip_verify=bool(skip_verify),
                skip_mock=bool(skip_mock),
            )

        payload = asdict(result_obj)
        _write_json(result_path, payload)
        spec_defaults = _resolve_case_spec_defaults(model_name=str(model_name), cache_dir=effective_cache_dir)
        metrics = _adapt_model_result_to_legacy_metrics(
            model_name=str(model_name),
            prob_k=int(prob_k),
            prob_ops=prob_ops,
            prob_seed_mode=str(prob_seed_mode),
            payload=payload,
            spec_defaults=spec_defaults,
        )
        _write_legacy_case_report(report_path=report_path, metrics=metrics, payload=payload)

        elapsed_s = time.perf_counter() - t0
        return {
            "ok": not payload.get("error"),
            "model_name": model_name,
            "prob_k": int(prob_k),
            "run_index": int(run_index),
            "elapsed_s": float(elapsed_s),
            "run_dir": str(run_dir),
            "metrics": metrics,
            "error": payload.get("error"),
            "traceback": payload.get("traceback"),
            "bench_report_path": str(report_path),
            "bench_report_exists": report_path.exists(),
            "model_result_path": str(result_path),
            "model_result_exists": result_path.exists(),
        }
    except Exception as e:  # noqa: BLE001
        elapsed_s = time.perf_counter() - t0
        tb = traceback.format_exc()

        diag_path = run_dir / "inprocess_error.txt"
        try:
            _ensure_dir(run_dir)
            diag_path.write_text(tb, encoding="utf-8")
        except Exception:
            # Best-effort only; do not mask the original exception.
            pass

        return {
            "ok": False,
            "model_name": model_name,
            "prob_k": int(prob_k),
            "run_index": int(run_index),
            "elapsed_s": float(elapsed_s),
            "run_dir": str(run_dir),
            "error": repr(e),
            "traceback": tb,
            "diagnostic": str(diag_path),
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


def _format_metric_s(metrics: Dict[str, Any], key: str) -> str:
    val = _extract_timing_s(metrics, key)
    if val is None:
        return "n/a"
    return f"{val:.2f}s"


def _print_case_result(case: Dict[str, Any]) -> None:
    if case.get("ok") and isinstance(case.get("metrics"), dict):
        metrics = case["metrics"]
        print(
            "  Summary: "
            f"logrows={metrics.get('logrows')}, "
            f"setup={_format_metric_s(metrics, 'setup_s')}, "
            f"prove={_format_metric_s(metrics, 'prove_s')}, "
            f"verify={_format_metric_s(metrics, 'verify_s')}"
        )
        return

    print("  Failure details:")
    if case.get("subprocess_error"):
        print(_indent_text(f"subprocess_error: {case['subprocess_error']}"))
    if case.get("error"):
        print(_indent_text(f"error: {case['error']}"))
    if case.get("metrics_error"):
        print(_indent_text(f"metrics_error: {case['metrics_error']}"))
    if case.get("diagnostic"):
        print(_indent_text(f"diagnostic: {case['diagnostic']}"))
    if case.get("stdout_log"):
        print(_indent_text(f"stdout_log: {case['stdout_log']}"))
    if case.get("stderr_log"):
        print(_indent_text(f"stderr_log: {case['stderr_log']}"))
    if case.get("stdout_tail"):
        print("    stdout tail:")
        print(_indent_text(str(case["stdout_tail"]), prefix="      "))
    if case.get("stderr_tail"):
        print("    stderr tail:")
        print(_indent_text(str(case["stderr_tail"]), prefix="      "))
    if case.get("traceback"):
        print("    traceback:")
        print(_indent_text(str(case["traceback"]), prefix="      "))


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

    total_cases = len(models) * len(prob_k_values) * runs
    case_counter = 0

    print(f"\n{'='*80}")
    print(f"Starting benchmark suite:")
    print(f"  Models: {models}")
    print(f"  prob_k values: {prob_k_values}")
    print(f"  Runs per case: {runs}")
    print(f"  Total cases: {total_cases}")
    print(f"  Isolation: {isolate}")
    print(f"{'='*80}\n")

    for model_name in models:
        for prob_k in prob_k_values:
            for run_index in range(runs):
                case_counter += 1
                run_dir = outdir / "runs" / str(model_name) / f"k{int(prob_k)}" / f"run{int(run_index)}"
                _ensure_dir(run_dir)

                print(f"\n[{case_counter}/{total_cases}] Running: model={model_name}, prob_k={prob_k}, run={run_index}")
                print(f"  Output dir: {run_dir}")
                case_start = time.perf_counter()

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
                case_elapsed = time.perf_counter() - case_start

                status = "✓ OK" if case.get("ok") else "✗ FAILED"
                print(f"  {status} - Elapsed: {case_elapsed:.1f}s")
                _print_case_result(case)

                # Calculate and show estimated time remaining
                if case_counter < total_cases:
                    avg_time_per_case = (time.perf_counter() - suite_t0) / case_counter
                    remaining_cases = total_cases - case_counter
                    eta_seconds = avg_time_per_case * remaining_cases
                    eta_minutes = eta_seconds / 60
                    print(f"  ETA for remaining {remaining_cases} cases: ~{eta_minutes:.1f} minutes")

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
