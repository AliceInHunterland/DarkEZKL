from __future__ import annotations

import logging
import os
import platform
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

from .bench_model import ModelRunResult, run_single_model
from .models import ModelSpec
from .utils import ensure_dir, get_system_stats, run_command, run_subprocess_streaming, slugify, write_json

logger = logging.getLogger(__name__)


def _run_single_model_in_worker(
    spec: ModelSpec,
    artifacts_root: Path,
    cache_root: Path,
    repeats: int,
    warmup: int,
    log_level: str,
    split_onnx: bool,
    split_min_params: int,
    skip_verify: bool,
    skip_mock: bool,
) -> ModelRunResult:
    """
    Runs a single model in a fresh subprocess.
    This prevents a single model OOM kill (exit code 137) from killing the whole benchmark run.
    """
    slug = slugify(spec.display_name)
    model_dir = ensure_dir(artifacts_root / slug)
    out_path = model_dir / "model_result.json"

    tail_lines = int(os.environ.get("MODEL_PROCESS_TAIL_LINES", "200"))

    cmd = [
        sys.executable,
        "-m",
        "ezkl_bench.worker",
        "--model",
        spec.key,
        "--cache",
        str(cache_root),
        "--artifacts",
        str(artifacts_root),
        "--repeats",
        str(repeats),
        "--warmup",
        str(warmup),
        "--out",
        str(out_path),
        "--log-level",
        str(log_level),
        "--split-onnx" if split_onnx else "--no-split-onnx",
        "--split-min-params",
        str(int(split_min_params)),
    ]
    if skip_verify:
        cmd.append("--skip-verify")
    if skip_mock:
        cmd.append("--skip-mock")

    logger.info("Spawning worker for model=%s cmd=%s", spec.key, cmd)
    res = run_subprocess_streaming(
        cmd=cmd,
        cwd=Path("/app"),
        env=dict(os.environ),
        timeout_s=None,
        tail_lines=tail_lines,
        line_prefix="",
    )

    rc = res.get("returncode")
    if rc != 0:
        tail = res.get("tail", [])
        logger.error("Worker failed for model %s (rc=%s). Tail:\n%s", spec.key, rc, "\n".join(tail))
        return ModelRunResult(
            display_name=spec.display_name,
            slug=slug,
            logrows=None,
            prove_times_s=[],
            verify_times_s=[],
            error=f"WorkerFailed: exit_code={rc}",
            traceback="\n".join(tail),
            diagnostics={"worker_cmd": cmd, "worker_returncode": rc, "worker_tail": tail},
        )

    if not out_path.exists():
        tail = res.get("tail", [])
        logger.error("Worker succeeded but no output file for model %s. Tail:\n%s", spec.key, "\n".join(tail))
        return ModelRunResult(
            display_name=spec.display_name,
            slug=slug,
            logrows=None,
            prove_times_s=[],
            verify_times_s=[],
            error="WorkerFailed: no result file produced",
            traceback="\n".join(tail),
            diagnostics={"worker_cmd": cmd, "worker_returncode": rc, "worker_tail": tail},
        )

    payload = __import__("json").loads(out_path.read_text(encoding="utf-8"))
    return ModelRunResult(
        display_name=payload.get("display_name", spec.display_name),
        slug=payload.get("slug", slug),
        logrows=payload.get("logrows"),
        prove_times_s=list(payload.get("prove_times_s", []) or []),
        verify_times_s=list(payload.get("verify_times_s", []) or []),
        setup_time_s=payload.get("setup_time_s"),
        compile_time_s=payload.get("compile_time_s"),
        calibrate_time_s=payload.get("calibrate_time_s"),
        error=payload.get("error"),
        traceback=payload.get("traceback"),
        diagnostics=payload.get("diagnostics") or {},
    )


def run_benchmark(
    specs: List[ModelSpec],
    out_json_path: Path,
    artifacts_root: Path,
    cache_root: Path,
    repeats: int,
    warmup: int,
    isolate_models: bool = True,
    fail_fast: bool = False,
    log_level: str = "INFO",
    split_onnx: bool = True,
    split_min_params: int = 50_000,
    skip_verify: bool = False,
    skip_mock: bool = False,
) -> Dict[str, Any]:
    ensure_dir(out_json_path.parent)
    ensure_dir(artifacts_root)
    ensure_dir(cache_root)

    try:
        import ezkl
    except Exception:
        ezkl = None  # type: ignore

    try:
        import torch
    except Exception:
        torch = None  # type: ignore

    try:
        import onnxruntime as ort
    except Exception:
        ort = None  # type: ignore

    meta = {
        "ezkl_version": getattr(ezkl, "__version__", "unknown") if ezkl else "import_failed",
        "ezkl_cli_version": run_command(["ezkl", "--version"]),
        "ezkl_cli_path": run_command(["bash", "-lc", "command -v ezkl || true"]),
        "rustc_version": run_command(["rustc", "--version"]),
        "nvidia_smi": run_command(["bash", "-lc", "nvidia-smi -L || true"]),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch_version": getattr(torch, "__version__", None),
        "cuda_available": bool(torch and torch.cuda.is_available()),
        "cuda_device": (torch.cuda.get_device_name(0) if torch and torch.cuda.is_available() else None),
        "onnxruntime_version": getattr(ort, "__version__", None) if ort else None,
        "onnxruntime_available_providers": (ort.get_available_providers() if ort else None),
        "env": {
            "ENABLE_ICICLE_GPU": os.environ.get("ENABLE_ICICLE_GPU"),
            "ICICLE_SMALL_K": os.environ.get("ICICLE_SMALL_K"),
            "NVIDIA_VISIBLE_DEVICES": os.environ.get("NVIDIA_VISIBLE_DEVICES"),
            "NVIDIA_DRIVER_CAPABILITIES": os.environ.get("NVIDIA_DRIVER_CAPABILITIES"),
            "ISOLATE_MODELS": os.environ.get("ISOLATE_MODELS"),
            "SPLIT_ONNX": str(split_onnx),
            "SPLIT_MIN_PARAMS": str(int(split_min_params)),
            "EZKL_ONNX_PRECISION": os.environ.get("EZKL_ONNX_PRECISION"),
            "EZKL_CHECK_MODE": os.environ.get("EZKL_CHECK_MODE"),
            "EZKL_LOOKUP_SAFETY_MARGIN": os.environ.get("EZKL_LOOKUP_SAFETY_MARGIN"),
            "EZKL_PROB_SEED_MODE": os.environ.get("EZKL_PROB_SEED_MODE"),
        },
        "system": get_system_stats([Path("/app"), artifacts_root, cache_root]),
    }
    logger.info("Metadata: %s", meta)

    models_out: Dict[str, Any] = {}
    final = {"meta": meta, "models": models_out}
    write_json(out_json_path, final)

    for spec in specs:
        if isolate_models:
            r = _run_single_model_in_worker(
                spec=spec,
                artifacts_root=artifacts_root,
                cache_root=cache_root,
                repeats=repeats,
                warmup=warmup,
                log_level=log_level,
                split_onnx=split_onnx,
                split_min_params=split_min_params,
                skip_verify=skip_verify,
                skip_mock=skip_mock,
            )
        else:
            r = run_single_model(
                spec=spec,
                artifacts_root=artifacts_root,
                cache_root=cache_root,
                repeats=repeats,
                warmup=warmup,
                split_onnx=split_onnx,
                split_min_params=split_min_params,
                skip_verify=skip_verify,
                skip_mock=skip_mock,
            )

        models_out[spec.display_name] = asdict(r)

        final = {"meta": meta, "models": models_out}
        write_json(out_json_path, final)
        logger.info("Progressive save: updated %s", out_json_path)

        if fail_fast and r.error:
            logger.error("Fail-fast enabled; aborting after model failure: %s", spec.display_name)
            break

    return final

