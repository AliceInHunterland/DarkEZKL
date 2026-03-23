from __future__ import annotations

import asyncio
import gc
import hashlib
import inspect
import logging
import os
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .models import ModelSpec
from .onnx_precision import convert_onnx_to_fp16, maybe_convert_onnx_precision
from .onnx_sanitize import (
    onnx_debug_dump_for_exception,
    sanitize_exported_onnx_inplace,
)
from .onnx_splitter import SplitSegment, run_onnx_segment, split_onnx_model
from .srs_utils import resolve_compatible_srs
from .utils import (
    cat_file,
    ensure_dir,
    env_flag,
    get_system_stats,
    read_json,
    run_subprocess_streaming,
    slugify,
    timed,
    write_json,
    write_json_compact,
)

logger = logging.getLogger(__name__)

_SEGMENT_RESULT_CACHE_VERSION = 1
_FILE_SHA256_CACHE: Dict[Tuple[str, int, int], str] = {}


@dataclass
class SplitTuning:
    min_params: int
    max_nodes: Optional[int]
    max_logrows: Optional[int]
    max_rows: Optional[int]
    max_assignments: Optional[int]


class CLICommandFailed(RuntimeError):
    def __init__(self, cmd: List[str], returncode: int, tail: List[str]):
        self.cmd = list(cmd)
        self.returncode = int(returncode)
        self.tail = list(tail)
        super().__init__(
            f"CLI command failed (rc={self.returncode}): {' '.join(self.cmd)}\nTail:\n" + "\n".join(self.tail)
        )


class SegmentNeedsResplitError(RuntimeError):
    def __init__(
        self,
        *,
        segment_idx: int,
        stage: str,
        reason: str,
        logrows: Optional[int],
        split_tuning: SplitTuning,
    ):
        self.segment_idx = int(segment_idx)
        self.stage = str(stage)
        self.reason = str(reason)
        self.logrows = None if logrows is None else int(logrows)
        self.split_tuning = split_tuning
        super().__init__(
            f"segment {self.segment_idx} needs resplit at stage={self.stage}: {self.reason} "
            f"(logrows={self.logrows}, min_params={self.split_tuning.min_params}, max_nodes={self.split_tuning.max_nodes})"
        )


@dataclass
class ModelRunResult:
    display_name: str
    slug: str
    logrows: Optional[int]
    prove_times_s: List[float]
    verify_times_s: List[float]
    setup_time_s: Optional[float] = None
    compile_time_s: Optional[float] = None
    calibrate_time_s: Optional[float] = None
    error: Optional[str] = None
    traceback: Optional[str] = None
    diagnostics: Dict[str, Any] = field(default_factory=dict)


def _path_stats(p: Path) -> Dict[str, Any]:
    try:
        st = p.stat()
        return {"path": str(p), "exists": True, "bytes": int(st.st_size)}
    except FileNotFoundError:
        return {"path": str(p), "exists": False, "bytes": None}


def _sha256_path(path: Path) -> str:
    st = path.stat()
    mtime_ns = int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1_000_000_000)))
    key = (str(path.resolve()), int(st.st_size), mtime_ns)
    cached = _FILE_SHA256_CACHE.get(key)
    if cached is not None:
        return cached

    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    digest = h.hexdigest()
    _FILE_SHA256_CACHE[key] = digest
    return digest


def _segment_cache_manifest_path(work_dir: Path) -> Path:
    return work_dir / "segment_result_cache.json"


def _segment_required_artifacts(
    *,
    settings_path: Path,
    compiled_path: Path,
    vk_path: Path,
    pk_path: Path,
    witness_path: Path,
    proof_path: Path,
    repeats: int,
    warmup: int,
) -> List[Path]:
    paths = [settings_path, compiled_path, vk_path, pk_path, witness_path]
    if repeats > 0 or warmup > 0:
        paths.append(proof_path)
    return paths


def _segment_cache_key(
    *,
    spec: ModelSpec,
    segment_idx: int,
    onnx_path: Path,
    input_path: Path,
    effective_input_scale: int,
    effective_param_scale: int,
    effective_num_inner_cols: int,
    repeats: int,
    warmup: int,
    explicit_logrows: Optional[int],
    prob_overrides: Dict[str, Any],
    calibration_knobs: Dict[str, Any],
    skip_mock: bool,
    skip_verify: bool,
    output_visibility: str,
) -> Dict[str, Any]:
    check_mode = (os.environ.get("EZKL_CHECK_MODE") or "safe").strip().lower()
    ignore_io_range = env_flag("EZKL_IGNORE_RANGE_CHECK_IO", default=(check_mode == "unsafe"))
    return {
        "version": _SEGMENT_RESULT_CACHE_VERSION,
        "spec_key": spec.key,
        "segment_idx": int(segment_idx),
        "onnx_sha256": _sha256_path(onnx_path),
        "input_sha256": _sha256_path(input_path),
        "effective": {
            "input_scale": int(effective_input_scale),
            "param_scale": int(effective_param_scale),
            "num_inner_cols": int(effective_num_inner_cols),
            "explicit_logrows": None if explicit_logrows is None else int(explicit_logrows),
            "check_mode": check_mode,
            "output_visibility": str(output_visibility),
            "ignore_range_check_inputs_outputs": bool(ignore_io_range),
            "prob_overrides": {
                "execution_mode": prob_overrides.get("execution_mode"),
                "prob_k": prob_overrides.get("prob_k"),
                "prob_ops": list(prob_overrides.get("prob_ops") or []),
                "prob_seed_mode": prob_overrides.get("prob_seed_mode"),
            },
            "calibration_knobs": {
                "lookup_safety_margin": calibration_knobs.get("lookup_safety_margin"),
                "max_logrows": calibration_knobs.get("max_logrows"),
            },
            "repeats": int(repeats),
            "warmup": int(warmup),
            "skip_mock": bool(skip_mock),
            "skip_verify": bool(skip_verify),
        },
    }


def _load_cached_segment_result(
    *,
    work_dir: Path,
    onnx_path: Path,
    input_path: Path,
    cache_key: Dict[str, Any],
    required_artifacts: List[Path],
) -> Optional[Tuple[Dict[str, Any], List[float], List[float]]]:
    if env_flag("EZKL_BENCH_DISABLE_SEGMENT_CACHE", default=False):
        return None

    manifest_path = _segment_cache_manifest_path(work_dir)
    if not manifest_path.exists():
        return None

    try:
        payload = read_json(manifest_path)
    except Exception as e:
        logger.warning("Ignoring unreadable segment cache manifest %s: %s", manifest_path, e)
        return None

    if payload.get("version") != _SEGMENT_RESULT_CACHE_VERSION:
        return None
    if payload.get("cache_key") != cache_key:
        return None

    missing = [str(p) for p in required_artifacts if not p.exists()]
    if missing:
        logger.info("Ignoring cached segment result %s because artifacts are missing: %s", manifest_path, missing)
        return None

    prove_times = [float(x) for x in (payload.get("prove_times_s") or [])]
    verify_times = [float(x) for x in (payload.get("verify_times_s") or [])]
    expected_repeats = int(cache_key.get("effective", {}).get("repeats") or 0)
    expected_verify_repeats = 0 if cache_key.get("effective", {}).get("skip_verify") else expected_repeats
    if len(prove_times) != expected_repeats or len(verify_times) != expected_verify_repeats:
        logger.info(
            "Ignoring cached segment result %s because measured run counts do not match repeats=%s verify_repeats=%s",
            manifest_path,
            expected_repeats,
            expected_verify_repeats,
        )
        return None

    seg_meta = dict(payload.get("seg_meta") or {})
    seg_meta["work_dir"] = str(work_dir)
    seg_meta["onnx"] = _path_stats(onnx_path)
    seg_meta["input"] = _path_stats(input_path)
    seg_meta["cache"] = {
        "status": "hit",
        "manifest": str(manifest_path),
        "version": _SEGMENT_RESULT_CACHE_VERSION,
    }
    logger.info("Reusing cached segment result: %s", manifest_path)
    return seg_meta, prove_times, verify_times


def _write_cached_segment_result(
    *,
    work_dir: Path,
    cache_key: Dict[str, Any],
    seg_meta: Dict[str, Any],
    prove_times: List[float],
    verify_times: List[float],
    required_artifacts: List[Path],
) -> None:
    if env_flag("EZKL_BENCH_DISABLE_SEGMENT_CACHE", default=False):
        return

    manifest_path = _segment_cache_manifest_path(work_dir)
    seg_meta_payload = {k: v for k, v in seg_meta.items() if k != "cache"}
    payload = {
        "version": _SEGMENT_RESULT_CACHE_VERSION,
        "cache_key": cache_key,
        "seg_meta": seg_meta_payload,
        "prove_times_s": [float(x) for x in prove_times],
        "verify_times_s": [float(x) for x in verify_times],
        "artifacts": {p.name: _path_stats(p) for p in required_artifacts},
    }
    write_json(manifest_path, payload)


def _flatten_tensor_for_ezkl(t) -> List[float]:
    """
    EZKL input.json expects each input as a flat 1-D array.
    """
    return t.detach().cpu().reshape(-1).tolist()


def _flatten_numpy_for_ezkl(a) -> List[float]:
    import numpy as np

    if not isinstance(a, np.ndarray):
        a = np.asarray(a)
    # JSON has no dtype; keep float32 to be consistent across fp32/fp16/int8 models.
    return a.reshape(-1).astype(np.float32).tolist()


def _make_ezkl_input_json_from_torch(dummy_input) -> Dict[str, Any]:
    return {"input_data": [_flatten_tensor_for_ezkl(dummy_input)]}


def _make_ezkl_input_json_from_numpy_list(inputs: List[Any]) -> Dict[str, Any]:
    if not inputs:
        return {"input_data": []}
    return {"input_data": [_flatten_numpy_for_ezkl(x) for x in inputs]}


def _export_onnx(model, dummy_input, onnx_path: Path) -> None:
    import torch

    logger.info("Exporting ONNX to %s", onnx_path)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    model_cpu = model.to("cpu")
    dummy_cpu = dummy_input.to("cpu")

    export_kwargs: Dict[str, Any] = {}
    # Reduce huge models' ONNX footprint by using external data format if available.
    try:
        sig = inspect.signature(torch.onnx.export)
        if "use_external_data_format" in sig.parameters:
            export_kwargs["use_external_data_format"] = True
        elif "large_model" in sig.parameters:
            export_kwargs["large_model"] = True
    except Exception:
        pass

    # Opset 14 is significantly more stable for ezkl/tract than newer opsets.
    torch.onnx.export(
        model_cpu,
        dummy_cpu,
        str(onnx_path),
        input_names=["input"],
        output_names=["output"],
        opset_version=14,
        do_constant_folding=True,
        **export_kwargs,
    )

    logger.info("ONNX exported: %s", _path_stats(onnx_path))


def _probabilistic_overrides_from_env() -> Dict[str, Any]:
    """
    Step 7 plumbing: parse benchmark-level probabilistic execution knobs from env vars.

    These env vars are set by ezkl_bench/cli.py so they also propagate into worker subprocesses:
      - EZKL_EXECUTION_MODE: exact|probabilistic
      - EZKL_PROB_K: integer
      - EZKL_PROB_OPS: comma-separated ops list (e.g. MatMul,Gemm,Conv)
    """
    out: Dict[str, Any] = {}

    mode = (os.environ.get("EZKL_EXECUTION_MODE") or "").strip().lower()
    if mode in ("exact", "probabilistic"):
        out["execution_mode"] = mode
    elif mode:
        logger.warning("Ignoring invalid EZKL_EXECUTION_MODE=%r (expected exact|probabilistic)", mode)

    prob_k_raw = (os.environ.get("EZKL_PROB_K") or "").strip()
    if prob_k_raw:
        try:
            out["prob_k"] = int(prob_k_raw)
        except Exception:
            logger.warning("Ignoring invalid EZKL_PROB_K=%r (expected integer)", prob_k_raw)

    prob_ops_raw = (os.environ.get("EZKL_PROB_OPS") or "").strip()
    if prob_ops_raw:
        ops = [x.strip() for x in prob_ops_raw.split(",") if x.strip()]
        if ops:
            out["prob_ops"] = ops

    prob_seed_mode = (os.environ.get("EZKL_PROB_SEED_MODE") or "").strip()
    if prob_seed_mode:
        out["prob_seed_mode"] = prob_seed_mode

    return out


def _model_env_suffix(spec: ModelSpec) -> str:
    key = (spec.key or "").strip().upper()
    return "".join(ch if ch.isalnum() else "_" for ch in key)


def _read_positive_int_env(candidates: List[str]) -> Optional[Tuple[str, int]]:
    for name in candidates:
        raw = (os.environ.get(name) or "").strip()
        if not raw:
            continue
        try:
            value = int(raw)
        except Exception:
            logger.warning("Ignoring invalid %s=%r (expected integer > 0)", name, raw)
            continue
        if value <= 0:
            logger.warning("Ignoring invalid %s=%r (expected integer > 0)", name, raw)
            continue
        return name, value
    return None


def _effective_num_inner_cols(spec: ModelSpec) -> int:
    candidates = [
        f"EZKL_BENCH_NUM_INNER_COLS_{_model_env_suffix(spec)}",
        "EZKL_BENCH_NUM_INNER_COLS",
    ]
    for name in candidates:
        raw = (os.environ.get(name) or "").strip()
        if not raw:
            continue
        try:
            value = int(raw)
        except Exception:
            logger.warning("Ignoring invalid %s=%r (expected integer > 0)", name, raw)
            continue
        if value <= 0:
            logger.warning("Ignoring invalid %s=%r (expected integer > 0)", name, raw)
            continue
        logger.info("Using num_inner_cols override from %s=%s for model=%s", name, value, spec.key)
        return value
    return int(spec.num_inner_cols)


def _effective_input_scale(spec: ModelSpec) -> int:
    hit = _read_positive_int_env(
        [
            f"EZKL_BENCH_INPUT_SCALE_{_model_env_suffix(spec)}",
            "EZKL_BENCH_INPUT_SCALE",
        ]
    )
    if hit is not None:
        name, value = hit
        logger.info("Using input scale override from %s=%s for model=%s", name, value, spec.key)
        return value
    return int(spec.input_scale)


def _effective_param_scale(spec: ModelSpec) -> int:
    hit = _read_positive_int_env(
        [
            f"EZKL_BENCH_PARAM_SCALE_{_model_env_suffix(spec)}",
            "EZKL_BENCH_PARAM_SCALE",
        ]
    )
    if hit is not None:
        name, value = hit
        logger.info("Using param scale override from %s=%s for model=%s", name, value, spec.key)
        return value
    return int(spec.param_scale)


def _effective_split_min_params(spec: ModelSpec, cli_default: int) -> int:
    hit = _read_positive_int_env(
        [
            f"EZKL_BENCH_SPLIT_MIN_PARAMS_{_model_env_suffix(spec)}",
            "EZKL_BENCH_SPLIT_MIN_PARAMS",
        ]
    )
    if hit is not None:
        name, value = hit
        logger.info("Using split min params override from %s=%s for model=%s", name, value, spec.key)
        return value
    if spec.split_min_params is not None:
        return int(spec.split_min_params)
    return int(cli_default)


def _effective_split_max_nodes(spec: ModelSpec) -> Optional[int]:
    hit = _read_positive_int_env(
        [
            f"EZKL_BENCH_SPLIT_MAX_NODES_{_model_env_suffix(spec)}",
            "EZKL_BENCH_SPLIT_MAX_NODES",
            "SPLIT_MAX_NODES",
        ]
    )
    if hit is not None:
        name, value = hit
        logger.info("Using split max nodes override from %s=%s for model=%s", name, value, spec.key)
        return value
    if spec.split_max_nodes is not None:
        return int(spec.split_max_nodes)
    return None


def _effective_max_segment_logrows(spec: ModelSpec) -> Optional[int]:
    hit = _read_positive_int_env(
        [
            f"EZKL_BENCH_MAX_SEGMENT_LOGROWS_{_model_env_suffix(spec)}",
            "EZKL_BENCH_MAX_SEGMENT_LOGROWS",
        ]
    )
    if hit is not None:
        name, value = hit
        logger.info("Using max segment logrows override from %s=%s for model=%s", name, value, spec.key)
        return value
    if spec.max_segment_logrows is not None:
        return int(spec.max_segment_logrows)
    return None


def _effective_max_segment_rows(spec: ModelSpec) -> Optional[int]:
    hit = _read_positive_int_env(
        [
            f"EZKL_BENCH_MAX_SEGMENT_ROWS_{_model_env_suffix(spec)}",
            "EZKL_BENCH_MAX_SEGMENT_ROWS",
        ]
    )
    if hit is not None:
        name, value = hit
        logger.info("Using max segment rows override from %s=%s for model=%s", name, value, spec.key)
        return value
    if spec.max_segment_rows is not None:
        return int(spec.max_segment_rows)
    return None


def _effective_max_segment_assignments(spec: ModelSpec) -> Optional[int]:
    hit = _read_positive_int_env(
        [
            f"EZKL_BENCH_MAX_SEGMENT_ASSIGNMENTS_{_model_env_suffix(spec)}",
            "EZKL_BENCH_MAX_SEGMENT_ASSIGNMENTS",
        ]
    )
    if hit is not None:
        name, value = hit
        logger.info("Using max segment assignments override from %s=%s for model=%s", name, value, spec.key)
        return value
    if spec.max_segment_assignments is not None:
        return int(spec.max_segment_assignments)
    return None


def _effective_split_retry_budget(spec: ModelSpec) -> int:
    hit = _read_positive_int_env(
        [
            f"EZKL_BENCH_SPLIT_MAX_RETRIES_{_model_env_suffix(spec)}",
            "EZKL_BENCH_SPLIT_MAX_RETRIES",
        ]
    )
    if hit is not None:
        name, value = hit
        logger.info("Using split retry budget override from %s=%s for model=%s", name, value, spec.key)
        return value
    if spec.split_retry_budget is not None:
        return int(spec.split_retry_budget)
    return 3 if spec.enable_onnx_split else 0


def _tighten_split_tuning(current: SplitTuning) -> Optional[SplitTuning]:
    if current.max_nodes is None:
        return SplitTuning(
            min_params=current.min_params,
            max_nodes=12,
            max_logrows=current.max_logrows,
            max_rows=current.max_rows,
            max_assignments=current.max_assignments,
        )
    if current.max_nodes > 8:
        return SplitTuning(
            min_params=current.min_params,
            max_nodes=8,
            max_logrows=current.max_logrows,
            max_rows=current.max_rows,
            max_assignments=current.max_assignments,
        )
    if current.max_nodes > 6:
        return SplitTuning(
            min_params=current.min_params,
            max_nodes=6,
            max_logrows=current.max_logrows,
            max_rows=current.max_rows,
            max_assignments=current.max_assignments,
        )
    if current.max_nodes > 4:
        return SplitTuning(
            min_params=current.min_params,
            max_nodes=4,
            max_logrows=current.max_logrows,
            max_rows=current.max_rows,
            max_assignments=current.max_assignments,
        )
    if current.max_nodes > 3:
        return SplitTuning(
            min_params=current.min_params,
            max_nodes=3,
            max_logrows=current.max_logrows,
            max_rows=current.max_rows,
            max_assignments=current.max_assignments,
        )
    if current.max_nodes > 2:
        return SplitTuning(
            min_params=current.min_params,
            max_nodes=2,
            max_logrows=current.max_logrows,
            max_rows=current.max_rows,
            max_assignments=current.max_assignments,
        )
    if current.max_nodes > 1:
        return SplitTuning(
            min_params=current.min_params,
            max_nodes=1,
            max_logrows=current.max_logrows,
            max_rows=current.max_rows,
            max_assignments=current.max_assignments,
        )
    next_min_params = max(1_000, int(current.min_params) // 2)
    if next_min_params < int(current.min_params):
        return SplitTuning(
            min_params=next_min_params,
            max_nodes=current.max_nodes,
            max_logrows=current.max_logrows,
            max_rows=current.max_rows,
            max_assignments=current.max_assignments,
        )
    return None

def _effective_output_visibility(*, segment_pos: int, segment_count: int) -> str:
    if segment_count > 1 and segment_pos < (segment_count - 1):
        return "private"
    return "public"


def _enforce_settings_output_visibility(settings_path: Path, output_visibility: str) -> None:
    try:
        settings = read_json(settings_path)
    except Exception as e:
        logger.warning("Unable to read settings for output_visibility enforcement (%s): %s", settings_path, e)
        return

    run_args = settings.get("run_args")
    if not isinstance(run_args, dict):
        logger.warning("Skipping output_visibility enforcement for malformed settings: %s", settings_path)
        return

    desired = str(output_visibility).capitalize()
    current = str(run_args.get("output_visibility") or "")
    if current == desired:
        return

    logger.info(
        "Rewriting settings output_visibility for %s: %s -> %s",
        settings_path,
        current or "<missing>",
        desired,
    )
    run_args["output_visibility"] = desired
    write_json(settings_path, settings)


def _mk_run_args(spec: ModelSpec, explicit_logrows: Optional[int], output_visibility: str) -> Optional[Any]:
    import ezkl

    run_args_cls = getattr(ezkl, "PyRunArgs", None) or getattr(ezkl, "RunArgs", None)
    if run_args_cls is None:
        logger.warning(
            "ezkl python API missing PyRunArgs/RunArgs; falling back to CLI-driven gen-settings where needed."
        )
        return None

    run_args = run_args_cls()

    # Visibility enums changed across versions; support both.
    if hasattr(ezkl, "PyVisibility"):
        try:
            run_args.input_visibility = ezkl.PyVisibility.Private
            run_args.output_visibility = getattr(ezkl.PyVisibility, str(output_visibility).capitalize())
            run_args.param_visibility = ezkl.PyVisibility.Fixed
        except Exception:
            run_args.input_visibility = "private"
            run_args.output_visibility = str(output_visibility)
            run_args.param_visibility = "fixed"
    else:
        run_args.input_visibility = "private"
        run_args.output_visibility = str(output_visibility)
        run_args.param_visibility = "fixed"

    # --- performance/defensiveness knobs (best-effort) ---
    check_mode = (os.environ.get("EZKL_CHECK_MODE") or "safe").strip().lower()
    if hasattr(run_args, "check_mode"):
        run_args.check_mode = check_mode

    ignore_io_range = env_flag("EZKL_IGNORE_RANGE_CHECK_IO", default=(check_mode == "unsafe"))
    if hasattr(run_args, "ignore_range_check_inputs_outputs"):
        run_args.ignore_range_check_inputs_outputs = bool(ignore_io_range)

    run_args.input_scale = _effective_input_scale(spec)
    run_args.param_scale = _effective_param_scale(spec)
    run_args.num_inner_cols = _effective_num_inner_cols(spec)

    if explicit_logrows is not None:
        run_args.logrows = int(explicit_logrows)

    prob_seed_mode = (os.environ.get("EZKL_PROB_SEED_MODE") or "").strip()
    if prob_seed_mode and hasattr(run_args, "prob_seed_mode"):
        try:
            run_args.prob_seed_mode = prob_seed_mode
        except Exception:
            logger.debug("Unable to set prob_seed_mode=%s on run args", prob_seed_mode)

    # Note: probabilistic execution parameters are passed via ezkl.gen_settings kwargs
    # (execution_mode / prob_k / prob_ops) instead of trying to mutate RunArgs here.

    return run_args


def _run_cli_checked(cmd: List[str], *, cwd: Optional[Path] = None) -> None:
    logger.info("CLI fallback: %s", " ".join(cmd))
    res = run_subprocess_streaming(
        cmd=cmd,
        cwd=cwd,
        env=dict(os.environ),
        timeout_s=None,
        tail_lines=200,
        line_prefix="",
    )
    rc = res.get("returncode")
    if rc != 0:
        tail = "\n".join(res.get("tail", []))
        raise CLICommandFailed(cmd=cmd, returncode=int(rc), tail=list(res.get("tail", [])))


def _build_gen_settings_cli(
    *,
    spec: ModelSpec,
    onnx_path: Path,
    settings_path: Path,
    explicit_logrows: Optional[int],
    prob_overrides: Dict[str, Any],
    output_visibility: str,
) -> List[str]:
    input_scale = _effective_input_scale(spec)
    param_scale = _effective_param_scale(spec)
    cmd = [
        "ezkl",
        "gen-settings",
        "-M",
        str(onnx_path),
        "-O",
        str(settings_path),
        "--input-scale",
        str(input_scale),
        "--param-scale",
        str(param_scale),
        "--num-inner-cols",
        str(_effective_num_inner_cols(spec)),
        "--input-visibility",
        "private",
        "--output-visibility",
        str(output_visibility),
        "--param-visibility",
        "fixed",
        "--check-mode",
        (os.environ.get("EZKL_CHECK_MODE") or "safe").strip().lower(),
    ]

    if explicit_logrows is not None:
        cmd += ["--logrows", str(int(explicit_logrows))]

    execution_mode = prob_overrides.get("execution_mode")
    if execution_mode:
        cmd += ["--execution-mode", str(execution_mode)]

    prob_k = prob_overrides.get("prob_k")
    if prob_k is not None:
        cmd += ["--prob-k", str(int(prob_k))]

    prob_ops = prob_overrides.get("prob_ops")
    if prob_ops:
        cmd += ["--prob-ops", ",".join(str(x) for x in prob_ops)]

    prob_seed_mode = prob_overrides.get("prob_seed_mode")
    if prob_seed_mode:
        cmd += ["--prob-seed-mode", str(prob_seed_mode)]

    return cmd


def run_ezkl_safe(func, *args, **kwargs):
    """
    Executes an ezkl function ensuring an asyncio event loop is available.
    Handles both synchronous and coroutine-returning ezkl bindings.
    """
    name = getattr(func, "__name__", str(func))
    logger.debug("Running ezkl function safely: %s", name)

    async def _runner():
        if inspect.iscoroutinefunction(func):
            return await func(*args, **kwargs)
        res = func(*args, **kwargs)
        if inspect.isawaitable(res):
            return await res
        return res

    return asyncio.run(_runner())


def check_safetensor_error(e: Exception) -> bool:
    s = str(e)
    return (
        "SafetensorError" in s
        and (
            "header too large" in s.lower()
            or "metadata incomplete" in s.lower()
            or "invalid header" in s.lower()
        )
    )


def _calibration_knobs_from_env() -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    v = os.environ.get("EZKL_LOOKUP_SAFETY_MARGIN")
    if v:
        try:
            out["lookup_safety_margin"] = float(v)
        except Exception:
            pass
    v2 = os.environ.get("EZKL_MAX_LOGROWS")
    if v2:
        try:
            out["max_logrows"] = int(v2)
        except Exception:
            pass
    return out


def _run_single_onnx_pipeline(
    *,
    spec: ModelSpec,
    segment_pos: int,
    segment_idx: int,
    segment_count: int,
    onnx_path: Path,
    input_json_payload: Dict[str, Any],
    work_dir: Path,
    artifacts_root: Path,
    repeats: int,
    warmup: int,
    explicit_logrows: Optional[int] = None,
    split_tuning: Optional[SplitTuning] = None,
    skip_verify: bool = False,
    skip_mock: bool = False,
) -> Tuple[Dict[str, Any], List[float], List[float]]:
    import ezkl
    import torch

    ensure_dir(work_dir)

    settings_path = work_dir / "settings.json"
    input_path = work_dir / "input.json"
    compiled_path = work_dir / "network.ezkl"
    vk_path = work_dir / "vk.key"
    pk_path = work_dir / "pk.key"
    witness_path = work_dir / "witness.json"
    proof_path = work_dir / "proof.json"

    srs_cache_dir = ensure_dir(artifacts_root / "_srs_cache")

    seg_meta: Dict[str, Any] = {"work_dir": str(work_dir), "onnx": _path_stats(onnx_path)}
    prove_times: List[float] = []
    verify_times: List[float] = []

    write_json_compact(input_path, input_json_payload)
    seg_meta["input"] = _path_stats(input_path)

    # --- Step 7: probabilistic execution knobs from CLI/env ---
    prob_overrides = _probabilistic_overrides_from_env()
    seg_meta["probabilistic_overrides"] = dict(prob_overrides)
    calib_knobs = _calibration_knobs_from_env()
    seg_meta["calibration_knobs"] = dict(calib_knobs)
    output_visibility = _effective_output_visibility(segment_pos=segment_pos, segment_count=segment_count)
    seg_meta["output_visibility"] = output_visibility
    seg_meta["segment_position"] = int(segment_pos)
    eff_input_scale = _effective_input_scale(spec)
    eff_param_scale = _effective_param_scale(spec)
    eff_num_inner_cols = _effective_num_inner_cols(spec)
    required_artifacts = _segment_required_artifacts(
        settings_path=settings_path,
        compiled_path=compiled_path,
        vk_path=vk_path,
        pk_path=pk_path,
        witness_path=witness_path,
        proof_path=proof_path,
        repeats=repeats,
        warmup=warmup,
    )

    cache_key = _segment_cache_key(
        spec=spec,
        segment_idx=segment_idx,
        onnx_path=onnx_path,
        input_path=input_path,
        effective_input_scale=eff_input_scale,
        effective_param_scale=eff_param_scale,
        effective_num_inner_cols=eff_num_inner_cols,
        repeats=repeats,
        warmup=warmup,
        explicit_logrows=explicit_logrows,
        prob_overrides=prob_overrides,
        calibration_knobs=calib_knobs,
        skip_mock=skip_mock,
        skip_verify=skip_verify,
        output_visibility=output_visibility,
    )
    cached_segment = _load_cached_segment_result(
        work_dir=work_dir,
        onnx_path=onnx_path,
        input_path=input_path,
        cache_key=cache_key,
        required_artifacts=required_artifacts,
    )
    if cached_segment is not None:
        return cached_segment

    logger.info("Generating settings (work_dir=%s)...", work_dir)
    run_args = _mk_run_args(spec=spec, explicit_logrows=explicit_logrows, output_visibility=output_visibility)

    if run_args is not None:
        logger.info(
            "RunArgs: input_scale=%s param_scale=%s num_inner_cols=%s explicit_logrows=%s "
            "check_mode=%s ignore_range_check_inputs_outputs=%s output_visibility=%s prob_overrides=%s",
            getattr(run_args, "input_scale", None),
            getattr(run_args, "param_scale", None),
            getattr(run_args, "num_inner_cols", None),
            explicit_logrows,
            getattr(run_args, "check_mode", None),
            getattr(run_args, "ignore_range_check_inputs_outputs", None),
            output_visibility,
            prob_overrides,
        )
    else:
        logger.info(
            "RunArgs unavailable in Python binding; using CLI fallback. "
            "spec_input_scale=%s spec_param_scale=%s spec_num_inner_cols=%s explicit_logrows=%s "
            "output_visibility=%s prob_overrides=%s",
            eff_input_scale,
            eff_param_scale,
            eff_num_inner_cols,
            explicit_logrows,
            output_visibility,
            prob_overrides,
        )

    # kwargs passed to ezkl.gen_settings for probabilistic execution
    prob_kwargs: Dict[str, Any] = {}
    if "execution_mode" in prob_overrides:
        prob_kwargs["execution_mode"] = prob_overrides["execution_mode"]
    if "prob_k" in prob_overrides:
        prob_kwargs["prob_k"] = int(prob_overrides["prob_k"])
    if "prob_ops" in prob_overrides:
        prob_kwargs["prob_ops"] = list(prob_overrides["prob_ops"])

    with timed("gen_settings") as t:
        if run_args is None:
            _run_cli_checked(
                _build_gen_settings_cli(
                    spec=spec,
                    onnx_path=onnx_path,
                    settings_path=settings_path,
                    explicit_logrows=explicit_logrows,
                    prob_overrides=prob_overrides,
                    output_visibility=output_visibility,
                )
            )
            ok = True
        else:

            def _do_gen_settings():
                # Try a few calling conventions (ezkl Python API varies across versions).
                # Step 7 requirement: pass execution_mode / prob_k / prob_ops into gen_settings.
                try:
                    return ezkl.gen_settings(
                        model=str(onnx_path),
                        output=str(settings_path),
                        py_run_args=run_args,
                        **prob_kwargs,
                    )
                except TypeError:
                    pass

                try:
                    return ezkl.gen_settings(
                        model=str(onnx_path),
                        output=str(settings_path),
                        run_args=run_args,
                        **prob_kwargs,
                    )
                except TypeError:
                    pass

                # Some versions use `settings_path` naming.
                try:
                    return ezkl.gen_settings(
                        model=str(onnx_path),
                        settings_path=str(settings_path),
                        py_run_args=run_args,
                        **prob_kwargs,
                    )
                except TypeError:
                    pass

                # Kata-snapshot / minimal bindings: allow settings-only update.
                # (This won't generate full run_args; it's just best-effort compatibility.)
                return ezkl.gen_settings(settings_path=str(settings_path), **prob_kwargs)

            ok = run_ezkl_safe(_do_gen_settings)

    seg_meta["gen_settings_time_s"] = float(t["elapsed"] or 0.0)
    if not ok:
        raise RuntimeError("ezkl.gen_settings returned false")
    _enforce_settings_output_visibility(settings_path, output_visibility)
    seg_meta["settings"] = _path_stats(settings_path)

    logger.info("Calibrating settings (work_dir=%s)...", work_dir)
    with timed("calibrate_settings") as t:
        try:

            def _do_calibrate():
                kwargs: Dict[str, Any] = {}
                if hasattr(ezkl, "PyCalibrationTarget"):
                    kwargs["target"] = ezkl.PyCalibrationTarget.Resources
                else:
                    kwargs["target"] = "resources"
                kwargs.update(calib_knobs)

                # Best-effort: preserve probabilistic fields through calibration if the binding supports it.
                kwargs.update(prob_kwargs)

                try:
                    return ezkl.calibrate_settings(
                        data=str(input_path),
                        model=str(onnx_path),
                        settings=str(settings_path),
                        **kwargs,
                    )
                except TypeError:
                    try:
                        return ezkl.calibrate_settings(str(input_path), str(onnx_path), str(settings_path), **kwargs)
                    except TypeError:
                        return ezkl.calibrate_settings(
                            str(input_path), str(onnx_path), str(settings_path), kwargs.get("target", "resources")
                        )

            ok = run_ezkl_safe(_do_calibrate)
        except Exception as e:
            logger.warning("calibrate_settings python call failed (%s); falling back to CLI", e)
            cli_cmd = [
                "ezkl",
                "calibrate-settings",
                "-M",
                str(onnx_path),
                "-D",
                str(input_path),
                "--settings-path",
                str(settings_path),
                "--target",
                "resources",
            ]
            if "lookup_safety_margin" in calib_knobs:
                cli_cmd += ["--lookup-safety-margin", str(calib_knobs["lookup_safety_margin"])]
            if "max_logrows" in calib_knobs:
                cli_cmd += ["--max-logrows", str(int(calib_knobs["max_logrows"]))]
            _run_cli_checked(cli_cmd)
            ok = True

    seg_meta["calibrate_time_s"] = float(t["elapsed"] or 0.0)
    if not ok:
        raise RuntimeError("ezkl.calibrate_settings returned false")
    _enforce_settings_output_visibility(settings_path, output_visibility)

    settings = read_json(settings_path)
    seg_logrows = int(settings["run_args"]["logrows"])
    seg_rows = settings.get("num_rows")
    seg_assignments = settings.get("total_assignments")
    seg_meta["logrows"] = seg_logrows
    seg_meta["num_rows"] = None if seg_rows is None else int(seg_rows)
    seg_meta["total_assignments"] = None if seg_assignments is None else int(seg_assignments)
    logger.info("Calibration done. Selected logrows=%s (work_dir=%s)", seg_logrows, work_dir)
    if split_tuning is not None and split_tuning.max_logrows is not None and seg_logrows > split_tuning.max_logrows:
        raise SegmentNeedsResplitError(
            segment_idx=segment_idx,
            stage="calibration",
            reason=f"calibrated logrows {seg_logrows} exceeds cap {split_tuning.max_logrows}",
            logrows=seg_logrows,
            split_tuning=split_tuning,
        )
    if split_tuning is not None and split_tuning.max_rows is not None and seg_rows is not None:
        if int(seg_rows) > split_tuning.max_rows:
            raise SegmentNeedsResplitError(
                segment_idx=segment_idx,
                stage="calibration",
                reason=f"calibrated num_rows {int(seg_rows)} exceeds cap {split_tuning.max_rows}",
                logrows=seg_logrows,
                split_tuning=split_tuning,
            )
    if split_tuning is not None and split_tuning.max_assignments is not None and seg_assignments is not None:
        if int(seg_assignments) > split_tuning.max_assignments:
            raise SegmentNeedsResplitError(
                segment_idx=segment_idx,
                stage="calibration",
                reason=f"calibrated total_assignments {int(seg_assignments)} exceeds cap {split_tuning.max_assignments}",
                logrows=seg_logrows,
                split_tuning=split_tuning,
            )

    logger.info("Compiling circuit (work_dir=%s)...", work_dir)
    with timed("compile_circuit") as t:
        try:

            def _do_compile():
                try:
                    return ezkl.compile_circuit(
                        model=str(onnx_path),
                        compiled_circuit=str(compiled_path),
                        settings_path=str(settings_path),
                    )
                except TypeError:
                    return ezkl.compile_circuit(str(onnx_path), str(compiled_path), str(settings_path))

            ok = run_ezkl_safe(_do_compile)
        except Exception as e:
            logger.warning("compile_circuit python call failed (%s); falling back to CLI", e)
            _run_cli_checked(
                [
                    "ezkl",
                    "compile-circuit",
                    "-M",
                    str(onnx_path),
                    "-S",
                    str(settings_path),
                    "--compiled-circuit",
                    str(compiled_path),
                ]
            )
            ok = True

    seg_meta["compile_time_s"] = float(t["elapsed"] or 0.0)
    if not ok:
        raise RuntimeError("ezkl.compile_circuit returned false")
    seg_meta["compiled"] = _path_stats(compiled_path)

    local_srs_path = srs_cache_dir / f"k{seg_logrows}.srs"
    resolved_srs = resolve_compatible_srs(
        seg_logrows,
        preferred_paths=[local_srs_path],
        search_dirs=[srs_cache_dir],
    )
    if resolved_srs is None:
        srs_path = local_srs_path
        logger.info("Downloading SRS for k=%s (work_dir=%s)...", seg_logrows, work_dir)
        with timed("get_srs") as t:
            try:
                ok = run_ezkl_safe(ezkl.get_srs, settings_path=str(settings_path), srs_path=str(srs_path))
            except Exception as e:
                logger.warning("get_srs python call failed (%s); falling back to CLI", e)
                _run_cli_checked(
                    [
                        "ezkl",
                        "get-srs",
                        "--settings-path",
                        str(settings_path),
                        "--srs-path",
                        str(srs_path),
                    ]
                )
                ok = True
        seg_meta["get_srs_time_s"] = float(t["elapsed"] or 0.0)
        if not ok:
            raise RuntimeError("ezkl.get_srs returned false")
        logger.info("SRS downloaded: %s", _path_stats(srs_path))
    else:
        srs_path = resolved_srs.path
        seg_meta["srs_requested_logrows"] = seg_logrows
        seg_meta["srs_named_logrows"] = resolved_srs.available_logrows
        seg_meta["srs_exact_match"] = resolved_srs.exact_match
        if resolved_srs.exact_match:
            logger.info("Using cached SRS: %s", _path_stats(srs_path))
        else:
            logger.info(
                "Using compatible larger SRS: requested_k=%s available_k=%s path=%s",
                seg_logrows,
                resolved_srs.available_logrows,
                srs_path,
            )
    seg_meta["srs"] = _path_stats(srs_path)

    logger.info("Running setup (pk/vk gen) (work_dir=%s)...", work_dir)
    with timed("setup") as t:
        try:
            ok = run_ezkl_safe(
                ezkl.setup,
                model=str(compiled_path),
                vk_path=str(vk_path),
                pk_path=str(pk_path),
                srs_path=str(srs_path),
            )
        except Exception as e:
            logger.warning("setup python call failed (%s); falling back to CLI", e)
            try:
                _run_cli_checked(
                    [
                        "ezkl",
                        "setup",
                        "-M",
                        str(compiled_path),
                        "--vk-path",
                        str(vk_path),
                        "--pk-path",
                        str(pk_path),
                        "--srs-path",
                        str(srs_path),
                    ]
                )
            except CLICommandFailed as cli_err:
                if cli_err.returncode == -9 and split_tuning is not None:
                    raise SegmentNeedsResplitError(
                        segment_idx=segment_idx,
                        stage="setup",
                        reason=f"setup killed with rc={cli_err.returncode}",
                        logrows=seg_logrows,
                        split_tuning=split_tuning,
                    ) from cli_err
                raise
            ok = True
    seg_meta["setup_time_s"] = float(t["elapsed"] or 0.0)
    if not ok:
        raise RuntimeError("ezkl.setup returned false")
    seg_meta["vk"] = _path_stats(vk_path)
    seg_meta["pk"] = _path_stats(pk_path)

    logger.info("Generating witness (work_dir=%s)...", work_dir)
    with timed("gen_witness") as t:
        try:
            ok = run_ezkl_safe(
                ezkl.gen_witness,
                data=str(input_path),
                model=str(compiled_path),
                output=str(witness_path),
                vk_path=str(vk_path),
                srs_path=str(srs_path),
            )
        except Exception as e:
            logger.warning("gen_witness python call failed (%s); falling back to CLI", e)
            _run_cli_checked(
                [
                    "ezkl",
                    "gen-witness",
                    "-M",
                    str(compiled_path),
                    "-D",
                    str(input_path),
                    "--output",
                    str(witness_path),
                    "--vk-path",
                    str(vk_path),
                    "--srs-path",
                    str(srs_path),
                ]
            )
            ok = True
    seg_meta["gen_witness_time_s"] = float(t["elapsed"] or 0.0)
    if not ok:
        raise RuntimeError("ezkl.gen_witness returned false")
    seg_meta["witness"] = _path_stats(witness_path)

    if skip_mock:
        logger.info("Skipping mock (work_dir=%s)...", work_dir)
        seg_meta["mock_skipped"] = True
        seg_meta["mock_time_s"] = 0.0
    else:
        logger.info("Running mock (work_dir=%s)...", work_dir)
        with timed("mock") as t:
            try:

                def _do_mock():
                    try:
                        ezkl.mock(witness=str(witness_path), model=str(compiled_path))
                    except TypeError:
                        ezkl.mock(str(witness_path), str(compiled_path))

                run_ezkl_safe(_do_mock)
            except Exception as e:
                logger.warning("mock python call failed (%s); falling back to CLI", e)
                _run_cli_checked(
                    [
                        "ezkl",
                        "mock",
                        "-M",
                        str(compiled_path),
                        "--witness",
                        str(witness_path),
                    ]
                )

        seg_meta["mock_time_s"] = float(t["elapsed"] or 0.0)

    logger.info("Warming up (%s iters) (work_dir=%s)...", warmup, work_dir)
    for wi in range(max(0, warmup)):
        logger.info("Warmup iteration %s/%s (work_dir=%s)", wi + 1, warmup, work_dir)
        try:
            ok = run_ezkl_safe(
                ezkl.prove,
                witness=str(witness_path),
                model=str(compiled_path),
                pk_path=str(pk_path),
                proof_path=str(proof_path),
                srs_path=str(srs_path),
            )
        except Exception as e:
            logger.warning("warmup prove python call failed (%s); falling back to CLI", e)
            _run_cli_checked(
                [
                    "ezkl",
                    "prove",
                    "-M",
                    str(compiled_path),
                    "--witness",
                    str(witness_path),
                    "--pk-path",
                    str(pk_path),
                    "--proof-path",
                    str(proof_path),
                    "--srs-path",
                    str(srs_path),
                ]
            )
            ok = True
        if not ok:
            raise RuntimeError("Warmup prove failed")

        if skip_verify:
            continue

        try:
            ok = run_ezkl_safe(
                ezkl.verify,
                proof_path=str(proof_path),
                settings_path=str(settings_path),
                vk_path=str(vk_path),
                srs_path=str(srs_path),
            )
        except Exception as e:
            logger.warning("warmup verify python call failed (%s); falling back to CLI", e)
            _run_cli_checked(
                [
                    "ezkl",
                    "verify",
                    "--proof-path",
                    str(proof_path),
                    "--settings-path",
                    str(settings_path),
                    "--vk-path",
                    str(vk_path),
                    "--srs-path",
                    str(srs_path),
                ]
            )
            ok = True
        if not ok:
            raise RuntimeError("Warmup verify failed")

    logger.info("Starting %s measured runs (work_dir=%s)...", repeats, work_dir)
    for i in range(repeats):
        with timed(f"prove(run {i+1})") as tprove:
            try:
                ok = run_ezkl_safe(
                    ezkl.prove,
                    witness=str(witness_path),
                    model=str(compiled_path),
                    pk_path=str(pk_path),
                    proof_path=str(proof_path),
                    srs_path=str(srs_path),
                )
            except Exception as e:
                logger.warning("prove python call failed (%s); falling back to CLI", e)
                _run_cli_checked(
                    [
                        "ezkl",
                        "prove",
                        "-M",
                        str(compiled_path),
                        "--witness",
                        str(witness_path),
                        "--pk-path",
                        str(pk_path),
                        "--proof-path",
                        str(proof_path),
                        "--srs-path",
                        str(srs_path),
                    ]
                )
                ok = True
        if not ok:
            raise RuntimeError("ezkl.prove returned false")
        prove_times.append(float(tprove["elapsed"] or 0.0))

        if skip_verify:
            continue

        with timed(f"verify(run {i+1})") as tver:
            try:
                ok = run_ezkl_safe(
                    ezkl.verify,
                    proof_path=str(proof_path),
                    settings_path=str(settings_path),
                    vk_path=str(vk_path),
                    srs_path=str(srs_path),
                )
            except Exception as e:
                logger.warning("verify python call failed (%s); falling back to CLI", e)
                _run_cli_checked(
                    [
                        "ezkl",
                        "verify",
                        "--proof-path",
                        str(proof_path),
                        "--settings-path",
                        str(settings_path),
                        "--vk-path",
                        str(vk_path),
                        "--srs-path",
                        str(srs_path),
                    ]
                )
                ok = True
        if not ok:
            raise RuntimeError("ezkl.verify returned false")
        verify_times.append(float(tver["elapsed"] or 0.0))

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    seg_meta["cache"] = {
        "status": "stored",
        "manifest": str(_segment_cache_manifest_path(work_dir)),
        "version": _SEGMENT_RESULT_CACHE_VERSION,
    }
    _write_cached_segment_result(
        work_dir=work_dir,
        cache_key=cache_key,
        seg_meta=seg_meta,
        prove_times=prove_times,
        verify_times=verify_times,
        required_artifacts=required_artifacts,
    )
    return seg_meta, prove_times, verify_times


def _apply_precision_to_segments(
    *,
    segments_fp32: List[SplitSegment],
    out_dir: Path,
    precision_mode: str,
) -> Tuple[List[SplitSegment], Dict[str, Any]]:
    """
    For split mode we apply precision *per-segment* so each segment keeps
    stable float32 IO types (keep_io_types=True) while internal weights/ops
    can be fp16.

    This fixes the ORT load error observed on 2026-01-08 where split fp16
    segments had inconsistent declared output types.
    """
    out_dir = ensure_dir(out_dir)
    mode = (precision_mode or "fp16").strip().lower()

    meta: Dict[str, Any] = {"mode": mode, "segments": []}

    # fp32: reuse the fp32 segments as-is.
    if mode in ("fp32", "float32", "f32", "none"):
        return segments_fp32, {"mode": "fp32", "segments": [], "note": "no per-segment conversion"}

    # fp16: convert each segment with keep_io_types=True.
    if mode in ("fp16", "float16", "f16"):
        segs_out: List[SplitSegment] = []
        for s in segments_fp32:
            src = Path(s.onnx_path)
            dst = out_dir / Path(s.onnx_path).name  # seg_XXX.onnx
            convert_onnx_to_fp16(src, dst, keep_io_types=True)
            # Also sanitize (infer shapes) to keep IO types consistent.
            sanitize_exported_onnx_inplace(dst, rewrite_gemm=False)

            segs_out.append(
                SplitSegment(
                    idx=s.idx,
                    onnx_path=str(dst),
                    input_names=list(s.input_names),
                    output_names=list(s.output_names),
                    node_count=s.node_count,
                    param_count=s.param_count,
                )
            )
            meta["segments"].append({"idx": s.idx, "src": str(src), "dst": str(dst)})
        return segs_out, meta

    # int8: keep current behavior as "best-effort" on the whole model, not per segment.
    # For now, fall back to fp32 segments with a clear warning.
    logger.warning("Split-mode per-segment precision: unsupported mode=%s; falling back to fp32 segments", mode)
    return segments_fp32, {"mode": mode, "segments": [], "warning": "unsupported in split-mode; used fp32 segments"}


def run_single_model(
    spec: ModelSpec,
    artifacts_root: Path,
    cache_root: Path,
    repeats: int,
    warmup: int = 1,
    explicit_logrows: Optional[int] = None,
    split_onnx: Optional[bool] = None,
    split_min_params: int = 50_000,
    skip_verify: bool = False,
    skip_mock: bool = False,
) -> ModelRunResult:
    import torch

    slug = slugify(spec.display_name)
    logger.info("--- Running Model: %s (slug=%s) ---", spec.display_name, slug)

    model_dir = ensure_dir(artifacts_root / slug)
    onnx_path_fp32 = model_dir / "network.onnx"

    result = ModelRunResult(
        display_name=spec.display_name,
        slug=slug,
        logrows=None,
        prove_times_s=[],
        verify_times_s=[],
    )

    # IMPORTANT: global split flag should not force splitting models that are known to be unsafe to split.
    # Only split if:
    #   - global split flag is enabled, AND
    #   - model spec explicitly enables splitting.
    if split_onnx is None:
        split_enabled = bool(spec.enable_onnx_split)
    else:
        split_enabled = bool(split_onnx) and bool(spec.enable_onnx_split)

    execution_mode = (os.environ.get("EZKL_EXECUTION_MODE") or "exact").strip().lower()
    auto_skip_mock = bool(split_enabled) and execution_mode == "probabilistic"
    requested_skip_mock = bool(skip_mock)
    effective_skip_mock = requested_skip_mock or auto_skip_mock
    if auto_skip_mock and not requested_skip_mock:
        logger.info(
            "Forcing skip_mock=True for split+probabilistic run (model=%s, execution_mode=%s)",
            spec.display_name,
            execution_mode,
        )

    result.diagnostics["skip_mock"] = {
        "requested": requested_skip_mock,
        "effective": effective_skip_mock,
        "forced_split_probabilistic": bool(auto_skip_mock and not requested_skip_mock),
    }

    try:
        result.diagnostics["system_start"] = get_system_stats([Path("/app"), artifacts_root, cache_root])
        logger.info(
            "System stats (start): mem_available_mb=%.1f swap_free_mb=%.1f disk_free_mb(/app)=%.1f vmrss_mb=%.1f vmhwm_mb=%.1f",
            float(result.diagnostics["system_start"].get("mem_available_mb") or 0.0),
            float(result.diagnostics["system_start"].get("swap_free_mb") or 0.0),
            float(result.diagnostics["system_start"].get("disk", {}).get("/app", {}).get("disk_free_mb") or 0.0),
            float(result.diagnostics["system_start"].get("process", {}).get("vmrss_mb") or 0.0),
            float(result.diagnostics["system_start"].get("process", {}).get("vmhwm_mb") or 0.0),
        )

        if spec.precision:
            precision_mode = spec.precision.strip().lower()
            logger.info("Precision override from ModelSpec: %s", precision_mode)
        else:
            precision_mode = (os.environ.get("EZKL_ONNX_PRECISION") or "fp16").strip().lower()

        result.diagnostics["onnx_precision_requested"] = precision_mode

        logger.info(
            "Runtime env: ENABLE_ICICLE_GPU=%s ICICLE_SMALL_K=%s SPLIT_ONNX(global)=%s SPLIT_ONNX(actual)=%s "
            "SPLIT_MIN_PARAMS=%s SPLIT_MAX_NODES=%s EZKL_ONNX_PRECISION=%s EZKL_CHECK_MODE=%s EZKL_LOOKUP_SAFETY_MARGIN=%s "
            "EZKL_EXECUTION_MODE=%s EZKL_PROB_K=%s EZKL_PROB_OPS=%s EZKL_PROB_SEED_MODE=%s "
            "skip_mock(requested/effective)=%s/%s skip_verify=%s",
            os.environ.get("ENABLE_ICICLE_GPU"),
            os.environ.get("ICICLE_SMALL_K"),
            split_onnx,
            split_enabled,
            split_min_params,
            os.environ.get("SPLIT_MAX_NODES"),
            precision_mode,
            os.environ.get("EZKL_CHECK_MODE"),
            os.environ.get("EZKL_LOOKUP_SAFETY_MARGIN"),
            os.environ.get("EZKL_EXECUTION_MODE"),
            os.environ.get("EZKL_PROB_K"),
            os.environ.get("EZKL_PROB_OPS"),
            os.environ.get("EZKL_PROB_SEED_MODE"),
            requested_skip_mock,
            effective_skip_mock,
            skip_verify,
        )

        # 1) Build model + export ONNX (fp32)
        with timed("Model build + ONNX export"):
            try:
                model, dummy = spec.factory()
            except Exception as e:
                if check_safetensor_error(e):
                    logger.error("!!! SafetensorError detected during model creation !!!")
                    logger.error("This usually means a corrupt / partial safetensors download.")
                    logger.error("We retry + purge cache in timm_weights.py; if this persists, check network + cache volume.")
                raise

            _export_onnx(model, dummy, onnx_path_fp32)

            # Post-export sanitize: shape/type inference + optional Gemm rewrite for tract stability.
            sanitize_exported_onnx_inplace(
                onnx_path_fp32,
                rewrite_gemm=bool(spec.rewrite_gemm),
            )

        # Keep numpy input for ONNXRuntime splitting flow
        dummy_np = dummy.detach().cpu().numpy()
        result.diagnostics["model_input_shape"] = [int(x) for x in dummy.shape]

        # Free model memory ASAP before Rust-heavy ezkl steps
        try:
            del model
        except Exception:
            pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 2) Prepare ONNX to use in EZKL pipeline.
        #    - Non-split mode: convert whole model.
        #    - Split mode: split fp32 first, then apply precision per-segment (keep_io_types=True).
        result.diagnostics["onnx_precision"] = {
            "requested": precision_mode,
            "onnx_fp32": _path_stats(onnx_path_fp32),
        }

        current_split_tuning = SplitTuning(
            min_params=_effective_split_min_params(spec, split_min_params),
            max_nodes=_effective_split_max_nodes(spec),
            max_logrows=_effective_max_segment_logrows(spec),
            max_rows=_effective_max_segment_rows(spec),
            max_assignments=_effective_max_segment_assignments(spec),
        )
        split_retry_budget = _effective_split_retry_budget(spec) if split_enabled else 0
        result.diagnostics["split_policy"] = {
            "enabled": split_enabled,
            "initial": {
                "min_params": current_split_tuning.min_params,
                "max_nodes": current_split_tuning.max_nodes,
                "max_logrows": current_split_tuning.max_logrows,
                "max_rows": current_split_tuning.max_rows,
                "max_assignments": current_split_tuning.max_assignments,
            },
            "retry_budget": split_retry_budget,
        }
        result.diagnostics["split_attempts"] = []

        split_attempt_idx = 0
        while True:
            attempt_meta: Dict[str, Any] = {
                "attempt": split_attempt_idx,
                "min_params": current_split_tuning.min_params,
                "max_nodes": current_split_tuning.max_nodes,
                "max_logrows_cap": current_split_tuning.max_logrows,
                "max_rows_cap": current_split_tuning.max_rows,
                "max_assignments_cap": current_split_tuning.max_assignments,
            }
            result.diagnostics["split_attempts"].append(attempt_meta)
            logger.info(
                "Split attempt %s for model=%s: min_params=%s max_nodes=%s max_segment_logrows=%s max_segment_rows=%s max_segment_assignments=%s",
                split_attempt_idx + 1,
                spec.key,
                current_split_tuning.min_params,
                current_split_tuning.max_nodes,
                current_split_tuning.max_logrows,
                current_split_tuning.max_rows,
                current_split_tuning.max_assignments,
            )

            segments_fp32: List[SplitSegment] = []
            segments_ezkl: List[SplitSegment] = []

            if split_enabled:
                split_dir_fp32 = ensure_dir(model_dir / "onnx_split_fp32")
                segments_fp32 = split_onnx_model(
                    onnx_path=onnx_path_fp32,
                    out_dir=split_dir_fp32,
                    min_params_per_segment=int(current_split_tuning.min_params),
                    max_nodes_per_segment=current_split_tuning.max_nodes,
                    force=split_attempt_idx > 0,
                )

                split_dir_ezkl = ensure_dir(model_dir / "onnx_split")
                segments_ezkl, seg_prec_meta = _apply_precision_to_segments(
                    segments_fp32=segments_fp32,
                    out_dir=split_dir_ezkl,
                    precision_mode=precision_mode,
                )
                result.diagnostics["split_segment_precision"] = seg_prec_meta
            else:
                precision_dir = ensure_dir(model_dir / "_onnx_precision")
                onnx_path_used, prec_meta = maybe_convert_onnx_precision(
                    onnx_path_fp32, out_dir=precision_dir, mode=precision_mode
                )
                sanitize_exported_onnx_inplace(Path(onnx_path_used), rewrite_gemm=False)

                result.diagnostics["onnx_precision"].update(
                    {
                        "applied": prec_meta.applied,
                        "ok": prec_meta.ok,
                        "error": prec_meta.error,
                        "details": prec_meta.details or {},
                        "onnx_used": _path_stats(Path(onnx_path_used)),
                    }
                )

                segments_ezkl = [
                    SplitSegment(
                        idx=0,
                        onnx_path=str(onnx_path_used),
                        input_names=["input"],
                        output_names=["output"],
                        node_count=0,
                        param_count=0,
                    )
                ]
                segments_fp32 = segments_ezkl

            seg_inputs_payloads: List[Dict[str, Any]] = []
            seg_meta_inputs: List[Dict[str, Any]] = []

            if len(segments_fp32) == 1 and not split_enabled:
                payload = _make_ezkl_input_json_from_torch(dummy)
                seg_inputs_payloads.append(payload)
                seg_meta_inputs.append({"mode": "single", "inputs": ["input"]})
            else:
                logger.info(
                    "Preparing split-segment inputs via ONNXRuntime (%s segments, fp32 segments)...",
                    len(segments_fp32),
                )
                values: Dict[str, Any] = {"input": dummy_np}
                for seg in segments_fp32:
                    seg_path = Path(seg.onnx_path)
                    needed = list(seg.input_names)
                    missing = [n for n in needed if n not in values]
                    if missing:
                        raise RuntimeError(f"Split input preparation failed: segment {seg.idx} missing inputs {missing}")

                    ordered_inputs = [values[name] for name in needed]
                    payload = _make_ezkl_input_json_from_numpy_list(ordered_inputs)
                    seg_inputs_payloads.append(payload)
                    seg_meta_inputs.append(
                        {
                            "segment_idx": seg.idx,
                            "onnx": str(seg_path),
                            "input_names": needed,
                            "input_elems": [int(getattr(v, "size", 0)) for v in ordered_inputs],
                            "input_count": len(payload.get("input_data", [])),
                        }
                    )

                    out_map = run_onnx_segment(
                        seg_path,
                        {k: values[k] for k in needed},
                        output_names=list(seg.output_names),
                    )
                    for k, v in out_map.items():
                        values[k] = v

            result.diagnostics["split_input_prep"] = seg_meta_inputs
            result.diagnostics["segments"] = []

            total_setup = 0.0
            total_compile = 0.0
            total_calibrate = 0.0
            max_logrows: Optional[int] = None

            prove_sums = [0.0 for _ in range(repeats)]
            verify_sums = [0.0 for _ in range(repeats)]
            cached_segments = 0

            if len(segments_ezkl) != len(seg_inputs_payloads):
                raise RuntimeError(
                    f"Internal error: segments_ezkl={len(segments_ezkl)} but seg_inputs_payloads={len(seg_inputs_payloads)}"
                )

            try:
                for seg_pos, seg in enumerate(segments_ezkl):
                    seg_path = Path(seg.onnx_path)
                    seg_work_dir = ensure_dir(model_dir / "segments" / f"seg_{seg.idx:03d}")
                    logger.info(
                        "Running EZKL segment %s/%s: onnx=%s params=%.3fM nodes=%s work_dir=%s",
                        seg.idx + 1,
                        len(segments_ezkl),
                        seg_path,
                        float(seg.param_count) / 1_000_000.0,
                        seg.node_count,
                        seg_work_dir,
                    )

                    seg_meta, seg_prove, seg_verify = _run_single_onnx_pipeline(
                        spec=spec,
                        segment_pos=seg_pos,
                        segment_idx=seg.idx,
                        segment_count=len(segments_ezkl),
                        onnx_path=seg_path,
                        input_json_payload=seg_inputs_payloads[seg_pos],
                        work_dir=seg_work_dir,
                        artifacts_root=artifacts_root,
                        repeats=repeats,
                        warmup=warmup,
                        explicit_logrows=explicit_logrows,
                        split_tuning=current_split_tuning if split_enabled else None,
                        skip_verify=skip_verify,
                        skip_mock=effective_skip_mock,
                    )

                    seg_meta["segment_idx"] = seg.idx
                    seg_meta["segment_param_count"] = seg.param_count
                    seg_meta["segment_node_count"] = seg.node_count
                    result.diagnostics["segments"].append(seg_meta)
                    if isinstance(seg_meta.get("cache"), dict) and seg_meta["cache"].get("status") == "hit":
                        cached_segments += 1

                    total_setup += float(seg_meta.get("setup_time_s") or 0.0)
                    total_compile += float(seg_meta.get("compile_time_s") or 0.0)
                    total_calibrate += float(seg_meta.get("calibrate_time_s") or 0.0)

                    seg_logrows = seg_meta.get("logrows")
                    if seg_logrows is not None:
                        max_logrows = seg_logrows if max_logrows is None else max(max_logrows, int(seg_logrows))

                    for i in range(repeats):
                        prove_sums[i] += float(seg_prove[i])
                        verify_sums[i] += float(seg_verify[i])
            except SegmentNeedsResplitError as e:
                attempt_meta["cached_segments"] = cached_segments
                attempt_meta["failure"] = {
                    "segment_idx": e.segment_idx,
                    "stage": e.stage,
                    "reason": e.reason,
                    "logrows": e.logrows,
                }
                next_tuning = _tighten_split_tuning(current_split_tuning)
                if (not split_enabled) or split_attempt_idx >= split_retry_budget or next_tuning is None:
                    attempt_meta["result"] = "failed"
                    raise

                attempt_meta["result"] = "retry"
                attempt_meta["next_tuning"] = {
                    "min_params": next_tuning.min_params,
                    "max_nodes": next_tuning.max_nodes,
                    "max_logrows_cap": next_tuning.max_logrows,
                    "max_rows_cap": next_tuning.max_rows,
                    "max_assignments_cap": next_tuning.max_assignments,
                }
                logger.warning(
                    "Retrying model=%s after oversized segment %s at stage=%s: %s. "
                    "Next split policy: min_params=%s max_nodes=%s max_segment_logrows=%s max_segment_rows=%s max_segment_assignments=%s",
                    spec.key,
                    e.segment_idx,
                    e.stage,
                    e.reason,
                    next_tuning.min_params,
                    next_tuning.max_nodes,
                    next_tuning.max_logrows,
                    next_tuning.max_rows,
                    next_tuning.max_assignments,
                )
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                current_split_tuning = next_tuning
                split_attempt_idx += 1
                continue

            attempt_meta["result"] = "ok"
            attempt_meta["segment_count"] = len(segments_ezkl)
            attempt_meta["cached_segments"] = cached_segments
            attempt_meta["observed_max_logrows"] = max_logrows

            result.setup_time_s = total_setup
            result.compile_time_s = total_compile
            result.calibrate_time_s = total_calibrate
            result.logrows = max_logrows
            result.prove_times_s = [float(x) for x in prove_sums]
            result.verify_times_s = [float(x) for x in verify_sums]

            result.diagnostics["system_end"] = get_system_stats([Path("/app"), artifacts_root, cache_root])
            return result

    except Exception as e:
        logger.error("Error running model %s: %s", spec.display_name, e)
        logger.error(traceback.format_exc())

        # Extra ONNX debug specifically for the kinds of failures we saw on 2026-01-08.
        try:
            onnx_debug_dump_for_exception(
                exc=e,
                candidate_onnx_paths=[
                    onnx_path_fp32,
                    model_dir / "_onnx_precision" / "network.fp16.onnx",
                    model_dir / "_onnx_precision" / "network.int8.onnx",
                ],
            )
        except Exception:
            logger.debug("onnx_debug_dump_for_exception failed (ignored)")

        logger.error("--- Dumping relevant files for debugging ---")
        split_manifest = model_dir / "onnx_split" / "split_manifest.json"
        if split_manifest.exists():
            cat_file(split_manifest, max_lines=200, header="SPLIT MANIFEST (EZKL)")
        split_manifest_fp32 = model_dir / "onnx_split_fp32" / "split_manifest.json"
        if split_manifest_fp32.exists():
            cat_file(split_manifest_fp32, max_lines=200, header="SPLIT MANIFEST (FP32/ORT)")
        first_settings = model_dir / "segments" / "seg_000" / "settings.json"
        if first_settings.exists():
            cat_file(first_settings, max_lines=200, header="SEG_000 SETTINGS")

        result.error = f"{type(e).__name__}: {e}"
        result.traceback = traceback.format_exc()
        result.diagnostics["system_error"] = get_system_stats([Path("/app"), artifacts_root, cache_root])
        result.diagnostics["files"] = {"onnx_fp32": _path_stats(onnx_path_fp32)}
        return result
