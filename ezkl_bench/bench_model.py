from __future__ import annotations

import asyncio
import gc
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
from .utils import (
    cat_file,
    ensure_dir,
    env_flag,
    get_system_stats,
    read_json,
    slugify,
    timed,
    write_json_compact,
)

logger = logging.getLogger(__name__)


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

    return out


def _mk_run_args(spec: ModelSpec, explicit_logrows: Optional[int]) -> Any:
    import ezkl

    run_args_cls = getattr(ezkl, "PyRunArgs", None) or getattr(ezkl, "RunArgs", None)
    if run_args_cls is None:
        raise RuntimeError("ezkl python API missing PyRunArgs/RunArgs; incompatible ezkl package installed?")

    run_args = run_args_cls()

    # Visibility enums changed across versions; support both.
    if hasattr(ezkl, "PyVisibility"):
        try:
            run_args.input_visibility = ezkl.PyVisibility.Private
            run_args.output_visibility = ezkl.PyVisibility.Public
            run_args.param_visibility = ezkl.PyVisibility.Fixed
        except Exception:
            run_args.input_visibility = "private"
            run_args.output_visibility = "public"
            run_args.param_visibility = "fixed"
    else:
        run_args.input_visibility = "private"
        run_args.output_visibility = "public"
        run_args.param_visibility = "fixed"

    # --- performance/defensiveness knobs (best-effort) ---
    check_mode = (os.environ.get("EZKL_CHECK_MODE") or "safe").strip().lower()
    if hasattr(run_args, "check_mode"):
        run_args.check_mode = check_mode

    ignore_io_range = env_flag("EZKL_IGNORE_RANGE_CHECK_IO", default=(check_mode == "unsafe"))
    if hasattr(run_args, "ignore_range_check_inputs_outputs"):
        run_args.ignore_range_check_inputs_outputs = bool(ignore_io_range)

    run_args.input_scale = int(spec.input_scale)
    run_args.param_scale = int(spec.param_scale)
    run_args.num_inner_cols = int(spec.num_inner_cols)

    if explicit_logrows is not None:
        run_args.logrows = int(explicit_logrows)

    # Note: probabilistic execution parameters are passed via ezkl.gen_settings kwargs
    # (execution_mode / prob_k / prob_ops) instead of trying to mutate RunArgs here.

    return run_args


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
    onnx_path: Path,
    input_json_payload: Dict[str, Any],
    work_dir: Path,
    artifacts_root: Path,
    repeats: int,
    warmup: int,
    explicit_logrows: Optional[int] = None,
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

    logger.info("Generating settings (work_dir=%s)...", work_dir)
    run_args = _mk_run_args(spec=spec, explicit_logrows=explicit_logrows)

    logger.info(
        "RunArgs: input_scale=%s param_scale=%s num_inner_cols=%s explicit_logrows=%s "
        "check_mode=%s ignore_range_check_inputs_outputs=%s prob_overrides=%s",
        getattr(run_args, "input_scale", None),
        getattr(run_args, "param_scale", None),
        getattr(run_args, "num_inner_cols", None),
        explicit_logrows,
        getattr(run_args, "check_mode", None),
        getattr(run_args, "ignore_range_check_inputs_outputs", None),
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
    seg_meta["settings"] = _path_stats(settings_path)

    logger.info("Calibrating settings (work_dir=%s)...", work_dir)
    calib_knobs = _calibration_knobs_from_env()
    seg_meta["calibration_knobs"] = dict(calib_knobs)

    with timed("calibrate_settings") as t:

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

    seg_meta["calibrate_time_s"] = float(t["elapsed"] or 0.0)
    if not ok:
        raise RuntimeError("ezkl.calibrate_settings returned false")

    settings = read_json(settings_path)
    seg_logrows = int(settings["run_args"]["logrows"])
    seg_meta["logrows"] = seg_logrows
    logger.info("Calibration done. Selected logrows=%s (work_dir=%s)", seg_logrows, work_dir)

    logger.info("Compiling circuit (work_dir=%s)...", work_dir)
    with timed("compile_circuit") as t:

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

    seg_meta["compile_time_s"] = float(t["elapsed"] or 0.0)
    if not ok:
        raise RuntimeError("ezkl.compile_circuit returned false")
    seg_meta["compiled"] = _path_stats(compiled_path)

    srs_path = srs_cache_dir / f"k{seg_logrows}.srs"
    if not srs_path.exists():
        logger.info("Downloading SRS for k=%s (work_dir=%s)...", seg_logrows, work_dir)
        with timed("get_srs") as t:
            ok = run_ezkl_safe(ezkl.get_srs, settings_path=str(settings_path), srs_path=str(srs_path))
        seg_meta["get_srs_time_s"] = float(t["elapsed"] or 0.0)
        if not ok:
            raise RuntimeError("ezkl.get_srs returned false")
        logger.info("SRS downloaded: %s", _path_stats(srs_path))
    else:
        logger.info("Using cached SRS: %s", _path_stats(srs_path))

    logger.info("Running setup (pk/vk gen) (work_dir=%s)...", work_dir)
    with timed("setup") as t:
        ok = run_ezkl_safe(
            ezkl.setup,
            model=str(compiled_path),
            vk_path=str(vk_path),
            pk_path=str(pk_path),
            srs_path=str(srs_path),
        )
    seg_meta["setup_time_s"] = float(t["elapsed"] or 0.0)
    if not ok:
        raise RuntimeError("ezkl.setup returned false")
    seg_meta["vk"] = _path_stats(vk_path)
    seg_meta["pk"] = _path_stats(pk_path)

    logger.info("Generating witness (work_dir=%s)...", work_dir)
    with timed("gen_witness") as t:
        ok = run_ezkl_safe(
            ezkl.gen_witness,
            data=str(input_path),
            model=str(compiled_path),
            output=str(witness_path),
            vk_path=str(vk_path),
            srs_path=str(srs_path),
        )
    seg_meta["gen_witness_time_s"] = float(t["elapsed"] or 0.0)
    if not ok:
        raise RuntimeError("ezkl.gen_witness returned false")
    seg_meta["witness"] = _path_stats(witness_path)

    logger.info("Running mock (work_dir=%s)...", work_dir)
    with timed("mock") as t:

        def _do_mock():
            try:
                ezkl.mock(witness=str(witness_path), model=str(compiled_path))
            except TypeError:
                ezkl.mock(str(witness_path), str(compiled_path))

        run_ezkl_safe(_do_mock)

    seg_meta["mock_time_s"] = float(t["elapsed"] or 0.0)

    logger.info("Warming up (%s iters) (work_dir=%s)...", warmup, work_dir)
    for wi in range(max(0, warmup)):
        logger.info("Warmup iteration %s/%s (work_dir=%s)", wi + 1, warmup, work_dir)
        ok = run_ezkl_safe(
            ezkl.prove,
            witness=str(witness_path),
            model=str(compiled_path),
            pk_path=str(pk_path),
            proof_path=str(proof_path),
            srs_path=str(srs_path),
        )
        if not ok:
            raise RuntimeError("Warmup prove failed")

        ok = run_ezkl_safe(
            ezkl.verify,
            proof_path=str(proof_path),
            settings_path=str(settings_path),
            vk_path=str(vk_path),
            srs_path=str(srs_path),
        )
        if not ok:
            raise RuntimeError("Warmup verify failed")

    logger.info("Starting %s measured runs (work_dir=%s)...", repeats, work_dir)
    for i in range(repeats):
        with timed(f"prove(run {i+1})") as tprove:
            ok = run_ezkl_safe(
                ezkl.prove,
                witness=str(witness_path),
                model=str(compiled_path),
                pk_path=str(pk_path),
                proof_path=str(proof_path),
                srs_path=str(srs_path),
            )
        if not ok:
            raise RuntimeError("ezkl.prove returned false")
        prove_times.append(float(tprove["elapsed"] or 0.0))

        with timed(f"verify(run {i+1})") as tver:
            ok = run_ezkl_safe(
                ezkl.verify,
                proof_path=str(proof_path),
                settings_path=str(settings_path),
                vk_path=str(vk_path),
                srs_path=str(srs_path),
            )
        if not ok:
            raise RuntimeError("ezkl.verify returned false")
        verify_times.append(float(tver["elapsed"] or 0.0))

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

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
            "SPLIT_MIN_PARAMS=%s EZKL_ONNX_PRECISION=%s EZKL_CHECK_MODE=%s EZKL_LOOKUP_SAFETY_MARGIN=%s "
            "EZKL_EXECUTION_MODE=%s EZKL_PROB_K=%s EZKL_PROB_OPS=%s",
            os.environ.get("ENABLE_ICICLE_GPU"),
            os.environ.get("ICICLE_SMALL_K"),
            split_onnx,
            split_enabled,
            split_min_params,
            precision_mode,
            os.environ.get("EZKL_CHECK_MODE"),
            os.environ.get("EZKL_LOOKUP_SAFETY_MARGIN"),
            os.environ.get("EZKL_EXECUTION_MODE"),
            os.environ.get("EZKL_PROB_K"),
            os.environ.get("EZKL_PROB_OPS"),
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

        segments_fp32: List[SplitSegment] = []
        segments_ezkl: List[SplitSegment] = []

        if split_enabled:
            split_dir_fp32 = ensure_dir(model_dir / "onnx_split_fp32")
            segments_fp32 = split_onnx_model(
                onnx_path=onnx_path_fp32,
                out_dir=split_dir_fp32,
                min_params_per_segment=int(split_min_params),
                force=False,
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
            # For non-split flow, also sanitize converted model (helps future debugging and tract stability).
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

        # 3) Materialize per-segment input payloads using ONNXRuntime (always on fp32 segments).
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
                seg_inputs_payloads.append(_make_ezkl_input_json_from_numpy_list(ordered_inputs))
                seg_meta_inputs.append(
                    {
                        "segment_idx": seg.idx,
                        "onnx": str(seg_path),
                        "input_names": needed,
                        "input_elems": [int(getattr(v, "size", 0)) for v in ordered_inputs],
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

        # 4) Run ezkl pipeline for each segment and aggregate.
        result.diagnostics["segments"] = []
        total_setup = 0.0
        total_compile = 0.0
        total_calibrate = 0.0
        max_logrows: Optional[int] = None

        prove_sums = [0.0 for _ in range(repeats)]
        verify_sums = [0.0 for _ in range(repeats)]

        if len(segments_ezkl) != len(seg_inputs_payloads):
            raise RuntimeError(
                f"Internal error: segments_ezkl={len(segments_ezkl)} but seg_inputs_payloads={len(seg_inputs_payloads)}"
            )

        for seg in segments_ezkl:
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
                onnx_path=seg_path,
                input_json_payload=seg_inputs_payloads[seg.idx],
                work_dir=seg_work_dir,
                artifacts_root=artifacts_root,
                repeats=repeats,
                warmup=warmup,
                explicit_logrows=explicit_logrows,
            )

            seg_meta["segment_idx"] = seg.idx
            seg_meta["segment_param_count"] = seg.param_count
            seg_meta["segment_node_count"] = seg.node_count
            result.diagnostics["segments"].append(seg_meta)

            total_setup += float(seg_meta.get("setup_time_s") or 0.0)
            total_compile += float(seg_meta.get("compile_time_s") or 0.0)
            total_calibrate += float(seg_meta.get("calibrate_time_s") or 0.0)

            seg_logrows = seg_meta.get("logrows")
            if seg_logrows is not None:
                max_logrows = seg_logrows if max_logrows is None else max(max_logrows, int(seg_logrows))

            for i in range(repeats):
                prove_sums[i] += float(seg_prove[i])
                verify_sums[i] += float(seg_verify[i])

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
