#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import csv
import logging
import os
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ezkl_bench.utils import ensure_dir, setup_logging, write_json


logger = logging.getLogger(__name__)
DEFAULT_FULL_CIRCUIT_MODELS = ["lenet-5-small"]


def _mean(values: Iterable[float]) -> Optional[float]:
    vals = [float(v) for v in values]
    if not vals:
        return None
    return sum(vals) / float(len(vals))


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


def _csv_list(raw: str) -> List[str]:
    return [item.strip() for item in str(raw or "").split(",") if item.strip()]


def _resolve_scoped_models(selected_models: List[str], requested_models: Optional[List[str]], *, default: List[str]) -> List[str]:
    if requested_models is None:
        requested = list(default)
    else:
        requested = [str(x).strip() for x in requested_models if str(x).strip()]

    if not requested:
        return []
    if len(requested) == 1 and requested[0].lower() == "all":
        return list(selected_models)
    if len(requested) == 1 and requested[0].lower() == "none":
        return []

    selected_set = set(selected_models)
    out: List[str] = []
    ignored: List[str] = []
    for name in requested:
        if name in selected_set:
            out.append(name)
        else:
            ignored.append(name)
    if ignored:
        logger.warning("Ignoring scoped models not present in --models: %s", ignored)
    return out


def _segment_metric_sum(diagnostics: Dict[str, Any], key: str) -> float:
    total = 0.0
    for segment in diagnostics.get("segments") or []:
        if not isinstance(segment, dict):
            continue
        value = segment.get(key)
        if value is None:
            continue
        try:
            total += float(value)
        except Exception:
            continue
    return total


def _full_circuit_stage_summary(result: ModelRunResult) -> Dict[str, float]:
    diagnostics = result.diagnostics or {}
    return {
        "gen_settings_s": _segment_metric_sum(diagnostics, "gen_settings_time_s"),
        "calibrate_s": float(result.calibrate_time_s or 0.0),
        "compile_s": float(result.compile_time_s or 0.0),
        "get_srs_s": _segment_metric_sum(diagnostics, "get_srs_time_s"),
        "setup_s": float(result.setup_time_s or 0.0),
        "witness_s": _segment_metric_sum(diagnostics, "gen_witness_time_s"),
        "mock_s": _segment_metric_sum(diagnostics, "mock_time_s"),
        "prove_mean_s": float(_mean(result.prove_times_s) or 0.0),
        "verify_mean_s": float(_mean(result.verify_times_s) or 0.0),
        "prove_total_s": float(sum(float(x) for x in result.prove_times_s)),
        "verify_total_s": float(sum(float(x) for x in result.verify_times_s)),
    }


def _module_stage_summary(seg_meta: Dict[str, Any], prove_times: List[float], verify_times: List[float]) -> Dict[str, float]:
    return {
        "gen_settings_s": float(seg_meta.get("gen_settings_time_s") or 0.0),
        "calibrate_s": float(seg_meta.get("calibrate_time_s") or 0.0),
        "compile_s": float(seg_meta.get("compile_time_s") or 0.0),
        "get_srs_s": float(seg_meta.get("get_srs_time_s") or 0.0),
        "setup_s": float(seg_meta.get("setup_time_s") or 0.0),
        "witness_s": float(seg_meta.get("gen_witness_time_s") or 0.0),
        "mock_s": float(seg_meta.get("mock_time_s") or 0.0),
        "prove_mean_s": float(_mean(prove_times) or 0.0),
        "verify_mean_s": float(_mean(verify_times) or 0.0),
        "prove_total_s": float(sum(float(x) for x in prove_times)),
        "verify_total_s": float(sum(float(x) for x in verify_times)),
    }


def _aggregate_adapter_stage_totals(modules: List[Dict[str, Any]]) -> Dict[str, float]:
    keys = [
        "gen_settings_s",
        "calibrate_s",
        "compile_s",
        "get_srs_s",
        "setup_s",
        "witness_s",
        "mock_s",
        "prove_mean_s",
        "verify_mean_s",
        "prove_total_s",
        "verify_total_s",
    ]
    totals = {key: 0.0 for key in keys}
    for module in modules:
        stage_summary = module.get("stage_summary") or {}
        for key in keys:
            totals[key] += float(stage_summary.get(key) or 0.0)
    return totals


def _adapter_progress_summary(modules: List[Dict[str, Any]], *, selected_count: int) -> Dict[str, int]:
    completed = 0
    failed = 0
    for module in modules:
        status = str(module.get("status") or "").strip().lower()
        if status == "failed" or module.get("error"):
            failed += 1
        elif status in {"ok", "capture_only"} or (module.get("stage_summary") or {}):
            completed += 1

    return {
        "selected_module_count": int(selected_count),
        "completed_module_count": int(completed),
        "failed_module_count": int(failed),
        "remaining_module_count": max(int(selected_count) - int(completed) - int(failed), 0),
    }


def _make_adapter_spec(
    *,
    model_name: str,
    module_name: str,
    input_scale: int,
    param_scale: int,
    num_inner_cols: int,
) -> ModelSpec:
    from ezkl_bench.models import ModelSpec

    display_name = f"{model_name}::{module_name}"
    return ModelSpec(
        key=f"{model_name}-adapter-{module_name}",
        display_name=display_name,
        factory=lambda: (None, None),
        input_scale=int(input_scale),
        param_scale=int(param_scale),
        num_inner_cols=int(num_inner_cols),
        enable_onnx_split=False,
        precision="fp32",
    )


def _env_overrides_for_mode(
    *,
    execution_mode: str,
    prob_k: int,
    prob_ops: List[str],
    prob_seed_mode: str,
    input_visibility: str,
    param_visibility: str,
) -> Dict[str, Optional[str]]:
    mode = str(execution_mode).strip().lower()
    if mode not in {"exact", "probabilistic"}:
        raise ValueError(f"Unsupported execution mode: {execution_mode}")

    return {
        "EZKL_EXECUTION_MODE": mode,
        "EZKL_PROB_K": str(int(prob_k)) if mode == "probabilistic" else None,
        "EZKL_PROB_OPS": ",".join(prob_ops) if mode == "probabilistic" else None,
        "EZKL_PROB_SEED_MODE": str(prob_seed_mode) if mode == "probabilistic" else None,
        "EZKL_INPUT_VISIBILITY": str(input_visibility),
        "EZKL_PARAM_VISIBILITY": str(param_visibility),
    }


def _run_full_circuit_benchmark(
    *,
    model_name: str,
    spec: ModelSpec,
    artifacts_root: Path,
    cache_root: Path,
    repeats: int,
    warmup: int,
    split_onnx: bool,
    split_min_params: int,
    skip_verify: bool,
    skip_mock: bool,
    execution_mode: str,
    prob_k: int,
    prob_ops: List[str],
    prob_seed_mode: str,
) -> Dict[str, Any]:
    from ezkl_bench.bench_model import run_single_model

    env_overrides = _env_overrides_for_mode(
        execution_mode=execution_mode,
        prob_k=prob_k,
        prob_ops=prob_ops,
        prob_seed_mode=prob_seed_mode,
        input_visibility="private",
        param_visibility="fixed",
    )
    with _temporary_env(env_overrides):
        result = run_single_model(
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

    payload = asdict(result)
    payload["stage_summary"] = _full_circuit_stage_summary(result)
    payload["model_name"] = model_name
    payload["status"] = "failed" if payload.get("error") else "ok"
    return payload


def _run_adapter_benchmark(
    *,
    model_name: str,
    spec: ModelSpec,
    out_dir: Path,
    repeats: int,
    warmup: int,
    rank: int,
    alpha: float,
    max_modules: int,
    min_base_params: int,
    module_types: List[str],
    skip_verify: bool,
    skip_mock: bool,
    capture_only: bool,
    execution_mode: str,
    prob_k: int,
    prob_ops: List[str],
    prob_seed_mode: str,
    adapter_input_visibility: str,
    adapter_param_visibility: str,
    adapter_input_scale: int,
    adapter_param_scale: int,
    adapter_num_inner_cols: int,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    from ezkl_bench.bench_model import _run_single_onnx_pipeline
    from ezkl_bench.vision_adapters import (
        adapter_coverage_summary,
        capture_candidate_inputs,
        discover_adapter_candidates,
        export_adapter_artifacts,
    )

    model, dummy_input = spec.factory()
    model.eval().cpu()

    selected, skipped = discover_adapter_candidates(
        model,
        rank=rank,
        alpha=alpha,
        min_base_params=min_base_params,
        max_modules=max_modules,
        include_module_types=module_types,
    )
    captured = capture_candidate_inputs(model, dummy_input, selected)
    coverage = adapter_coverage_summary(model=model, selected=selected, skipped=skipped)

    model_dir = ensure_dir(out_dir / model_name)
    module_artifacts_root = ensure_dir(model_dir / "modules")

    modules_payload: List[Dict[str, Any]] = []
    result_payload: Dict[str, Any] = {
        "model_name": model_name,
        "display_name": spec.display_name,
        "coverage": coverage,
        "modules": modules_payload,
        "totals": {},
        "capture_only": bool(capture_only),
        "status": "running",
        "progress": _adapter_progress_summary(modules_payload, selected_count=len(selected)),
        "config": {
            "rank": int(rank),
            "alpha": float(alpha),
            "max_modules": int(max_modules),
            "min_base_params": int(min_base_params),
            "module_types": list(module_types),
            "execution_mode": str(execution_mode),
            "prob_k": int(prob_k),
            "prob_ops": list(prob_ops),
            "prob_seed_mode": str(prob_seed_mode),
            "input_visibility": str(adapter_input_visibility),
            "param_visibility": str(adapter_param_visibility),
            "input_scale": int(adapter_input_scale),
            "param_scale": int(adapter_param_scale),
            "num_inner_cols": int(adapter_num_inner_cols),
        },
    }

    def emit_progress(*, current_module: Optional[str] = None) -> None:
        result_payload["totals"] = _aggregate_adapter_stage_totals(modules_payload)
        result_payload["progress"] = _adapter_progress_summary(modules_payload, selected_count=len(selected))
        if current_module:
            result_payload["current_module"] = str(current_module)
        else:
            result_payload.pop("current_module", None)
        if progress_callback is not None:
            progress_callback(result_payload)

    env_overrides = _env_overrides_for_mode(
        execution_mode=execution_mode,
        prob_k=prob_k,
        prob_ops=prob_ops,
        prob_seed_mode=prob_seed_mode,
        input_visibility=adapter_input_visibility,
        param_visibility=adapter_param_visibility,
    )

    emit_progress()

    for index, candidate in enumerate(selected):
        emit_progress(current_module=candidate.name)
        captured_input = captured.get(candidate.name)
        if captured_input is None:
            modules_payload.append(
                {
                    "name": candidate.name,
                    "module_type": candidate.module_type,
                    "base_param_count": int(candidate.base_param_count),
                    "adapter_param_count": int(candidate.adapter_param_count),
                    "error": "input_capture_failed",
                    "status": "failed",
                }
            )
            emit_progress()
            continue

        try:
            export_dir = ensure_dir(module_artifacts_root / f"{index:02d}_{candidate.name.replace('.', '__').replace('/', '__')}")
            export_info = export_adapter_artifacts(candidate=candidate, example_input=captured_input, out_dir=export_dir)

            module_payload = {
                "name": candidate.name,
                "module_type": candidate.module_type,
                "base_param_count": int(candidate.base_param_count),
                "adapter_param_count": int(candidate.adapter_param_count),
                "rank": int(candidate.rank),
                "alpha": float(candidate.alpha),
                "captured_input_shape": list(export_info["input_shape"]),
                "artifacts": {
                    "onnx_path": str(export_info["onnx_path"]),
                    "input_path": str(export_info["input_path"]),
                    "metadata_path": str(export_info["metadata_path"]),
                },
            }

            if capture_only:
                module_payload["stage_summary"] = {}
                module_payload["status"] = "capture_only"
                modules_payload.append(module_payload)
                emit_progress()
                continue

            adapter_spec = _make_adapter_spec(
                model_name=model_name,
                module_name=str(export_info["safe_name"]),
                input_scale=adapter_input_scale,
                param_scale=adapter_param_scale,
                num_inner_cols=adapter_num_inner_cols,
            )

            with _temporary_env(env_overrides):
                seg_meta, prove_times, verify_times = _run_single_onnx_pipeline(
                    spec=adapter_spec,
                    segment_pos=0,
                    segment_idx=index,
                    segment_count=1,
                    onnx_path=Path(export_info["onnx_path"]),
                    input_json_payload={"input_data": [[float(x) for x in captured_input.reshape(-1).tolist()]]},
                    work_dir=export_dir / "proof",
                    artifacts_root=model_dir / "proof_artifacts",
                    repeats=repeats,
                    warmup=warmup,
                    explicit_logrows=None,
                    split_tuning=None,
                    skip_verify=skip_verify,
                    skip_mock=skip_mock,
                )

            module_payload["seg_meta"] = seg_meta
            module_payload["prove_times_s"] = [float(x) for x in prove_times]
            module_payload["verify_times_s"] = [float(x) for x in verify_times]
            module_payload["stage_summary"] = _module_stage_summary(seg_meta, prove_times, verify_times)
            module_payload["status"] = "ok"
            modules_payload.append(module_payload)
            emit_progress()
        except Exception as exc:
            logger.exception("Adapter module benchmark failed for %s:%s", model_name, candidate.name)
            modules_payload.append(
                {
                    "name": candidate.name,
                    "module_type": candidate.module_type,
                    "base_param_count": int(candidate.base_param_count),
                    "adapter_param_count": int(candidate.adapter_param_count),
                    "rank": int(candidate.rank),
                    "alpha": float(candidate.alpha),
                    "status": "failed",
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )
            emit_progress()

    progress = _adapter_progress_summary(modules_payload, selected_count=len(selected))
    result_payload["progress"] = progress
    if progress["failed_module_count"] > 0:
        result_payload["status"] = "partial_failed"
    else:
        result_payload["status"] = "capture_only" if capture_only else "ok"
    emit_progress()
    return result_payload


def _write_module_csv(summary: Dict[str, Any], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model_name",
        "display_name",
        "module_name",
        "module_type",
        "status",
        "error",
        "base_param_count",
        "adapter_param_count",
        "rank",
        "gen_settings_s",
        "calibrate_s",
        "compile_s",
        "get_srs_s",
        "setup_s",
        "witness_s",
        "prove_mean_s",
        "verify_mean_s",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for model_payload in (summary.get("models") or {}).values():
            adapter_payload = model_payload.get("adapter_benchmark") or {}
            display_name = model_payload.get("display_name") or adapter_payload.get("display_name") or ""
            for module in adapter_payload.get("modules") or []:
                stage = module.get("stage_summary") or {}
                writer.writerow(
                    {
                        "model_name": model_payload.get("model_name"),
                        "display_name": display_name,
                        "module_name": module.get("name"),
                        "module_type": module.get("module_type"),
                        "status": module.get("status") or "",
                        "error": module.get("error") or "",
                        "base_param_count": module.get("base_param_count"),
                        "adapter_param_count": module.get("adapter_param_count"),
                        "rank": module.get("rank"),
                        "gen_settings_s": stage.get("gen_settings_s"),
                        "calibrate_s": stage.get("calibrate_s"),
                        "compile_s": stage.get("compile_s"),
                        "get_srs_s": stage.get("get_srs_s"),
                        "setup_s": stage.get("setup_s"),
                        "witness_s": stage.get("witness_s"),
                        "prove_mean_s": stage.get("prove_mean_s"),
                        "verify_mean_s": stage.get("verify_mean_s"),
                    }
                )


def _write_model_csv(summary: Dict[str, Any], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model_name",
        "display_name",
        "paper_full_circuit_supported",
        "full_circuit_status",
        "full_circuit_reason",
        "adapter_status",
        "full_prove_mean_s",
        "full_verify_mean_s",
        "adapter_prove_total_s",
        "adapter_verify_total_s",
        "adapter_selected_base_param_ratio",
        "adapter_selected_adapter_param_ratio",
        "adapter_selected_module_count",
        "adapter_completed_module_count",
        "adapter_failed_module_count",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for model_payload in (summary.get("models") or {}).values():
            full_payload = model_payload.get("full_circuit") or {}
            adapter_payload = model_payload.get("adapter_benchmark") or {}
            stage = full_payload.get("stage_summary") or {}
            totals = adapter_payload.get("totals") or {}
            coverage = adapter_payload.get("coverage") or {}
            progress = adapter_payload.get("progress") or {}
            writer.writerow(
                {
                    "model_name": model_payload.get("model_name"),
                    "display_name": model_payload.get("display_name"),
                    "paper_full_circuit_supported": model_payload.get("paper_full_circuit_supported"),
                    "full_circuit_status": full_payload.get("status") or "ok",
                    "full_circuit_reason": full_payload.get("reason") or full_payload.get("error") or "",
                    "adapter_status": adapter_payload.get("status") or "",
                    "full_prove_mean_s": stage.get("prove_mean_s"),
                    "full_verify_mean_s": stage.get("verify_mean_s"),
                    "adapter_prove_total_s": totals.get("prove_total_s"),
                    "adapter_verify_total_s": totals.get("verify_total_s"),
                    "adapter_selected_base_param_ratio": coverage.get("selected_base_param_ratio"),
                    "adapter_selected_adapter_param_ratio": coverage.get("selected_adapter_param_ratio"),
                    "adapter_selected_module_count": coverage.get("selected_module_count"),
                    "adapter_completed_module_count": progress.get("completed_module_count"),
                    "adapter_failed_module_count": progress.get("failed_module_count"),
                }
            )


def _write_progress_bundle(
    *,
    summary: Dict[str, Any],
    out_dir: Path,
    write_plots: bool = False,
) -> Path:
    summary["updated_at_unix"] = time.time()
    summary_path = out_dir / "vision_adapter_summary.json"
    write_json(summary_path, summary)
    _write_module_csv(summary, out_dir / "adapter_module_metrics.csv")
    _write_model_csv(summary, out_dir / "model_comparison.csv")

    if write_plots:
        try:
            from ezkl_bench.vision_adapter_plotting import plot_vision_adapter_summary

            plot_vision_adapter_summary(summary_path=summary_path, output_dir=out_dir / "plots")
        except Exception as exc:
            logger.exception("Plot generation failed: %s", exc)

    return summary_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark full-circuit EZKL proving against adapter-level vision-module proofs."
    )
    parser.add_argument("--models", nargs="+", default=["lenet-5-small", "repvgg-a0", "vit"])
    parser.add_argument("--out-dir", default="results/vision_adapter_bench")
    parser.add_argument("--cache-dir", default=".cache/ezkl_bench")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=8.0)
    parser.add_argument("--max-modules", type=int, default=8)
    parser.add_argument("--min-base-params", type=int, default=1024)
    parser.add_argument("--module-types", default="linear,conv2d")
    parser.add_argument("--split-onnx", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--split-min-params", type=int, default=50_000)
    parser.add_argument("--run-full-circuit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--run-adapters", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--full-circuit-models",
        nargs="*",
        default=None,
        help="Subset of --models that should run full-circuit benchmarking. Defaults to paper-safe set: lenet-5-small. Use 'all' or 'none' to override.",
    )
    parser.add_argument("--skip-verify", action="store_true")
    parser.add_argument("--skip-mock", action="store_true")
    parser.add_argument("--capture-only", action="store_true")
    parser.add_argument("--full-execution-mode", default="probabilistic", choices=["exact", "probabilistic"])
    parser.add_argument("--full-prob-k", type=int, default=2)
    parser.add_argument("--full-prob-ops", default="MatMul,Gemm,Conv")
    parser.add_argument("--full-prob-seed-mode", default="fiat_shamir")
    parser.add_argument("--adapter-execution-mode", default="exact", choices=["exact", "probabilistic"])
    parser.add_argument("--adapter-prob-k", type=int, default=2)
    parser.add_argument("--adapter-prob-ops", default="MatMul,Gemm,Conv")
    parser.add_argument("--adapter-prob-seed-mode", default="fiat_shamir")
    parser.add_argument("--adapter-input-visibility", default="public", choices=["public", "private"])
    parser.add_argument("--adapter-param-visibility", default="private", choices=["fixed", "private"])
    parser.add_argument("--adapter-input-scale", type=int, default=7)
    parser.add_argument("--adapter-param-scale", type=int, default=7)
    parser.add_argument("--adapter-num-inner-cols", type=int, default=2)
    parser.add_argument("--plots", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    setup_logging(args.log_level)

    out_dir = ensure_dir(Path(args.out_dir))
    cache_dir = ensure_dir(Path(args.cache_dir))
    from ezkl_bench.models import get_model_specs

    specs = get_model_specs(cache_dir)
    module_types = _csv_list(args.module_types)
    full_prob_ops = _csv_list(args.full_prob_ops)
    adapter_prob_ops = _csv_list(args.adapter_prob_ops)
    full_circuit_models = _resolve_scoped_models(
        list(args.models),
        args.full_circuit_models,
        default=DEFAULT_FULL_CIRCUIT_MODELS,
    )

    summary: Dict[str, Any] = {
        "created_at_unix": time.time(),
        "status": "running",
        "config": {
            "models": list(args.models),
            "full_circuit_models": list(full_circuit_models),
            "repeats": int(args.repeats),
            "warmup": int(args.warmup),
            "rank": int(args.rank),
            "alpha": float(args.alpha),
            "max_modules": int(args.max_modules),
            "min_base_params": int(args.min_base_params),
            "module_types": module_types,
            "run_full_circuit": bool(args.run_full_circuit),
            "run_adapters": bool(args.run_adapters),
            "capture_only": bool(args.capture_only),
        },
        "models": {},
    }

    def checkpoint(*, write_plots: bool = False) -> Path:
        return _write_progress_bundle(summary=summary, out_dir=out_dir, write_plots=write_plots)

    try:
        checkpoint()
        for model_name in args.models:
            if model_name not in specs:
                raise ValueError(f"Unknown model: {model_name}. Available: {sorted(specs.keys())}")

            spec = specs[model_name]
            model_payload: Dict[str, Any] = {
                "model_name": model_name,
                "display_name": spec.display_name,
                "paper_full_circuit_supported": bool(model_name in full_circuit_models),
            }
            summary["models"][model_name] = model_payload
            checkpoint()

            if args.run_full_circuit:
                if model_name in full_circuit_models:
                    logger.info("Running full-circuit benchmark for %s", model_name)
                    model_payload["full_circuit"] = {"status": "running"}
                    checkpoint()
                    try:
                        full_payload = _run_full_circuit_benchmark(
                            model_name=model_name,
                            spec=spec,
                            artifacts_root=ensure_dir(out_dir / "full_circuit" / "artifacts"),
                            cache_root=cache_dir,
                            repeats=args.repeats,
                            warmup=args.warmup,
                            split_onnx=bool(args.split_onnx),
                            split_min_params=int(args.split_min_params),
                            skip_verify=bool(args.skip_verify),
                            skip_mock=bool(args.skip_mock),
                            execution_mode=args.full_execution_mode,
                            prob_k=int(args.full_prob_k),
                            prob_ops=full_prob_ops,
                            prob_seed_mode=args.full_prob_seed_mode,
                        )
                        model_payload["full_circuit"] = full_payload
                    except Exception as exc:
                        logger.exception("Full-circuit benchmark failed for %s", model_name)
                        model_payload["full_circuit"] = {"error": str(exc), "status": "failed"}
                else:
                    logger.info("Skipping full-circuit benchmark for %s (outside --full-circuit-models)", model_name)
                    model_payload["full_circuit"] = {
                        "status": "skipped",
                        "reason": "full_circuit_disabled_for_model",
                    }
                checkpoint()

            if args.run_adapters:
                logger.info("Running adapter benchmark for %s", model_name)
                model_payload["adapter_benchmark"] = {
                    "status": "running",
                    "modules": [],
                    "totals": {},
                    "progress": {
                        "selected_module_count": 0,
                        "completed_module_count": 0,
                        "failed_module_count": 0,
                        "remaining_module_count": 0,
                    },
                }
                checkpoint()

                def _adapter_progress_callback(payload: Dict[str, Any], *, _model_payload: Dict[str, Any] = model_payload) -> None:
                    _model_payload["adapter_benchmark"] = payload
                    checkpoint()

                try:
                    adapter_payload = _run_adapter_benchmark(
                        model_name=model_name,
                        spec=spec,
                        out_dir=ensure_dir(out_dir / "adapter_mode"),
                        repeats=args.repeats,
                        warmup=args.warmup,
                        rank=args.rank,
                        alpha=args.alpha,
                        max_modules=args.max_modules,
                        min_base_params=args.min_base_params,
                        module_types=module_types,
                        skip_verify=bool(args.skip_verify),
                        skip_mock=bool(args.skip_mock),
                        capture_only=bool(args.capture_only),
                        execution_mode=args.adapter_execution_mode,
                        prob_k=int(args.adapter_prob_k),
                        prob_ops=adapter_prob_ops,
                        prob_seed_mode=args.adapter_prob_seed_mode,
                        adapter_input_visibility=args.adapter_input_visibility,
                        adapter_param_visibility=args.adapter_param_visibility,
                        adapter_input_scale=int(args.adapter_input_scale),
                        adapter_param_scale=int(args.adapter_param_scale),
                        adapter_num_inner_cols=int(args.adapter_num_inner_cols),
                        progress_callback=_adapter_progress_callback,
                    )
                    model_payload["adapter_benchmark"] = adapter_payload
                except Exception as exc:
                    logger.exception("Adapter benchmark failed for %s", model_name)
                    adapter_payload = dict(model_payload.get("adapter_benchmark") or {})
                    adapter_payload["error"] = str(exc)
                    adapter_payload["status"] = "failed"
                    model_payload["adapter_benchmark"] = adapter_payload
                checkpoint()

        summary["status"] = "completed"
        summary["completed_at_unix"] = time.time()
        summary_path = checkpoint(write_plots=bool(args.plots))
        logger.info("Wrote summary to %s", summary_path)
        return 0
    except KeyboardInterrupt:
        summary["status"] = "interrupted"
        summary["interrupted_at_unix"] = time.time()
        summary_path = checkpoint()
        logger.warning("Interrupted. Partial summary retained at %s", summary_path)
        return 130
    except Exception as exc:
        summary["status"] = "failed"
        summary["failed_at_unix"] = time.time()
        summary["fatal_error"] = str(exc)
        summary_path = checkpoint()
        logger.exception("Benchmark aborted. Partial summary retained at %s", summary_path)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
