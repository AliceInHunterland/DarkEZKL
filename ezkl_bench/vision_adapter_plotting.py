from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from .utils import read_json


STAGE_ORDER: List[Tuple[str, str, str]] = [
    ("gen_settings_s", "Settings", "#4C78A8"),
    ("calibrate_s", "Calibrate", "#F58518"),
    ("compile_s", "Compile", "#E45756"),
    ("get_srs_s", "SRS", "#72B7B2"),
    ("setup_s", "Setup", "#54A24B"),
    ("witness_s", "Witness", "#EECA3B"),
    ("prove_mean_s", "Prove", "#B279A2"),
    ("verify_mean_s", "Verify", "#FF9DA6"),
]


def _model_rows(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = []
    for model_name, model_payload in (payload.get("models") or {}).items():
        rows.append(
            {
                "model_name": model_name,
                "display_name": model_payload.get("display_name") or model_name,
                "full_circuit": model_payload.get("full_circuit") or {},
                "adapter_benchmark": model_payload.get("adapter_benchmark") or {},
            }
        )
    return rows


def _stage_value(model_payload: Dict[str, Any], mode_key: str, stage_key: str) -> float:
    if mode_key == "full_circuit":
        summary = (model_payload.get("full_circuit") or {}).get("stage_summary") or {}
    else:
        summary = (model_payload.get("adapter_benchmark") or {}).get("totals") or {}
    try:
        return float(summary.get(stage_key) or 0.0)
    except Exception:
        return 0.0


def _has_stage_summary(model_payload: Dict[str, Any], mode_key: str) -> bool:
    if mode_key == "full_circuit":
        summary = (model_payload.get("full_circuit") or {}).get("stage_summary") or {}
    else:
        summary = (model_payload.get("adapter_benchmark") or {}).get("totals") or {}
    return any(float(summary.get(stage_key) or 0.0) > 0.0 for stage_key, _, _ in STAGE_ORDER)


def _adapter_module_rows(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for model_name, model_payload in (payload.get("models") or {}).items():
        display_name = model_payload.get("display_name") or model_name
        adapter_payload = model_payload.get("adapter_benchmark") or {}
        for module in adapter_payload.get("modules") or []:
            stage_summary = module.get("stage_summary") or {}
            rows.append(
                {
                    "model_name": model_name,
                    "display_name": display_name,
                    "module_name": module.get("name") or "",
                    "base_param_count": float(module.get("base_param_count") or 0.0),
                    "adapter_param_count": float(module.get("adapter_param_count") or 0.0),
                    "prove_mean_s": float(stage_summary.get("prove_mean_s") or 0.0),
                    "verify_mean_s": float(stage_summary.get("verify_mean_s") or 0.0),
                }
            )
    return rows


def _coverage_rows(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for model_name, model_payload in (payload.get("models") or {}).items():
        display_name = model_payload.get("display_name") or model_name
        coverage = ((model_payload.get("adapter_benchmark") or {}).get("coverage") or {})
        rows.append(
            {
                "model_name": model_name,
                "display_name": display_name,
                "selected_base_param_count": float(coverage.get("selected_base_param_count") or 0.0),
                "selected_adapter_param_count": float(coverage.get("selected_adapter_param_count") or 0.0),
                "selected_base_param_ratio": float(coverage.get("selected_base_param_ratio") or 0.0),
                "selected_adapter_param_ratio": float(coverage.get("selected_adapter_param_ratio") or 0.0),
            }
        )
    return rows


def _save(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(output_dir / f"{stem}.{ext}", bbox_inches="tight")
    plt.close(fig)


def _set_log_scale_if_positive(ax: plt.Axes, axis: str, values: List[float]) -> None:
    if any(float(v) > 0.0 for v in values):
        if axis == "x":
            ax.set_xscale("log")
        elif axis == "y":
            ax.set_yscale("log")


def _plot_full_vs_adapter(rows: List[Dict[str, Any]], output_dir: Path) -> None:
    rows = [
        row
        for row in rows
        if _has_stage_summary(row, "full_circuit") and _has_stage_summary(row, "adapter_benchmark")
    ]
    if not rows:
        return

    x = np.arange(len(rows), dtype=float)
    width = 0.34
    modes = [("full_circuit", "Full circuit", ""), ("adapter_benchmark", "Adapter total", "//")]

    fig, ax = plt.subplots(figsize=(11.5, 5.8))
    for mode_idx, (mode_key, mode_label, hatch) in enumerate(modes):
        bottoms = np.zeros(len(rows), dtype=float)
        bar_x = x + (mode_idx - 0.5) * width
        for stage_key, stage_label, color in STAGE_ORDER:
            values = np.array([_stage_value(row, mode_key, stage_key) for row in rows], dtype=float)
            ax.bar(
                bar_x,
                values,
                width=width,
                bottom=bottoms,
                color=color,
                edgecolor="black",
                linewidth=0.4,
                hatch=hatch,
                label=stage_label if mode_idx == 0 else None,
            )
            bottoms += values

        for xpos, total in zip(bar_x, bottoms):
            if total <= 0.0:
                continue
            ax.annotate(f"{total:.1f}s", (xpos, total), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([row["display_name"] for row in rows])
    ax.set_ylabel("Seconds")
    ax.set_title("Supported Full-Circuit vs Adapter-Mode ZK Pipeline")

    stage_handles = [mpatches.Patch(facecolor=color, edgecolor="black", label=label) for _, label, color in STAGE_ORDER]
    mode_handles = [
        mpatches.Patch(facecolor="white", edgecolor="black", hatch=hatch, label=label)
        for _, label, hatch in modes
    ]
    stage_legend = ax.legend(handles=stage_handles, ncol=4, loc="upper left", frameon=True)
    ax.add_artist(stage_legend)
    ax.legend(handles=mode_handles, loc="upper right", frameon=True)
    _save(fig, output_dir, "vision_full_vs_adapter_totals")


def _plot_adapter_only_totals(rows: List[Dict[str, Any]], output_dir: Path) -> None:
    rows = [row for row in rows if _has_stage_summary(row, "adapter_benchmark")]
    if not rows:
        return

    x = np.arange(len(rows), dtype=float)
    fig, ax = plt.subplots(figsize=(10.2, 5.6))
    bottoms = np.zeros(len(rows), dtype=float)
    for stage_key, stage_label, color in STAGE_ORDER:
        values = np.array([_stage_value(row, "adapter_benchmark", stage_key) for row in rows], dtype=float)
        ax.bar(
            x,
            values,
            bottom=bottoms,
            color=color,
            edgecolor="black",
            linewidth=0.4,
            label=stage_label,
        )
        bottoms += values

    ax.set_xticks(x)
    ax.set_xticklabels([row["display_name"] for row in rows])
    ax.set_ylabel("Seconds")
    ax.set_title("Adapter-Mode Pipeline Totals Across Models")
    ax.legend(ncol=4, loc="upper left", frameon=True)
    for xpos, total in zip(x, bottoms):
        if total <= 0.0:
            continue
        ax.annotate(f"{total:.1f}s", (xpos, total), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=8)
    _save(fig, output_dir, "vision_adapter_totals_all_models")


def _plot_module_scaling(module_rows: List[Dict[str, Any]], output_dir: Path) -> None:
    rows = [row for row in module_rows if row["prove_mean_s"] > 0.0 and row["base_param_count"] > 0.0]
    if not rows:
        return

    fig, ax = plt.subplots(figsize=(9.4, 5.2))
    colors = {"lenet-5-small": "#4C78A8", "repvgg-a0": "#E45756", "vit": "#54A24B"}
    for row in rows:
        color = colors.get(row["model_name"], "#72B7B2")
        size = max(40.0, min(220.0, row["adapter_param_count"] / 10.0))
        ax.scatter(row["base_param_count"], row["prove_mean_s"], s=size, color=color, alpha=0.8, edgecolors="black")
        ax.annotate(
            row["module_name"].split(".")[-1],
            (row["base_param_count"], row["prove_mean_s"]),
            xytext=(6, 4),
            textcoords="offset points",
            fontsize=8,
        )

    _set_log_scale_if_positive(ax, "x", [row["base_param_count"] for row in rows])
    _set_log_scale_if_positive(ax, "y", [row["prove_mean_s"] for row in rows])
    ax.set_xlabel("Base module parameters")
    ax.set_ylabel("Adapter prove time (s)")
    ax.set_title("Vision Adapter Scaling by Proof Time")
    legend_handles = [
        mpatches.Patch(color=color, label=model_name)
        for model_name, color in colors.items()
        if any(row["model_name"] == model_name for row in rows)
    ]
    if legend_handles:
        ax.legend(handles=legend_handles, loc="best", frameon=True)
    _save(fig, output_dir, "vision_adapter_module_scaling")


def _plot_coverage(rows: List[Dict[str, Any]], output_dir: Path) -> None:
    if not rows:
        return

    x = np.arange(len(rows), dtype=float)
    width = 0.34
    base_counts = np.array([row["selected_base_param_count"] for row in rows], dtype=float)
    adapter_counts = np.array([row["selected_adapter_param_count"] for row in rows], dtype=float)

    fig, ax = plt.subplots(figsize=(9.4, 5.0))
    ax.bar(x - width / 2.0, base_counts, width=width, color="#4C78A8", label="Covered base params")
    ax.bar(x + width / 2.0, adapter_counts, width=width, color="#F58518", label="Adapter params")
    _set_log_scale_if_positive(ax, "y", list(base_counts) + list(adapter_counts))
    ax.set_xticks(x)
    ax.set_xticklabels([row["display_name"] for row in rows])
    ax.set_ylabel("Parameter count (log scale)")
    ax.set_title("Adapter Coverage vs Added Parameters")
    ax.legend(loc="best", frameon=True)

    for xpos, row in zip(x, rows):
        ax.annotate(
            f"{100.0 * row['selected_base_param_ratio']:.1f}%",
            (xpos - width / 2.0, row["selected_base_param_count"] or 1.0),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )
        ax.annotate(
            f"{100.0 * row['selected_adapter_param_ratio']:.3f}%",
            (xpos + width / 2.0, row["selected_adapter_param_count"] or 1.0),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )
    _save(fig, output_dir, "vision_adapter_coverage")


def _plot_module_prove_bars(module_rows: List[Dict[str, Any]], output_dir: Path) -> None:
    rows = [row for row in module_rows if row["prove_mean_s"] > 0.0]
    if not rows:
        return

    rows.sort(key=lambda row: row["prove_mean_s"], reverse=True)
    labels = [f"{row['display_name']}::{row['module_name']}" for row in rows]
    values = [row["prove_mean_s"] for row in rows]
    colors = {"lenet-5-small": "#4C78A8", "repvgg-a0": "#E45756", "vit": "#54A24B"}
    bar_colors = [colors.get(row["model_name"], "#72B7B2") for row in rows]

    fig_height = max(4.8, 0.34 * len(rows) + 1.8)
    fig, ax = plt.subplots(figsize=(11.5, fig_height))
    y = np.arange(len(rows), dtype=float)
    ax.barh(y, values, color=bar_colors)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Seconds")
    ax.set_title("Adapter Prove Time by Module")
    for ypos, value in zip(y, values):
        ax.annotate(f"{value:.2f}s", (value, ypos), xytext=(4, 0), textcoords="offset points", va="center", fontsize=8)
    _save(fig, output_dir, "vision_adapter_prove_by_module")


def plot_vision_adapter_summary(summary_path: Path, output_dir: Path) -> None:
    payload = read_json(summary_path)
    rows = _model_rows(payload)
    module_rows = _adapter_module_rows(payload)
    coverage_rows = _coverage_rows(payload)

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "figure.dpi": 160,
        }
    )

    _plot_full_vs_adapter(rows, output_dir)
    _plot_adapter_only_totals(rows, output_dir)
    _plot_module_scaling(module_rows, output_dir)
    _plot_coverage(coverage_rows, output_dir)
    _plot_module_prove_bars(module_rows, output_dir)
