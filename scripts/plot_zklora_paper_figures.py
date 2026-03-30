#!/usr/bin/env python3
from __future__ import annotations

import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "external" / "zkLoRA" / "benchmarks" / "proof_metrics.csv"
SMOKE_DIR = ROOT / "results" / "zklora" / "smoke-proof-v2"
OUT_DIR = ROOT / "img"


def _short_model(name: str) -> str:
    mapping = {
        "distilgpt2": "DistilGPT2",
        "openai-community/gpt2": "GPT-2",
        "meta-llama/Llama-3.2-1B": "Llama-3.2-1B",
        "meta-llama/Llama-3.1-8B-Instruct": "Llama-3.1-8B",
        "meta-llama/Llama-3.3-70B-Instruct": "Llama-3.3-70B",
        "mistralai/Mixtral-8x7B-Instruct-v0.1": "Mixtral-8x7B",
    }
    return mapping.get(name, name.split("/")[-1])


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(f))


def _mean(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


def _aggregate_by_model(rows: list[dict[str, str]]) -> list[dict[str, float | str]]:
    buckets: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        buckets[row["base_model"]].append(row)

    out = []
    for model, model_rows in buckets.items():
        out.append(
            {
                "base_model": model,
                "label": _short_model(model),
                "avg_params": _mean([float(r["avg_params"]) for r in model_rows]),
                "avg_settings": _mean([float(r["avg_settings"]) for r in model_rows]),
                "avg_witness": _mean([float(r["avg_witness"]) for r in model_rows]),
                "avg_prove": _mean([float(r["avg_prove"]) for r in model_rows]),
                "avg_verify": _mean([float(r["avg_verify"]) for r in model_rows]),
                "number_of_loras": _mean(
                    [float(r["number_of_loras"]) for r in model_rows]
                ),
            }
        )

    out.sort(key=lambda r: float(r["avg_params"]))
    return out


def _write_aggregated_csv(rows: list[dict[str, float | str]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "base_model",
                "label",
                "avg_params",
                "avg_settings",
                "avg_witness",
                "avg_prove",
                "avg_verify",
                "number_of_loras",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def _style() -> None:
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
            "savefig.bbox": "tight",
        }
    )


def plot_stage_breakdown(rows: list[dict[str, float | str]], out_dir: Path) -> None:
    labels = [str(r["label"]) for r in rows]
    settings = [float(r["avg_settings"]) for r in rows]
    witness = [float(r["avg_witness"]) for r in rows]
    prove = [float(r["avg_prove"]) for r in rows]
    verify = [float(r["avg_verify"]) for r in rows]

    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    x = range(len(labels))
    colors = {
        "settings": "#4C78A8",
        "witness": "#F58518",
        "prove": "#E45756",
        "verify": "#72B7B2",
    }
    ax.bar(x, settings, label="Setup", color=colors["settings"])
    ax.bar(x, witness, bottom=settings, label="Witness", color=colors["witness"])
    bottom_2 = [a + b for a, b in zip(settings, witness)]
    ax.bar(x, prove, bottom=bottom_2, label="Prove", color=colors["prove"])
    bottom_3 = [a + b + c for a, b, c in zip(settings, witness, prove)]
    ax.bar(x, verify, bottom=bottom_3, label="Verify", color=colors["verify"])
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.set_ylabel("Seconds per LoRA module")
    ax.set_title("ZKLoRA Stage Breakdown by Base Model")
    ax.legend(ncol=4, frameon=True, loc="upper left")

    totals = [a + b + c + d for a, b, c, d in zip(settings, witness, prove, verify)]
    for idx, total in enumerate(totals):
        ax.annotate(
            f"{total:.1f}s",
            (idx, total),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            fontsize=9,
        )

    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"zklora_stage_breakdown.{ext}")
    plt.close(fig)


def plot_scaling(rows: list[dict[str, float | str]], out_dir: Path) -> None:
    params = [float(r["avg_params"]) for r in rows]
    prove = [float(r["avg_prove"]) for r in rows]
    verify = [float(r["avg_verify"]) for r in rows]
    labels = [str(r["label"]) for r in rows]

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.scatter(params, prove, s=90, color="#E45756", label="Prove")
    ax.scatter(params, verify, s=90, color="#72B7B2", label="Verify")
    label_offsets = {
        "DistilGPT2": (6, 5),
        "Llama-3.2-1B": (12, 2),
        "GPT-2": (8, 8),
        "Llama-3.3-70B": (10, 5),
        "Llama-3.1-8B": (10, 10),
        "Mixtral-8x7B": (10, 5),
    }
    for x, y, label in zip(params, prove, labels):
        ax.annotate(
            label,
            (x, y),
            xytext=label_offsets.get(label, (6, 5)),
            textcoords="offset points",
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Average parameters per LoRA module")
    ax.set_ylabel("Seconds per LoRA module")
    ax.set_title("ZKLoRA Scaling: Parameters vs Prove/Verify Time")
    ax.legend(frameon=True)

    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"zklora_scaling.{ext}")
    plt.close(fig)


def plot_smoke_artifacts(smoke_dir: Path, out_dir: Path) -> None:
    artifact_order = [
        "transformer_h_0_attn_c_attn.onnx",
        "transformer_h_0_attn_c_attn.ezkl",
        "transformer_h_0_attn_c_attn_witness.json",
        "transformer_h_0_attn_c_attn.pf",
        "transformer_h_0_attn_c_attn.vk",
        "kzg.srs",
    ]
    label_map = {
        "transformer_h_0_attn_c_attn.onnx": "ONNX",
        "transformer_h_0_attn_c_attn.ezkl": "Compiled",
        "transformer_h_0_attn_c_attn_witness.json": "Witness",
        "transformer_h_0_attn_c_attn.pf": "Proof",
        "transformer_h_0_attn_c_attn.vk": "VK",
        "kzg.srs": "SRS",
    }

    present = []
    sizes_mb = []
    for name in artifact_order:
        path = smoke_dir / name
        if path.exists():
            present.append(label_map[name])
            sizes_mb.append(path.stat().st_size / (1024 * 1024))

    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    x = range(len(present))
    ax.bar(x, sizes_mb, color="#54A24B")
    ax.set_xticks(list(x))
    ax.set_xticklabels(present)
    ax.set_yscale("log")
    ax.set_ylabel("Size (MB, log scale)")
    ax.set_title("Local 1-Module zkLoRA Smoke Run: Artifact Sizes")

    for idx, value in enumerate(sizes_mb):
        label = f"{value:.2f}" if value < 0.1 else f"{value:.1f}"
        ax.annotate(
            label,
            (idx, value),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
        )

    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"zklora_smoke_artifacts.{ext}")
    plt.close(fig)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _style()
    rows = _read_rows(CSV_PATH)
    agg_rows = _aggregate_by_model(rows)
    _write_aggregated_csv(agg_rows, ROOT / "results" / "zklora" / "zklora_aggregated_metrics.csv")
    plot_stage_breakdown(agg_rows, OUT_DIR)
    plot_scaling(agg_rows, OUT_DIR)
    if SMOKE_DIR.exists():
        plot_smoke_artifacts(SMOKE_DIR, OUT_DIR)
    print(f"Wrote figures to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
