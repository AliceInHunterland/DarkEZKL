import argparse
import os
import re
import math
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

from .utils import read_json


def _to_records(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    models = payload.get("models", {})

    for model_name, m in models.items():
        logrows = m.get("logrows")
        err = m.get("error")
        if err:
            # Skip failed models in plots (but they remain in JSON for debugging)
            continue

        label = f"{model_name}\n(k={logrows})" if logrows is not None else model_name

        for t in m.get("prove_times_s", []):
            records.append({"Model": label, "Operation": "Prove", "Time (s)": float(t)})
        for t in m.get("verify_times_s", []):
            records.append({"Model": label, "Operation": "Verify", "Time (s)": float(t)})

    return records


def plot(json_path: Path, output_dir: Path) -> None:
    payload = read_json(json_path)
    records = _to_records(payload)
    df = pd.DataFrame(records)

    if df.empty:
        raise RuntimeError("No successful model timings found to plot (all models errored or no data).")

    # Calculate milliseconds for the new plot requirement
    df["Time (ms)"] = df["Time (s)"] * 1000.0

    sns.set_theme(style="whitegrid")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Existing Combined box plot (Log Scale)
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x="Model", y="Time (s)", hue="Operation")
    plt.title("EZKL Performance: ZK-Prove vs ZK-Verify (GPU)")
    plt.yscale("log")
    plt.ylabel("Time (seconds) - Log Scale")
    plt.tight_layout()
    out1 = output_dir / "ezkl_benchmark_boxplot.png"
    plt.savefig(out1)
    plt.close()

    # 2. Verify bar plot
    verify_df = df[df["Operation"] == "Verify"]
    if not verify_df.empty:
        plt.figure(figsize=(10, 5))
        sns.barplot(data=verify_df, x="Model", y="Time (s)", errorbar="sd", capsize=0.1)
        plt.title("ZK-Verify Latency")
        plt.ylabel("Time (seconds)")
        plt.tight_layout()
        out2 = output_dir / "ezkl_verify_barplot.png"
        plt.savefig(out2)
        plt.close()
    else:
        out2 = "skipped (no verify data)"

    # 3. Prove bar plot
    prove_df = df[df["Operation"] == "Prove"]
    if not prove_df.empty:
        plt.figure(figsize=(10, 5))
        sns.barplot(data=prove_df, x="Model", y="Time (s)", errorbar="sd", capsize=0.1)
        plt.title("ZK-Prove Latency")
        plt.ylabel("Time (seconds)")
        plt.tight_layout()
        out3 = output_dir / "ezkl_prove_barplot.png"
        plt.savefig(out3)
        plt.close()
    else:
        out3 = "skipped (no prove data)"

    # 4. NEW PLOT: Box Plots, Linear Scale (ms), Annotated, DarkBlue/LightBlue
    plt.figure(figsize=(12, 7))

    palette = {"Prove": "#00008B", "Verify": "#ADD8E6"}  # Dark Blue, Light Blue

    ax = sns.boxplot(
        data=df,
        x="Model",
        y="Time (ms)",
        hue="Operation",
        palette=palette,
        showfliers=False,
    )

    plt.title("EZKL Latency (Linear Scale)")
    plt.ylabel("Time (ms, absolute)")

    medians = df.groupby(["Model", "Operation"])["Time (ms)"].median().reset_index()

    models_order = [t.get_text() for t in ax.get_xticklabels()]
    operations_order = ["Prove", "Verify"]

    offsets = {"Prove": -0.2, "Verify": 0.2}

    for i, model_label in enumerate(models_order):
        for op in operations_order:
            row = medians[(medians["Model"] == model_label) & (medians["Operation"] == op)]
            if row.empty:
                continue

            med_val = row["Time (ms)"].values[0]

            if med_val > 10000:
                txt = f"{med_val/1000:.1f}s"
            else:
                txt = f"{int(med_val)}"

            x_pos = i + offsets.get(op, 0)

            ax.text(
                x_pos,
                med_val,
                txt,
                ha="center",
                va="bottom",
                fontweight="bold",
                fontsize=9,
                color="black",
                bbox=dict(facecolor="white", alpha=0.5, edgecolor="none", pad=1),
            )

    plt.tight_layout()
    out4 = output_dir / "ezkl_latency_linear_ms.png"
    plt.savefig(out4)
    plt.close()

    print(f"Wrote: {out1}")
    print(f"Wrote: {out2}")
    print(f"Wrote: {out3}")
    print(f"Wrote: {out4}")

    # 5. Individual plots per model
    unique_models = df["Model"].unique()
    for m_label in unique_models:
        sub_df = df[df["Model"] == m_label]
        clean_name = re.sub(r"[^a-zA-Z0-9_\-]+", "_", m_label).strip("_")

        plt.figure(figsize=(6, 5))
        sns.boxplot(data=sub_df, x="Operation", y="Time (s)")
        plt.title(f"Latency: {m_label.replace(chr(10), ' ')}")
        plt.ylabel("Time (seconds)")
        plt.tight_layout()

        out_ind = output_dir / f"ezkl_latency_{clean_name}.png"
        plt.savefig(out_ind)
        plt.close()
        print(f"Wrote: {out_ind}")


def main():
    p = argparse.ArgumentParser(description="Plot EZKL benchmark JSON.")
    p.add_argument("--json", default=os.environ.get("BENCH_JSON", "results/bench_metrics.json"))
    p.add_argument("--outdir", default=os.environ.get("PLOTS_OUTDIR", "results"))
    args = p.parse_args()
    plot(json_path=Path(args.json), output_dir=Path(args.outdir))
