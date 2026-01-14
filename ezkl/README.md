<h1 align="center">
	<br>
	Dark-EZKL
	<br>
	<br>
</h1>

> A scaling-focused fork of EZKL for ZKML on large Transformer / LLM-style models.

This repository contains **Dark-EZKL**: a research-oriented fork of the upstream **EZKL v23.0.3** project (original: `zkonduit/ezkl`). Dark-EZKL focuses on **making large-model ZK inference more practical** by introducing a **probabilistic verification execution mode** (e.g. Freivalds-style checks for expensive linear algebra) and by documenting/benchmarking scaling behavior.

---

## Why Dark-EZKL (Large LLM support)

Traditional ZKML compilation makes many large-model workloads (especially Transformer/LLM inference) prohibitively expensive because the circuit must fully constrain huge linear algebra layers.

**Dark-EZKL adds an alternative scaling lever**:

- **Probabilistic verification for selected ops** (e.g. `MatMul`, `Gemm`, `Conv`) to reduce constraint pressure and unlock larger model sizes.
- **Configurable soundness/performance knob (`prob_k`)**: increase repetitions for lower soundness error, decrease for higher performance.
- A workflow that pairs well with **model sharding / segmentation** (benchmarked in this repo) to reduce peak memory and enable more scalable proving pipelines.

If you want the upstream production-focused EZKL, see the original project; if you want to explore scaling trade-offs for LLM-like graphs, you’re in the right place.

---

## Key docs (Dark-EZKL additions)

- **Probabilistic verification (design, knobs, security model, examples):**  
  See `PROBABILISTIC.md` (repo root).  
- **Benchmarks (what the results mean, how to reproduce, scaling guidance):**  
  See `BENCHMARKS.md` (repo root).

> From this README (inside `ezkl/`), those files are located at:
> - `../PROBABILISTIC.md`
> - `../BENCHMARKS.md`

---

## What stays the same (EZKL fundamentals)

Like EZKL, Dark-EZKL provides a library and CLI for proving inference of deep learning models / computational graphs as ZK proofs:

1. Define a computational graph (e.g., PyTorch / TensorFlow model).
2. Export the graph as an ONNX file + inputs (often JSON).
3. Generate a proving system artifact so you can prove statements such as:
   - “I ran this public model on private data and got this public output”
   - “I ran my private model on public data and got this public output”
   - “I ran this public model on public data and got this public output”

Under the hood, this follows the same high-level “compile → witness → prove → verify” structure as EZKL.

---

## Dark-EZKL execution mode: probabilistic (high-level)

In probabilistic mode, Dark-EZKL can replace *full* constraints for selected expensive operators with cheaper **probabilistic checks** (e.g., a Freivalds-style identity check for matrix multiplication).

At a high level:

- You choose a set of ops to treat probabilistically (`prob_ops`).
- You set the repetition count (`prob_k`) to trade performance vs soundness.
- You choose how the challenge seed is derived (`prob_seed_mode`), typically via Fiat–Shamir for non-interactive proofs.

For details and examples, see `../PROBABILISTIC.md`.

---

## Quickstart (CLI flags relevant to Dark-EZKL)

This fork introduces the following **settings / CLI knobs**:

- `--execution-mode probabilistic`
- `--prob-ops MatMul,Gemm,Conv` (comma-delimited)
- `--prob-k 16` (example; increase for stronger soundness)
- `--prob-seed-mode fiat_shamir` (or `public_seed`)

Example (settings generation):

```bash
ezkl gen-settings \
  -M network.onnx \
  -O settings.json \
  --execution-mode probabilistic \
  --prob-ops MatMul,Gemm,Conv \
  --prob-k 16 \
  --prob-seed-mode fiat_shamir
```

---

## Building / installing (notes for this fork)

Dark-EZKL is a fork and may not match upstream packaging/distribution.

- If you are using the upstream EZKL Python wheels, the canonical install remains:

```bash
pip install ezkl
```

- For this fork, build from source as appropriate for your environment (Rust + Cargo). A typical pattern is:

```bash
# from within the `ezkl/` directory
cargo install --locked --path .
```

(Exact commands may vary depending on how your workspace is structured.)

---

## Security & correctness notes

- Probabilistic checks introduce a **soundness/performance trade-off**. Treat `prob_k`, seed handling, and the threat model as first-class security parameters.
- Quantization and circuitization can cause small numerical differences between native framework outputs and circuit outputs.

See `../PROBABILISTIC.md` for the security model and recommended configurations.

---

## Upstream attribution

Dark-EZKL is derived from the upstream EZKL project and keeps the same overall mission: ZK proofs for ML inference. This fork specifically focuses on large-model scaling via probabilistic verification and benchmarking.

---

## Usage (Python workflow)

This section shows a practical end-to-end workflow (export ONNX → generate settings → calibrate → compile → witness → prove → verify), using the same structure as this repo’s `bench_vit.py`.

### 1) Export a PyTorch model to ONNX + generate an `input.json`

Dark-EZKL (like EZKL) expects:

- `network.onnx`
- `input.json` containing a flattened input tensor (or tensors)

Example:

```python
from pathlib import Path
import json
import torch
import torch.nn as nn

out_dir = Path("artifacts")
out_dir.mkdir(parents=True, exist_ok=True)

# Example model (replace with your own)
model = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 10)).eval()

dummy = torch.randn(1, 64, dtype=torch.float32)

onnx_path = out_dir / "network.onnx"
input_path = out_dir / "input.json"

torch.onnx.export(
    model,
    dummy,
    str(onnx_path),
    export_params=True,
    opset_version=14,
    do_constant_folding=True,
    input_names=["input"],
    output_names=["output"],
)

payload = {"input_data": [dummy.detach().cpu().reshape(-1).tolist()]}
input_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
print("Wrote:", onnx_path, input_path)
```

Notes:

- `input_data` is typically a list-of-inputs; each input is a flat list of scalars.
- For multi-input models, pass multiple entries in `input_data`.

### 2) Generate settings in probabilistic mode (hybrid CLI + Python)

In this repo’s benchmark runner we:
1) generate a valid base `settings.json` using the **CLI** (robust across environments), then
2) inject the **probabilistic** knobs using the **Python API** (to exercise the fork’s updated bindings).

```python
import subprocess
from pathlib import Path

import ezkl  # Dark-EZKL python bindings in this fork

onnx_path = Path("artifacts/network.onnx")
settings_path = Path("artifacts/settings.json")

# (A) CLI: base settings
subprocess.run(
    [
        "ezkl",
        "gen-settings",
        "-M",
        str(onnx_path),
        "-O",
        str(settings_path),
        "--input-scale",
        "7",
        "--param-scale",
        "7",
        "--input-visibility",
        "private",
        "--output-visibility",
        "public",
        "--param-visibility",
        "fixed",
    ],
    check=True,
)

# (B) Python: probabilistic knobs (Dark-EZKL additions)
run_args = None
if hasattr(ezkl, "PyRunArgs"):
    run_args = ezkl.PyRunArgs()
    run_args.input_scale = 7
    run_args.param_scale = 7
    run_args.num_inner_cols = 2

    # Optional visibility, depending on binding version
    if hasattr(ezkl, "PyVisibility"):
        run_args.input_visibility = ezkl.PyVisibility.Private
        run_args.output_visibility = ezkl.PyVisibility.Public
        run_args.param_visibility = ezkl.PyVisibility.Fixed

# IMPORTANT: the exact keyword signature can vary across ezkl versions;
# `bench_vit.py` tries several variants for compatibility.
ezkl.gen_settings(
    model=str(onnx_path),
    settings_path=str(settings_path),
    execution_mode="probabilistic",
    prob_ops=["MatMul", "Gemm", "Conv"],
    prob_k=16,
    prob_seed_mode="fiat_shamir",
    py_run_args=run_args,
    run_args=run_args,
)
```

Parameters to understand:

- `prob_ops`: which ONNX ops get probabilistic checks (typical starting point: `MatMul,Gemm,Conv`)
- `prob_k`: repetition count / soundness knob
- `prob_seed_mode`: how randomness is derived (Fiat–Shamir is the usual choice for NIZKs)

### 3) Calibrate, compile, setup, witness, prove, verify

This mirrors the benchmark flow:

```python
import subprocess
from pathlib import Path
import json

onnx_path = Path("artifacts/network.onnx")
input_path = Path("artifacts/input.json")
settings_path = Path("artifacts/settings.json")

compiled_path = Path("artifacts/network.ezkl")
vk_path = Path("artifacts/vk.key")
pk_path = Path("artifacts/pk.key")
witness_path = Path("artifacts/witness.json")
proof_path = Path("artifacts/proof.json")
srs_path = Path("artifacts/k17.srs")  # example; usually derived from settings.logrows

# Calibrate (CLI recommended; robust)
subprocess.run(
    [
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
    ],
    check=True,
)

# Read logrows to pick SRS (recommended)
settings = json.loads(settings_path.read_text(encoding="utf-8"))
logrows = int(settings.get("run_args", {}).get("logrows", 17))
srs_path = Path("artifacts") / f"k{logrows}.srs"

# Compile (Python or CLI; here we show CLI)
subprocess.run(
    [
        "ezkl",
        "compile-circuit",
        "-M",
        str(onnx_path),
        "-S",
        str(settings_path),
        "--compiled-circuit",
        str(compiled_path),
    ],
    check=True,
)

# SRS
subprocess.run(
    ["ezkl", "get-srs", "--settings-path", str(settings_path), "--srs-path", str(srs_path)],
    check=True,
)

# Setup
subprocess.run(
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
    ],
    check=True,
)

# Witness
subprocess.run(
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
    ],
    check=True,
)

# Prove
subprocess.run(
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
    ],
    check=True,
)

# Verify
subprocess.run(
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
    ],
    check=True,
)

print("OK: proof verified:", proof_path)
```

---

## Reproduction (Docker)

This repo includes a `docker-compose.yml` that reproduces the fixed probabilistic benchmark suite inside a container.

### Run the full benchmark suite (as in `docker-compose.yml`)

From the repo root:

```bash
docker compose build
docker compose up test
```

Artifacts are written to `./results` on your host (mounted into the container at `/app/results`).

### Run a single benchmark case (override the container command)

Example: run only one model once (single-run executor: `bench_vit.py`):

```bash
docker compose run --rm test \
  python3 /app/bench_vit.py \
  --outdir /app/results/single \
  --model-name vit \
  --prob-k 16 \
  --prob-ops MatMul,Gemm,Conv \
  --prob-seed-mode fiat_shamir
```

### Run the suite but write to a different output directory

```bash
docker compose run --rm test \
  python3 /app/benchmark.py \
  --outdir /app/results_alt
```

Notes:

- GPU support is enabled in `docker-compose.yml` via the NVIDIA device reservation.
- Several environment variables are set for stability/performance (shared memory, icicle toggles, caching).

---

## Adding New User Models (Python + verification on your data)

This repo’s benchmark runner (`bench_vit.py`) demonstrates a pattern you can reuse:

1) export your model to ONNX
2) create `input.json` from *your* real input
3) generate probabilistic settings (`prob_ops`, `prob_k`, `prob_seed_mode`)
4) compile → witness → prove → verify

### Example: add a real ViT-B/16 (PyTorch) export script

You can export a real model (e.g., from `timm` or `torchvision`) with the same ONNX + JSON structure.

```python
from pathlib import Path
import json
import torch

# Example using timm (install separately): pip install timm
import timm  # type: ignore

out_dir = Path("user_model_artifacts")
out_dir.mkdir(parents=True, exist_ok=True)

model = timm.create_model("vit_base_patch16_224", pretrained=True).eval()

# Your real user input goes here:
# - Keep it on CPU for export
# - Ensure it matches the model's expected normalization/preprocessing
x = torch.randn(1, 3, 224, 224, dtype=torch.float32)

onnx_path = out_dir / "network.onnx"
input_path = out_dir / "input.json"

torch.onnx.export(
    model,
    x,
    str(onnx_path),
    export_params=True,
    opset_version=14,
    do_constant_folding=True,
    input_names=["input"],
    output_names=["output"],
)

input_path.write_text(json.dumps({"input_data": [x.reshape(-1).tolist()]}, indent=2), encoding="utf-8")
print("Exported:", onnx_path)
```

### Verifying on your users’ data

- Put the user’s input tensor into `input.json` (`input_data`).
- Keep `--input-visibility private` if the user’s data should not be revealed.
- Decide whether the output should be public (common) or private (less common; depends on protocol).

### Scaling guidance when models get large

When you move beyond “toy” models:

- Prefer probabilistic mode for the heavy ops (`MatMul`, `Gemm`, `Conv`) to reduce constraint growth.
- Consider ONNX splitting / segmentation if memory is a bottleneck (see the benchmark docs at `../BENCHMARKS.md`).
- Tune `prob_k` upward for stronger soundness and downward for faster proving.
- Use Docker + GPU for reproducibility/performance.

---

## No Warranty

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
