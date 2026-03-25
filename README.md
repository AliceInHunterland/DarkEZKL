<h1 align="center">
	<br>
	Dark-EZKL
	<br>
	<img src="DarkEzkl.png" width="300" />
	<br>
</h1>

> A scaling-focused fork of EZKL for ZKML on large Transformer / LLM-style models.

This repository contains **Dark-EZKL**: a research-oriented fork of the upstream **EZKL v23.0.3** project (original: `zkonduit/ezkl`). Dark-EZKL focuses on **making large-model ZK inference more practical** by introducing a **probabilistic verification execution mode** (e.g. Freivalds-style checks for expensive linear algebra) and by documenting/benchmarking scaling behavior.

---

## Key docs (start here)

- **Probabilistic verification (design, knobs, security notes):** `PROBABILISTIC.md`
- **Benchmarks (what is measured + how to interpret outputs):** `BENCHMARKS.md`
- **Trusted SRS reconstruction / offline cache (avoid dummy k=26 OOM):** `SRS_RECONSTRUCTION.md`

---

## Built-in benchmark models (defaults)

The benchmark runner ships with a few built-in models (exported to ONNX by the runner):

- `lenet-5-small`
- `vit`
- `repvgg-a0`

(See `BENCHMARKS.md` for what each run measures and what artifacts are produced.)

---

## Run the benchmarks in Docker (recommended)

### Host requirements

- Docker
- **NVIDIA GPU** + driver
- NVIDIA Container Toolkit installed (so Docker can access GPUs)

If you *don’t* have a GPU, you can still try running without `--gpus all`, but performance will be much slower and some configurations may not work.

### GPU server quick path

If you are preparing a fresh GPU server, the helper script gives you a reproducible flow:

```bash
chmod +x ./setup-gpu.sh
./setup-gpu.sh check
./setup-gpu.sh prepare
./setup-gpu.sh build
./setup-gpu.sh smoke
```

For the full suite once the smoke run succeeds:

```bash
./setup-gpu.sh suite
```

If your server cannot fetch the trusted SRS for large circuits, pre-seed it first:

```bash
./scripts/reconstruct_srs_kzg26.sh
```

---

## 1) Build the Docker image

From the repo root:

```bash
docker build -t dark-ezkl:bench .
```

This image compiles:
- the `ezkl` CLI (from `./ezkl`)
- the Python wheel/bindings used by the benchmark scripts

Build time can be substantial (Rust + CUDA/PyTorch base).

---

## 2) Prepare host output/cache directories (recommended)

These mounts make reruns much faster (weights + SRS + HF/Torch caches persist on the host):

```bash
mkdir -p results cache .ezkl
```

- `./results`  → benchmark outputs (JSON reports + artifacts per run)
- `./cache`    → Torch/HuggingFace/ONNX caches
- `./.ezkl`    → EZKL SRS cache (e.g. `./.ezkl/srs/`)

---

## 3) Quick sanity check: GPU visible in container

```bash
docker run --rm --gpus all \
  dark-ezkl:bench \
  python3 -c "import torch; print('cuda available:', torch.cuda.is_available())"
```

---

## 4) Run benchmarks

### Quick test (recommended for first run)

```bash
docker run --rm --gpus all --shm-size=16g \
  -v "$PWD/results:/app/results" \
  -v "$PWD/cache:/app/.cache" \
  -v "$PWD/.ezkl:/root/.ezkl" \
  dark-ezkl:bench \
  python3 /app/benchmark.py --outdir /app/results \
    --models repvgg-a0 \
    --prob-k-values 2 \
    --runs 1
```

### Full benchmark suite (11-21 hours)

This runs the **complete** matrix of cases (3 models × 2 prob_k × 3 runs = 18 tests):

```bash
docker run --rm --gpus all --shm-size=16g \
  -v "$PWD/results:/app/results" \
  -v "$PWD/cache:/app/.cache" \
  -v "$PWD/.ezkl:/root/.ezkl" \
  dark-ezkl:bench \
  python3 /app/benchmark.py --outdir /app/results
```

### Alternative: Docker Compose

The compose file is now server-oriented: it persists `results`, `.cache`, and the EZKL SRS cache under `./.ezkl`.

```bash
mkdir -p results cache .ezkl
docker compose up --build test
```

Override the suite matrix without editing YAML:

```bash
BENCHMARK_MODELS=repvgg-a0 \
BENCHMARK_PROB_K_VALUES=2 \
BENCHMARK_RUNS=1 \
docker compose up --build test
```

### What you get (suite)
- `results/benchmark.json` (suite summary: cases + aggregates + env)
- `results/runs/<model>/k<prob_k>/run<i>/...` (per-run directories + artifacts)
  - each run includes a `vit_bench_report.json` (name is historical; it’s used for all models)

---

## 5) Run a single model once (bench_vit.py)

Use this when iterating/debugging or when you only want LeNet / ViT / RepVGG.

### LeNet (single run)

```bash
docker run --rm --gpus all --shm-size=16g \
  -v "$PWD/results:/app/results" \
  -v "$PWD/cache:/app/.cache" \
  -v "$PWD/.ezkl:/root/.ezkl" \
  dark-ezkl:bench \
  python3 /app/bench_vit.py \
    --outdir /app/results/single/lenet_k4 \
    --model-name lenet-5-small \
    --prob-k 4 \
    --prob-ops MatMul,Gemm,Conv \
    --prob-seed-mode fiat_shamir
```

### ViT (single run)

```bash
docker run --rm --gpus all --shm-size=16g \
  -v "$PWD/results:/app/results" \
  -v "$PWD/cache:/app/.cache" \
  -v "$PWD/.ezkl:/root/.ezkl" \
  dark-ezkl:bench \
  python3 /app/bench_vit.py \
    --outdir /app/results/single/vit_k4 \
    --model-name vit \
    --prob-k 4 \
    --prob-ops MatMul,Gemm,Conv \
    --prob-seed-mode fiat_shamir
```

### RepVGG (single run)

```bash
docker run --rm --gpus all --shm-size=16g \
  -v "$PWD/results:/app/results" \
  -v "$PWD/cache:/app/.cache" \
  -v "$PWD/.ezkl:/root/.ezkl" \
  dark-ezkl:bench \
  python3 /app/bench_vit.py \
    --outdir /app/results/single/repvgg_k4 \
    --model-name repvgg-a0 \
    --prob-k 4 \
    --prob-ops MatMul,Gemm,Conv \
    --prob-seed-mode fiat_shamir
```

### What you get (single run)

Under the `--outdir` you pass (e.g. `results/single/vit_k4/`):

- `vit_bench_report.json` (the measured timings + logrows + paths to artifacts)
- various EZKL artifacts (settings, compiled circuit, keys, witness, proof, etc.)

See `BENCHMARKS.md` for the meaning of each timing key.

---

## Notes / troubleshooting

### 1) First run downloads weights/datasets
Depending on the model, first run may download:
- torchvision weights (ViT)
- timm weights (RepVGG)
- datasets (LeNet training or data generation, depending on runner behavior)

Mounting `./cache:/app/.cache` avoids redownloading.

Important:
- If you run with `HF_HUB_OFFLINE=1` and the required pretrained weights are not already cached,
  some loaders may fall back to `pretrained=False`.
- That is acceptable for pipeline smoke tests, but it is not representative if you intend to
  benchmark the pretrained model itself. Warm the cache once online first, then rerun offline.

### 2) SRS downloads / materialization can be large (and can look "stuck")
`ezkl get-srs` needs an SRS for the chosen `logrows` (`k`).

Tips:
- **Persist the cache**: mount `./.ezkl:/root/.ezkl` to avoid re-downloading / re-generating across runs.
- **No-network (benchmarking only, small k):** run with `-e EZKL_SRS_SOURCE=dummy` (**recommended only for `k <= 22`**).
  - For `k=26`, dummy generation is huge and often gets OOM-killed; instead **pre-seed a trusted SRS**
    at `./.ezkl/srs/kzg26.srs` (see `SRS_RECONSTRUCTION.md`, or run `./scripts/reconstruct_srs_kzg26.sh`).
- If your **run directory is on a different filesystem** (common with Docker bind mounts), copying a multi‑GB SRS
  can take a long time. Dark‑EZKL now tries **`hardlink → symlink → copy`** when writing to `--srs-path`.
  - You can force this with `EZKL_SRS_MATERIALIZE=symlink` (or `auto|hardlink|copy`).

### 3) ViT is heavy
`vit` can require higher `logrows` and significant RAM/VRAM.
If it fails, start with:
- fewer/lower `--prob-k`
- running a smaller model first (LeNet)
- increasing host resources

### 4) Where to look when something fails
Start with the per-run JSON report:
- `results/.../vit_bench_report.json`

It contains paths to the exact artifacts used for that run (settings, compiled circuit, witness, proof).

### 5) MNIST “HTTP Error 404” during download is expected
Torchvision’s MNIST downloader tries an old mirror first, then automatically falls back to an alternate mirror.
If the download eventually succeeds and extraction proceeds, you can ignore the earlier 404 lines.

### 6) PyTorch `TracerWarning` during export is expected
Warnings like:
- `TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect...`

usually occur during tracing / ONNX export. They are non-fatal for this benchmark flow.

### 7) “Is it stuck?” checklist
If you’re unsure whether the benchmark is still making progress:

- Check the container is still alive:
  - `docker ps`
- Check GPU/CPU activity:
  - `nvidia-smi -l 1`
  - `top` / `htop`
- Check the host-mounted output directory is changing:
  - new per-run folders under `results/runs/...`
  - a `vit_bench_report.json` appears when a run finishes
- Check the host-mounted `.ezkl/` is growing (SRS download / cache), especially on the first run.

If everything is idle for a long time (e.g. 10–15+ minutes) and there are no new files being written,
rerun a *single* small case first (`bench_vit.py --model-name lenet-5-small`) to validate the pipeline end-to-end.

---
