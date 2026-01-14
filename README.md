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
> - `./PROBABILISTIC.md`
> - `./BENCHMARKS.md`


## Dark EZKL GPU Benchmark — LeNet-5 vs ViT-B/16 vs RepVGG-A0

This repo benchmarks **ezkl prove/verify timings** on **GPU** (Icicle) for:

- LeNet-5 (MNIST-style; trained quickly on first run if weights missing)
- ViT-B/16 (torchvision pretrained weights)
- RepVGG-A0 (timm pretrained weights; deploy-fused)

It produces:
- `results/bench_metrics.json`
- `results/ezkl_benchmark_boxplot.png` (box plot comparing prove vs verify for each model)
- `results/ezkl_verify_barplot.png`
- `results/ezkl_prove_barplot.png`

Artifacts for debugging are written to `artifacts/<model_slug>/`:
- ONNX model
- settings.json
- compiled circuit
- keys
- witness
- proof(s)

---

## Requirements (Host)
- Docker + Docker Compose
- NVIDIA driver
- NVIDIA Container Toolkit installed (so `docker` can access GPUs)

---

## Quickstart

```bash
mkdir -p results artifacts cache
docker compose build
docker compose run --rm ezkl-bench
```

Outputs:
- `results/bench_metrics.json`
- plots under `results/`

---

## Run a single model

```bash
# LeNet only
docker compose run --rm ezkl-bench python3 benchmark.py --model lenet --repeats 5 --outdir /app/results

# ViT only
docker compose run --rm ezkl-bench python3 benchmark.py --model vit --repeats 5 --outdir /app/results

# RepVGG only
docker compose run --rm ezkl-bench python3 benchmark.py --model repvgg --repeats 5 --outdir /app/results
```

---

## Re-plot from an existing JSON
```bash
docker compose run --rm ezkl-bench python3 plot_results.py --input /app/results/bench_metrics.json --outdir /app/results
```

---

## Notes / Troubleshooting

### 1) Why we removed `setup-gpu.sh`
Your previous Docker build failed because `setup-gpu.sh` expected a `Cargo.lock` in the project root:

> `Cargo.lock not found. Please run this script from the project root.`

For **ezkl v23.0.3**, GPU acceleration is enabled by:
- building ezkl from source with feature `icicle`
- setting `ENABLE_ICICLE_GPU=true` at runtime

So this repo installs our own Dark-ezkl fork bassed on ezkl v23.0.3 from source with:
- `maturin develop --release --features python-bindings,icicle`

### 2) First run may download weights/datasets
- ViT weights are downloaded by torchvision
- RepVGG weights are downloaded by timm
- LeNet weights are produced by a quick MNIST training run if missing

All downloads and caches are stored under `./cache` (mounted to `/app/.cache`).

### 3) ViT is heavy
ViT-B/16 can be extremely large for ZK circuits and may require high `logrows` and lots of RAM/VRAM.
If it fails, inspect `artifacts/vit-b16/` and try fewer repeats or higher machine resources.
