# Dark-EZKL

Dark-EZKL is a Docker-first research fork of EZKL trimmed to one reproducible benchmark flow:

- paper benchmark: `lenet-5-small` full-circuit plus adapter-mode for `lenet-5-small`, `repvgg-a0`, and `vit`
- exploratory `repvgg-a0` full-circuit benchmarking
- plot generation from saved benchmark summaries

## What Is Current

- Main runner: `scripts/vision_adapter_benchmark.py`
- Plotter: `scripts/plot_vision_adapter_figures.py`
- Paper-safe full-circuit: `lenet-5-small`
- Paper-safe adapter-mode: `lenet-5-small repvgg-a0 vit`
- Exploratory full-circuit: `repvgg-a0`, only when you opt in explicitly with `--full-circuit-models repvgg-a0`
- Recommended execution mode: `exact`

This repo no longer uses the older `benchmark.py`, `bench_vit.py`, `docker-compose.yml`, or `BENCHMARKS.md` flow.

## Requirements

- Docker
- NVIDIA GPU
- NVIDIA Container Toolkit

The documented path is Docker. Native host installs are possible, but they are not the maintained workflow here.

## 1. Build The Image

Run this from the repo root on the server:

```bash
docker build --pull -t dark-ezkl:vision .
```

Important:
- Rebuild the image after pulling repo changes.
- The current image includes `onnxconverter-common`, which is required for RepVGG split fp16 conversion.

## 2. Prepare Persistent Directories

```bash
mkdir -p results cache .ezkl
```

These mounts persist across reruns:

- `results/` for benchmark outputs
- `cache/` for Torch and Hugging Face caches
- `.ezkl/` for SRS files

## 3. Optional: Pre-Seed Large SRS Files

If your machine cannot download large trusted setup files during the run:

```bash
./scripts/reconstruct_srs_kzg26.sh
```

See [SRS_RECONSTRUCTION.md](SRS_RECONSTRUCTION.md) for details.

## 4. Optional: Quick Sanity Check

This confirms the rebuilt image sees the GPU and the RepVGG fp16 conversion dependency:

```bash
docker run --rm --gpus all \
  dark-ezkl:vision \
  python3 -c "import torch, onnxconverter_common; print('cuda_available=', torch.cuda.is_available())"
```

## 5. Run The Paper Benchmark

This is the main reproducible command:

```bash
docker run --rm --gpus all --shm-size=16g \
  -v "$PWD/results:/app/results" \
  -v "$PWD/cache:/app/.cache" \
  -v "$PWD/.ezkl:/root/.ezkl" \
  dark-ezkl:vision \
  python3 /app/scripts/vision_adapter_benchmark.py \
    --models lenet-5-small repvgg-a0 vit \
    --full-circuit-models lenet-5-small \
    --out-dir /app/results/vision_adapter_bench \
    --rank 4 \
    --alpha 8 \
    --max-modules 8 \
    --full-execution-mode exact \
    --adapter-execution-mode exact
```

Expected outputs:

- `results/vision_adapter_bench/vision_adapter_summary.json`
- `results/vision_adapter_bench/model_comparison.csv`
- `results/vision_adapter_bench/adapter_module_metrics.csv`
- `results/vision_adapter_bench/plots/*.png`
- `results/vision_adapter_bench/plots/*.pdf`

## 6. Run A Faster Smoke Benchmark

Use this before the full paper run:

```bash
docker run --rm --gpus all --shm-size=16g \
  -v "$PWD/results:/app/results" \
  -v "$PWD/cache:/app/.cache" \
  -v "$PWD/.ezkl:/root/.ezkl" \
  dark-ezkl:vision \
  python3 /app/scripts/vision_adapter_benchmark.py \
    --models lenet-5-small repvgg-a0 vit \
    --full-circuit-models lenet-5-small \
    --out-dir /app/results/vision_adapter_smoke \
    --rank 4 \
    --alpha 8 \
    --max-modules 2 \
    --full-execution-mode exact \
    --adapter-execution-mode exact
```

## 7. Run RepVGG Full-Circuit Preflight

This is the safest exact command for checking whether the RepVGG full-circuit path gets through export, sanitize, split, calibrate, setup, witness, and mock before you spend time on proving:

```bash
docker run --rm --gpus all --shm-size=16g \
  -v "$PWD/results:/app/results" \
  -v "$PWD/cache:/app/.cache" \
  -v "$PWD/.ezkl:/root/.ezkl" \
  dark-ezkl:vision \
  python3 /app/scripts/vision_adapter_benchmark.py \
    --models repvgg-a0 \
    --full-circuit-models repvgg-a0 \
    --no-run-adapters \
    --out-dir /app/results/repvgg_full_circuit_preflight \
    --repeats 0 \
    --warmup 0 \
    --full-execution-mode exact \
    --no-plots
```

This path is exploratory, not part of the paper-safe default.

## 8. Run RepVGG Full-Circuit Benchmark

If the preflight passes, run the benchmark:

```bash
docker run --rm --gpus all --shm-size=16g \
  -v "$PWD/results:/app/results" \
  -v "$PWD/cache:/app/.cache" \
  -v "$PWD/.ezkl:/root/.ezkl" \
  dark-ezkl:vision \
  python3 /app/scripts/vision_adapter_benchmark.py \
    --models repvgg-a0 \
    --full-circuit-models repvgg-a0 \
    --no-run-adapters \
    --out-dir /app/results/repvgg_full_circuit \
    --repeats 1 \
    --warmup 0 \
    --full-execution-mode exact \
    --no-plots
```

## 9. Regenerate Plots Only

If the benchmark already finished and you only want plots:

```bash
docker run --rm --gpus all --shm-size=16g \
  -v "$PWD/results:/app/results" \
  -v "$PWD/cache:/app/.cache" \
  -v "$PWD/.ezkl:/root/.ezkl" \
  dark-ezkl:vision \
  python3 /app/scripts/plot_vision_adapter_figures.py \
    --summary /app/results/vision_adapter_bench/vision_adapter_summary.json \
    --out-dir /app/results/vision_adapter_bench/plots
```

## 10. Resume Long Runs

The benchmark checkpoints as it goes. If a run is interrupted, rerun the exact same command and completed work will be reused when possible.

For remote servers:

```bash
tmux new -s darkezkl
```

Then paste one of the `docker run ...` commands above inside that session.

## Docs

- [PROBABILISTIC.md](PROBABILISTIC.md) for probabilistic execution design and security notes
- [SRS_RECONSTRUCTION.md](SRS_RECONSTRUCTION.md) for offline trusted setup reconstruction
- [ezkl/README.md](ezkl/README.md) for the embedded proving engine note
