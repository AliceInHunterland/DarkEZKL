# BENCHMARKS.md — Dark-EZKL Benchmark Suite

This repo contains two benchmark entrypoints:

- `bench_vit.py`: **single-run executor** that exports a model to ONNX, generates/caches EZKL artifacts, and measures key phases (settings generation, calibration, compile, SRS, setup, witness, prove, verify).
- `benchmark.py`: **suite orchestrator** that runs a fixed matrix of cases (models × `prob_k` × runs) by repeatedly calling `bench_vit.py`, then aggregates results and generates plots.

The goal of the suite is to quantify how **probabilistic verification** (Freivalds-style checks) behaves in practice: runtime, chosen circuit size (`logrows`), and (best-effort) constraint counts.

---

## 1) Benchmark methodology (what we measure and why)

Each single run (`bench_vit.py`) executes the pipeline:

1. **Export model to ONNX + create input JSON**
   - For built-in small models (`vit`, `lenet-5-small`, `repvgg-a0`) this happens inside `bench_vit.py`.
2. **`gen-settings`**
   - Performed in a “hybrid” way:
     - First via the `ezkl` **CLI** to guarantee robust base settings generation.
     - Then via the `ezkl` **Python API** to inject the probabilistic parameters (`execution_mode=probabilistic`, `prob_k`, `prob_ops`, `prob_seed_mode`) into `settings.json`.
3. **`calibrate-settings`**
   - Runs via the **CLI** targeting `"resources"`, so EZKL chooses a feasible `logrows` for the current circuit and environment constraints.
4. **`compile-circuit`**
5. **`get-srs`**
   - Cached by `k=logrows` under `out_dir/_srs_cache/k{logrows}.srs`.
6. **`setup`** (key generation)
7. **`gen-witness`**
8. **`mock`** (optional, enabled by default)
9. **`prove`**
10. **`verify`** (enabled by default)

This mirrors a “real user workflow” (compile once, then prove/verify), while still measuring the costs that matter for scaling and iteration time.

---

## 2) The benchmark suite (benchmark.py)

`benchmark.py` Output structure:

- `results/benchmark.json`: full suite report (cases + aggregates + env + plot status)
- `results/runs/<model>/k<prob_k>/run<i>/`: per-run artifacts, including `vit_bench_report.json`

## 3) Interpreting `RunMetrics` (from bench_vit.py)

Each successful run writes a `vit_bench_report.json` with:

```json
{
  "single_run": true,
  "metrics": { ... RunMetrics ... }
}
```

### RunMetrics fields

- `model_name`: Name passed to `--model-name` (e.g. `vit`).
- `execution_mode`: Always `"probabilistic"` in this benchmark runner.
- `prob_k`: Freivalds repetition count (soundness knob + cost knob).
- `prob_ops`: Which ONNX ops are enabled for probabilistic verification (default: `MatMul,Gemm,Conv`).
  - The suite's model-default runs now keep `Conv` enabled for `repvgg-a0` too.
  - Excluding `Conv` there demotes the early split RepVGG branches back to exact execution,
    which can push the first 3x3 Conv segment past the configured split caps.
- `prob_seed_mode`: How the random challenge is derived (`fiat_shamir` by default).
- `input_scale`, `param_scale`, `num_inner_cols`: Quantization + layout knobs used when generating settings.
- `logrows`: The circuit size chosen during calibration (best-effort read from `settings.json`).
  - Circuit “capacity” is roughly proportional to `2^logrows`.
  - For `repvgg-a0`, the split benchmark keeps a `k=19` cap but allows up to 500k calibrated rows
    and 8.0M total assignments; the early Conv branches need that extra assignment headroom after
    calibration widens `num_inner_cols` to stay within `k=19`.
  - `repvgg-a0` also rewrites oversized Conv nodes into smaller channel chunks before ONNX
    splitting, so single 3x3 Conv branches can be retried as multiple smaller segments.
  - `repvgg-a0` also folds inference `BatchNormalization` identity branches into affine ops
    before splitting, which avoids tract failures on standalone split BN segments.
- `constraint_count`: Best-effort extraction from `settings.json` (may be `null` if not present).
- `timings_s`: Map of measured durations (seconds) for key phases (see below).
- `settings_path`, `compiled_path`, `witness_path`, `proof_path`, `work_dir`: Paths to artifacts for debugging / reproduction.

### Timing keys (timings_s)

All timings are wall-clock durations measured inside `bench_vit.py`:

- `gen_settings_s`
  - Includes **both**:
    - CLI `ezkl gen-settings`, and
    - Python API call that injects probabilistic parameters into `settings.json`.
- `calibrate_settings_s`
  - CLI `ezkl calibrate-settings`.
  - This is where `logrows` is typically selected.
- `compile_circuit_s`
  - Compile time (Python API with CLI fallback).
- `get_srs_s`
  - SRS download/load time. Often near-zero after caching.
- `setup_s`
  - Key generation time (depends strongly on `logrows` and backend).
- `gen_witness_s`
  - Witness generation time (depends on model/ops and ONNX runtime details).
- `mock_s`
  - A fast sanity-check run (optional).
- `prove_s`
  - Proving time (typically the dominant cost at scale).
- `verify_s`
  - Verification time (usually much smaller than prove).

### “Setup time, proof time, constraint counts”
When reading results:
- “setup time” = `timings_s["setup_s"]`
- “proof time” = `timings_s["prove_s"]`
- “constraint count” = `constraint_count` (if present), plus `logrows` as the capacity proxy

`benchmark.py` also produces an `aggregates` block with mean/std/min/max over runs per case.

---

## 4) What `check_mode=UNSAFE` means (and why it affects benchmarks)

In many benchmark environments (e.g. Docker), the suite sets:

- `EZKL_CHECK_MODE=unsafe`

This generally means “opt out of some safety checks” to improve performance. The practical impact:

- **Fewer / cheaper constraints** in some parts of the circuit (depending on EZKL internals).
- **Faster proving and witness generation**, especially for large circuits.
- **Weaker guarantees**: unsafe modes can skip or relax checks that would otherwise reject malformed or out-of-range values.

### How to think about it in results
- If you benchmark with `unsafe`, you should treat times as “best-case throughput”.
- If you need production-grade assurances, rerun with safe checks and compare.

### Reproducing safe vs unsafe
- Unsafe:
  - `EZKL_CHECK_MODE=unsafe`
- Safer:
  - `EZKL_CHECK_MODE=safe` (or unset it)

Because `check_mode` can change both runtime and constraints, comparisons should only be made between runs with the same check mode.

---

## 5) Scaling behavior: `logrows` and `prob_k`

### A) How `logrows` scales things
`logrows` is the log2 of the polynomial/circuit size. Roughly:

- Larger circuits → need higher `logrows`
- Higher `logrows` typically increases:
  - SRS size / load time,
  - proving time,
  - memory usage,
  - setup time.

Calibration chooses `logrows` to fit the circuit with some margin. If you change model size, quantization scales, check mode, or probabilistic settings, you can see calibration pick a different `logrows`.

**Rule of thumb:** if `logrows` increases by 1, capacity doubles (and many costs rise noticeably).

### B) How `prob_k` scales things
`prob_k` is the repetition count for probabilistic checks (Freivalds-style verification). For square-ish matrix operations of dimension `N`, the classical intuition is:

- Exact checking cost ~ `O(N^3)` constraints (full matmul relation)
- Freivalds checking cost ~ `O(prob_k * N^2)` constraints (mat-vec checks repeated `prob_k` times)

In Dark-EZKL, increasing `prob_k` generally:
- **increases constraint counts roughly linearly** (for prob-checked ops),
- increases witness/prove time roughly linearly for those checks,
- may indirectly increase `logrows` if the circuit no longer fits the previously selected `logrows`.

### C) Interaction: `prob_k` can change `logrows`
Even if the per-op cost scales linearly with `prob_k`, if the constraint count crosses a circuit “capacity boundary”, calibration may bump `logrows`, causing a non-linear jump in runtime.

---

## 6) How to run + reproduce

### Single-run (bench_vit.py)
Example:

```bash
python3 bench_vit.py \
  --outdir results_single/vit_k4 \
  --model-name vit \
  --prob-k 4 \
  --prob-ops MatMul,Gemm,Conv \
  --prob-seed-mode fiat_shamir
```

Artifacts and the run report will be under `results_single/vit_k4/`.

### Full suite (benchmark.py)
```bash
python3 benchmark.py --outdir results
```

Note: the suite is fixed; attempting to pass other `--prob-k-values` / `--runs` will be rejected by design.

---

## 7) Notes / gotchas

- `constraint_count` is “best-effort”:
  - It is extracted by searching `settings.json` for fields containing “constraint”.
  - Some EZKL versions/settings shapes may not include constraints explicitly → value may be `null`.
- Split / segmented pipelines normalize execution per concrete segment:
  - if a split segment has no original ops matching the requested `prob_ops`,
    Dark-EZKL now demotes that segment to exact execution.
  - This prevents internal Conv/einsum lowerings from accidentally using probabilistic checks
    just because the top-level run requested `execution_mode=probabilistic`.
- `bench_metrics.json` plot payload:
  - For compatibility with existing plotting code, `benchmark.py` reuses a `logrows` field in the plot input to label entries by `prob_k` (i.e. `logrows = prob_k` in the plot payload). This is just for plotting labels; the actual `logrows` is preserved in per-run `vit_bench_report.json`.
