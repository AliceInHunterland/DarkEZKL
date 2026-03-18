#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_NAME="${IMAGE_NAME:-dark-ezkl:bench}"
RESULTS_DIR="${RESULTS_DIR:-$REPO_ROOT/results}"
CACHE_DIR="${CACHE_DIR:-$REPO_ROOT/cache}"
EZKL_CACHE_DIR="${EZKL_CACHE_DIR:-$REPO_ROOT/.ezkl}"
SHM_SIZE="${SHM_SIZE:-16g}"

DEFAULT_MODELS="${BENCHMARK_MODELS:-lenet-5-small,repvgg-a0,vit}"
DEFAULT_PROB_K_VALUES="${BENCHMARK_PROB_K_VALUES:-2,4}"
DEFAULT_RUNS="${BENCHMARK_RUNS:-3}"

SMOKE_MODEL="${SMOKE_MODEL:-lenet-5-small}"
SMOKE_PROB_K="${SMOKE_PROB_K:-2}"
SMOKE_RUNS="${SMOKE_RUNS:-1}"

usage() {
  cat <<EOF
Usage:
  ./setup-gpu.sh check
  ./setup-gpu.sh prepare
  ./setup-gpu.sh build
  ./setup-gpu.sh smoke
  ./setup-gpu.sh suite
  ./setup-gpu.sh single [model] [prob_k] [runs]

Environment overrides:
  IMAGE_NAME=$IMAGE_NAME
  RESULTS_DIR=$RESULTS_DIR
  CACHE_DIR=$CACHE_DIR
  EZKL_CACHE_DIR=$EZKL_CACHE_DIR
  SHM_SIZE=$SHM_SIZE

  BENCHMARK_MODELS=$DEFAULT_MODELS
  BENCHMARK_PROB_K_VALUES=$DEFAULT_PROB_K_VALUES
  BENCHMARK_RUNS=$DEFAULT_RUNS

  SMOKE_MODEL=$SMOKE_MODEL
  SMOKE_PROB_K=$SMOKE_PROB_K
  SMOKE_RUNS=$SMOKE_RUNS
EOF
}

die() {
  echo "[error] $*" >&2
  exit 1
}

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "missing required command: $1"
}

prepare_dirs() {
  mkdir -p "$RESULTS_DIR" "$CACHE_DIR" "$EZKL_CACHE_DIR/srs"
}

warn_if_missing_large_srs() {
  if [[ ! -f "$EZKL_CACHE_DIR/srs/kzg26.srs" ]]; then
    cat >&2 <<EOF
[warn] Missing trusted SRS cache: $EZKL_CACHE_DIR/srs/kzg26.srs
       Small/medium runs may still work if the public SRS download succeeds.
       For k=26 in locked-down environments, pre-seed it first:
         ./scripts/reconstruct_srs_kzg26.sh
EOF
  fi
}

check_host() {
  need_cmd docker
  need_cmd nvidia-smi

  echo "[check] docker"
  docker --version

  echo "[check] nvidia-smi"
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

  echo "[check] docker daemon"
  docker info >/dev/null

  cat <<EOF

Host checks passed.

Next steps:
  ./setup-gpu.sh prepare
  ./setup-gpu.sh build
  ./setup-gpu.sh smoke
EOF
}

build_image() {
  prepare_dirs
  echo "[build] docker build -t $IMAGE_NAME $REPO_ROOT"
  docker build -t "$IMAGE_NAME" "$REPO_ROOT"
}

run_benchmark_container() {
  prepare_dirs
  warn_if_missing_large_srs

  docker run --rm --gpus all \
    --shm-size "$SHM_SIZE" \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -e ENABLE_ICICLE_GPU="${ENABLE_ICICLE_GPU:-true}" \
    -e ICICLE_SMALL_K="${ICICLE_SMALL_K:-0}" \
    -e NVIDIA_VISIBLE_DEVICES="${NVIDIA_VISIBLE_DEVICES:-all}" \
    -e NVIDIA_DRIVER_CAPABILITIES="${NVIDIA_DRIVER_CAPABILITIES:-compute,utility}" \
    -e TORCH_HOME=/app/.cache/torch \
    -e XDG_CACHE_HOME=/app/.cache \
    -e HF_HOME=/app/.cache/huggingface \
    -e HF_HUB_DISABLE_TELEMETRY=1 \
    -e EZKL_ONNX_PRECISION="${EZKL_ONNX_PRECISION:-fp16}" \
    -e EZKL_CHECK_MODE="${EZKL_CHECK_MODE:-unsafe}" \
    -e EZKL_LOOKUP_SAFETY_MARGIN="${EZKL_LOOKUP_SAFETY_MARGIN:-1.02}" \
    -e EZKL_SRS_SOURCE="${EZKL_SRS_SOURCE:-auto}" \
    -e EZKL_SRS_MAX_DUMMY_LOGROWS="${EZKL_SRS_MAX_DUMMY_LOGROWS:-22}" \
    -e PYTHONFAULTHANDLER=1 \
    -e MALLOC_ARENA_MAX=2 \
    -e LOG_LEVEL="${LOG_LEVEL:-INFO}" \
    -e ISOLATE_MODELS="${ISOLATE_MODELS:-true}" \
    -e MODEL_PROCESS_TAIL_LINES="${MODEL_PROCESS_TAIL_LINES:-250}" \
    -e SPLIT_ONNX="${SPLIT_ONNX:-true}" \
    -e SPLIT_MIN_PARAMS_M="${SPLIT_MIN_PARAMS_M:-0.05}" \
    -v "$RESULTS_DIR:/app/results" \
    -v "$CACHE_DIR:/app/.cache" \
    -v "$EZKL_CACHE_DIR:/root/.ezkl" \
    "$IMAGE_NAME" \
    "$@"
}

run_smoke() {
  echo "[run] smoke benchmark"
  run_benchmark_container \
    python3 /app/benchmark.py \
    --outdir /app/results/smoke \
    --models "$SMOKE_MODEL" \
    --prob-k-values "$SMOKE_PROB_K" \
    --runs "$SMOKE_RUNS"
}

run_suite() {
  echo "[run] benchmark suite"
  run_benchmark_container \
    python3 /app/benchmark.py \
    --outdir /app/results \
    --models "$DEFAULT_MODELS" \
    --prob-k-values "$DEFAULT_PROB_K_VALUES" \
    --runs "$DEFAULT_RUNS"
}

run_single() {
  local model="${1:-repvgg-a0}"
  local prob_k="${2:-2}"
  local runs="${3:-1}"

  echo "[run] single benchmark: model=$model prob_k=$prob_k runs=$runs"
  run_benchmark_container \
    python3 /app/benchmark.py \
    --outdir "/app/results/${model}_k${prob_k}" \
    --models "$model" \
    --prob-k-values "$prob_k" \
    --runs "$runs"
}

main() {
  local cmd="${1:-}"
  case "$cmd" in
    check)
      check_host
      ;;
    prepare)
      prepare_dirs
      echo "[ok] prepared:"
      echo "  $RESULTS_DIR"
      echo "  $CACHE_DIR"
      echo "  $EZKL_CACHE_DIR/srs"
      ;;
    build)
      build_image
      ;;
    smoke)
      run_smoke
      ;;
    suite)
      run_suite
      ;;
    single)
      shift
      run_single "${1:-repvgg-a0}" "${2:-2}" "${3:-1}"
      ;;
    ""|-h|--help|help)
      usage
      ;;
    *)
      usage
      die "unknown command: $cmd"
      ;;
  esac
}

main "$@"
