# CUDA + PyTorch base (includes CUDA toolkit + nvcc)
FROM nvcr.io/nvidia/pytorch:24.09-py3

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ARG EZKL_VERSION=23.0.3
ARG EZKL_GIT_TAG=v23.0.3
ARG ONNXRUNTIME_GPU_VERSION=1.20.1

# Force UTF-8 everywhere to avoid locale/encoding-dependent failures
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONUTF8=1 \
    PYTHONIOENCODING=UTF-8 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    RUST_BACKTRACE=1 \
    EZKL_DIR=/root/.ezkl \
    \
    # SRS downloads can be slow/blocked depending on environment.
    #
    # EZKL_SRS_SOURCE=auto:
    #   - try downloading public trusted-setup SRS
    #   - if that fails, fall back to generating a local "dummy" SRS ONLY for k <= EZKL_SRS_MAX_DUMMY_LOGROWS
    #
    # IMPORTANT:
    #   Some environments get HTTP 403 / timeouts from the default public SRS bucket.
    #   Generating a dummy SRS for very large k (especially k=26) can take hours and
    #   often gets OOM-killed in Docker.
    #
    #   We therefore cap *auto* dummy generation at k=22 by default.
    #   If you need k>22 without network access, pre-seed /root/.ezkl/srs/kzg{k}.srs
    #   (see SRS_RECONSTRUCTION.md), or explicitly raise EZKL_SRS_MAX_DUMMY_LOGROWS at runtime.
    #
    # Override at runtime with:
    #   -e EZKL_SRS_SOURCE=public   (never dummy; fail if download unavailable)
    #   -e EZKL_SRS_SOURCE=dummy    (always dummy; can take a long time for big k)
    EZKL_SRS_SOURCE=auto \
    EZKL_SRS_MAX_DUMMY_LOGROWS=22 \
    \
    # Download knobs (defaults chosen to avoid false timeouts on big files).
    # Override at runtime if needed.
    EZKL_SRS_CONNECT_TIMEOUT_SECS=10 \
    EZKL_SRS_STALL_TIMEOUT_SECS=600 \
    EZKL_SRS_PROGRESS_EVERY_SECS=30 \
    \
    ENABLE_ICICLE_GPU=true \
    TORCH_HOME=/app/.cache/torch \
    XDG_CACHE_HOME=/app/.cache \
    XDG_CONFIG_HOME=/app/ \
    HF_HOME=/app/.cache/huggingface \
    HF_HUB_DISABLE_TELEMETRY=1 \
    CARGO_NET_GIT_FETCH_WITH_CLI=true \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    LC_CTYPE=C.UTF-8 \
    PATH="/root/.ezkl:/root/.cargo/bin:${PATH}"

# ---- OS deps (runtime + build) ----
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    git-lfs \
    curl \
    ca-certificates \
    build-essential \
    pkg-config \
    perl \
    binutils \
    python3-dev \
    openssl \
    libssl-dev \
    clang \
    libclang-dev \
    cmake \
    ninja-build \
    patchelf \
    zlib1g \
    && rm -rf /var/lib/apt/lists/* \
    && git lfs install

# ---- Rust Installation ----
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

# Ensure we use nightly Cargo/Rust
RUN . "$HOME/.cargo/env" && \
    rustup toolchain install nightly-2025-12-01 --profile minimal && \
    rustup default nightly-2025-12-01 && \
    cargo --version && rustc --version

# ---- Python deps ----
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip uninstall -y onnxruntime onnxruntime-gpu || true && \
    python3 -m pip install \
      "maturin>=1.0,<2.0" \
      onnx \
      "onnxruntime-gpu==${ONNXRUNTIME_GPU_VERSION}" \
      onnxconverter-common \
      numpy \
      pandas \
      matplotlib \
      seaborn \
      tqdm \
      timm \
      huggingface_hub \
      safetensors \
      pytest \
      pyyaml \
      scipy \
      jupyter \
      kaggle \
      py-solc-x \
      web3 \
      librosa \
      keras

# ---- Runtime dirs ----
RUN mkdir -p /app/.cache /app/results /app/artifacts /app/layer_setup

WORKDIR /app

# Copy project code (benchmark.py etc.)
COPY . .

# ---- Build & install ezkl CLI ----
# IMPORTANT:
# We build the CLI with `--no-default-features --features ezkl` to avoid pulling in
# optional feature-sets (notably `eth`) that can introduce `halo2_proofs` version
# conflicts via downstream crates.
#
# We also install the resulting binary into /usr/local/bin so it is available even if
# users bind-mount /root/.ezkl (SRS cache) and hide anything that might exist there.
RUN . "$HOME/.cargo/env" && \
    cd ezkl && \
    LOCKED=""; if [ -f Cargo.lock ]; then LOCKED="--locked"; fi; \
    cargo install --force ${LOCKED} --path . --no-default-features --features "ezkl" && \
    install -m 0755 "$HOME/.cargo/bin/ezkl" /usr/local/bin/ezkl && \
    mkdir -p "${EZKL_DIR}" && \
    ln -sf /usr/local/bin/ezkl "${EZKL_DIR}/ezkl"

# Optional Python deps for ezkl (do not fail build if file isn't present)
RUN cd ezkl && \
    if [ -f requirements.txt ]; then python3 -m pip install -r requirements.txt; fi

# ezkl/requirements.txt currently pins CPU-only onnxruntime==1.17.1.
# Re-install the GPU wheel afterwards so CUDAExecutionProvider remains available at runtime.
RUN python3 -m pip uninstall -y onnxruntime onnxruntime-gpu || true && \
    python3 -m pip install --no-cache-dir "onnxruntime-gpu==${ONNXRUNTIME_GPU_VERSION}"

# ---- Build/install Python bindings (clean + deterministic) ----
# IMPORTANT:
# We do NOT parse `nm/readelf` output to find `PyInit_*` symbols.
# That approach is fragile and can fail due to host/build locale encoding issues
# (e.g. `'charmap' codec can't decode ...`), even when the wheel is valid.
# Instead, we validate by importing the installed wheel and calling a function.
RUN . "$HOME/.cargo/env" && \
    cd ezkl && \
    rm -rf /tmp/ezkl_wheels && mkdir -p /tmp/ezkl_wheels && \
    maturin build --release --no-default-features --features "python-bindings,pyo3/extension-module" -i python3 --out /tmp/ezkl_wheels && \
    python3 -m pip uninstall -y ezkl || true && \
    WHEEL="$(ls -1 /tmp/ezkl_wheels/ezkl-*.whl | head -n 1)" && \
    echo "Installing wheel: $WHEEL" && \
    python3 -m pip install --no-deps --force-reinstall "$WHEEL"

# ---- Verify the compiled Python extension actually imports ----
# NOTE: /app contains a directory named "ezkl" (the Rust crate). That directory can form a namespace-package
# and sometimes confuse import resolution. We aggressively remove CWD and /app from sys.path so we import the wheel.
#
# If the wheel installs as a package wrapper (e.g. dist-packages/ezkl/<something>.so) and does NOT expose
# ezkl.version at the package top-level, we auto-create/overwrite ezkl/__init__.py to re-export the compiled submodule.
RUN python3 - <<'PY'
import os, sys, importlib, pkgutil

def clean_syspath():
    cwd = os.getcwd()
    sys.path[:] = [p for p in sys.path if p not in ("", cwd, "/app")]

def callable_version(mod) -> bool:
    fn = getattr(mod, "version", None)
    if callable(fn):
        v = fn()
        print(f"{mod.__name__}.version() -> {v!r}")
        return True
    return False

clean_syspath()

import ezkl
print("Imported ezkl from:", getattr(ezkl, "__file__", None))

# If we accidentally imported a namespace package from the source tree, __file__ will be None.
if getattr(ezkl, "__file__", None) is None and not hasattr(ezkl, "__path__"):
    raise SystemExit(
        "ERROR: Imported a module with no __file__ and no __path__. "
        "Cannot validate installed wheel import."
    )

# Success if top-level already exposes version().
if callable_version(ezkl):
    print("ezkl __version__:", getattr(ezkl, "__version__", "unknown"))
    raise SystemExit(0)

# If ezkl is a package wrapper, try submodules and find the compiled one.
chosen = None
if hasattr(ezkl, "__path__"):
    for m in pkgutil.iter_modules(ezkl.__path__):
        full = f"{ezkl.__name__}.{m.name}"
        try:
            sub = importlib.import_module(full)
        except Exception as e:
            print(f"Skipping {full} (import error): {e}")
            continue
        if callable_version(sub):
            chosen = m.name
            break

if not chosen:
    raise SystemExit(
        "ERROR: Could not find a callable version() on ezkl or any ezkl.* submodule after installing the wheel. "
        "The compiled extension was not imported / not exposed."
    )

# Ensure future `import ezkl` exposes the compiled API at the package top-level.
# This replaces the old nm/readelf-based PyInit_* parsing with a runtime import-based approach.
pkg_dir = next(iter(getattr(ezkl, "__path__")))
init_py = os.path.join(pkg_dir, "__init__.py")

print(f"Auto-patching {init_py} to re-export .{chosen}")
with open(init_py, "w", encoding="utf-8") as f:
    f.write(
        "# Auto-patched in Docker build: re-export compiled extension submodule\n"
        f"from .{chosen} import *  # noqa: F401,F403\n"
    )

# Re-import in a clean state to verify patch worked.
for k in list(sys.modules.keys()):
    if k == "ezkl" or k.startswith("ezkl."):
        del sys.modules[k]

clean_syspath()
import ezkl as ezkl2
print("Re-imported ezkl from:", getattr(ezkl2, "__file__", None))

if not callable_version(ezkl2):
    raise SystemExit(
        "ERROR: After patching ezkl/__init__.py, ezkl.version() is still missing/not callable."
    )

print("ezkl __version__:", getattr(ezkl2, "__version__", "unknown"))
PY

# ---- Sanity check core imports + versions early ----
RUN python3 - <<'PY'
import os, sys, importlib, pkgutil

cwd = os.getcwd()
sys.path = [p for p in sys.path if p not in ("", cwd, "/app")]

import torch
try:
    import torchvision
    tv = torchvision.__version__
except Exception as e:
    tv = f"unavailable ({e})"

import ezkl
import onnxruntime as ort
import yaml

print("torch:", torch.__version__)
print("torchvision:", tv)
print("python ezkl __file__:", getattr(ezkl, "__file__", None))
print("python ezkl __version__:", getattr(ezkl, "__version__", "unknown"))

def find_and_call_version(mod) -> str | None:
    fn = getattr(mod, "version", None)
    if callable(fn):
        return fn()
    if hasattr(mod, "__path__"):
        for m in pkgutil.iter_modules(mod.__path__):
            full = f"{mod.__name__}.{m.name}"
            try:
                sub = importlib.import_module(full)
            except Exception:
                continue
            fn2 = getattr(sub, "version", None)
            if callable(fn2):
                return fn2()
    return None

v = find_and_call_version(ezkl)
if not v:
    raise SystemExit("ERROR: Could not locate a callable ezkl.version(); compiled extension not exposed.")
print("ezkl.version():", v)

print("onnxruntime:", ort.__version__)
print("onnxruntime file:", getattr(ort, "__file__", None))
print("onnxruntime providers:", ort.get_available_providers())
print("cuda available:", torch.cuda.is_available())
PY
