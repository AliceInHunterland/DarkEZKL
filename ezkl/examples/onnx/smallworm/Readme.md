## Smallworm (WormVAE) — Dark-EZKL example

This example contains an ONNX export of a [WormVAE](https://github.com/TuragaLab/wormvae?tab=readme-ov-file) model: a VAE / latent-space representation of the *C. elegans* connectome.

The model “is a large-scale latent variable model with a very high-dimensional latent space consisting of voltage dynamics of 300 neurons over 5 minutes of time at the simulation frequency of 160 Hz. The generative model for these latent variables is described by stochastic differential equations modeling the nonlinear dynamics of the network activity.” (see the paper [here](https://openreview.net/pdf?id=CJzi3dRlJE-)).

In effect: it’s a generative model of a worm’s voltage dynamics that can generate new worm-like voltage dynamics given previous connectome state.

This repo is **Dark-EZKL** (a scaling-focused fork of EZKL). You can use it to create a ZK circuit equivalent to this WormVAE model, allowing you to **prove** correct execution of the model on an input. Dark-EZKL additionally supports a **probabilistic verification execution mode** for selected expensive ops (where applicable) to reduce constraint pressure for very large graphs.

---

## 0) Fetch the model artifacts (Git LFS)

The ONNX file is stored via Git LFS.

```bash
git lfs fetch --all
git lfs pull
```

---

## 1) Standard EZKL flow (deterministic constraints)

From this directory (`ezkl/examples/onnx/smallworm/`), run the typical loop:

```bash
# 1) Generate settings (fixed params are recommended for large models)
ezkl gen-settings \
  -M network.onnx \
  -O settings.json \
  --param-visibility fixed

# 2) Calibrate (expects a data JSON)
cp input.json calibration.json
ezkl calibrate-settings \
  -M network.onnx \
  -D calibration.json \
  --settings-path settings.json \
  --target resources

# 3) Compile
ezkl compile-circuit \
  -M network.onnx \
  -S settings.json \
  --compiled-circuit network.ezkl

# 4) SRS + setup
ezkl get-srs --settings-path settings.json --srs-path k.srs
ezkl setup -M network.ezkl --vk-path vk.key --pk-path pk.key --srs-path k.srs

# 5) Witness, prove, verify
ezkl gen-witness -M network.ezkl -D input.json --output witness.json --vk-path vk.key --srs-path k.srs
ezkl prove -M network.ezkl --witness witness.json --pk-path pk.key --proof-path proof.json --srs-path k.srs
ezkl verify --proof-path proof.json --settings-path settings.json --vk-path vk.key --srs-path k.srs
```

If verification succeeds, you have a proof that the circuit (compiled from the WormVAE ONNX graph) was executed correctly on the provided input.

---

## 2) Dark-EZKL probabilistic verification mode (scaling lever)

For very large models, Dark-EZKL supports an **execution mode** that can replace *full* constraints for selected expensive linear algebra ops with cheaper **probabilistic checks** (e.g., Freivalds-style checks). This can substantially reduce constraints for graphs dominated by MatMul/Gemm/Conv.

Important notes:

- This is a **soundness/performance trade-off**.
- You must choose:
  - `--prob-ops` (which ops to check probabilistically)
  - `--prob-k` (number of repetitions; higher = lower soundness error, slower)
  - `--prob-seed-mode` (typically `fiat_shamir` for NIZK-style challenge derivation)

Example settings generation in probabilistic mode:

```bash
ezkl gen-settings \
  -M network.onnx \
  -O settings.json \
  --param-visibility fixed \
  --execution-mode probabilistic \
  --prob-ops MatMul,Gemm,Conv \
  --prob-k 16 \
  --prob-seed-mode fiat_shamir
```

Then continue with calibration / compile / prove / verify exactly as in the deterministic flow (Section 1).

If you are comparing results across modes, keep in mind:

- Deterministic mode: constraints scale with the full arithmetic of the selected ops.
- Probabilistic mode: constraints scale more like the *check* rather than the full op (at the cost of a configurable soundness error).

For deeper details (threat model, parameter guidance), see this repo’s probabilistic verification documentation in the repo root (commonly named `PROBABILISTIC.md`).

---

## 3) Resource / scaling notes

- This model is **very large**. Running the full pipeline can require extremely large RAM (historically, hundreds of GB depending on settings and environment).
- If you are constrained:
  - Use **probabilistic mode** for heavy ops to reduce constraints (Section 2).
  - Consider **model segmentation / sharding** (if your workflow supports it) to reduce peak memory.
  - Run on a large-memory machine or inside a reproducible Docker environment if your repo provides one.

Once you have a verifier + proof artifacts, you can deploy verification to the chain of your choice (workflow depends on the target chain / verifier backend).
