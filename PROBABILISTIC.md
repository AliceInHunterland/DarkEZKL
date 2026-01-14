# Dark-EZKL: Probabilistic Verification (Freivalds) — Design + Security Notes

This repo (“Dark-EZKL”) extends EZKL with **probabilistic verification** for expensive linear-algebra operations (notably MatMul / Gemm / Conv lowered into matmul-like einsums).

The core idea is to **replace an O(N³) in-circuit matrix multiplication check** with an **O(N²)** randomized check using **Freivalds’ algorithm**, repeated `prob_k` times to reduce soundness error.

This document explains:
- the Freivalds check we implement and why it is O(N²) vs O(N³),
- how randomness is generated **in-circuit** using an LCG (so it is constrained),
- how `prob_k` (and seed mode) affects soundness and what “secure enough” means in practice.

---

## 1) What is being checked?

For matrices over a field \( \mathbb{F} \):

- \( A \in \mathbb{F}^{m \times n} \)
- \( B \in \mathbb{F}^{n \times p} \)
- \( C \in \mathbb{F}^{m \times p} \) (claimed product)

We want to verify the relation:

\[
A \cdot B = C
\]

### Deterministic check (expensive)

A deterministic in-circuit check typically enforces:

\[
\forall i,j:\quad \sum_{k=1}^{n} A_{i,k} \cdot B_{k,j} = C_{i,j}
\]

This is **O(m·n·p)** multiplications/additions, i.e. **O(N³)** for square \(N\times N\).

---

## 2) Freivalds’ algorithm (cheap, probabilistic)

Freivalds’ trick verifies the matrix product using a random vector \( r \in \mathbb{F}^{p} \):

Compute:

1. \( u = B \cdot r \)   (vector in \( \mathbb{F}^{n} \))
2. \( v = A \cdot u \)   (vector in \( \mathbb{F}^{m} \))
3. \( w = C \cdot r \)   (vector in \( \mathbb{F}^{m} \))

Check:

\[
v = w
\]

Equivalently:

\[
A(B r) = C r
\]

### Complexity

- \( B r \): **O(n·p)**
- \( A u \): **O(m·n)**
- \( C r \): **O(m·p)**
- equality check: **O(m)**

Total per repetition:

\[
O(n p + m n + m p) \approx O(N^2)\ \text{(for square shapes)}
\]

So a single Freivalds repetition turns a cubic constraint pattern into a quadratic one.

---

## 3) Where this is implemented in Dark-EZKL

### 3.1 The in-circuit Freivalds gadget

The gadget is implemented in:

- `ezkl/src/circuit/ops/probabilistic.rs`

It:
- derives \( r \) **inside the circuit** from a scalar seed (constrained),
- computes `u = B*r`, `v = A*u`, `w = C*r` using existing einsum/dot gadgets,
- constrains `v == w` elementwise.

The Freivalds wrapper for einsum matmul-like contractions lives in:

- `ezkl/src/circuit/ops/chip/einsum/freivalds.rs`

This file:
- parses a 2-input einsum equation like `"ij,jk->ik"`,
- canonicalizes it into a batched matmul structure,
- slices per-batch matrices and calls the Freivalds gadget,
- repeats across `k_repetitions` (and across batch items).

### 3.2 Repetition count (`prob_k`) and op selection

The circuit/region settings carry:
- `prob_k` (how many repetitions),
- `prob_ops` (which ops are probabilistically verified),
- `prob_seed_mode` (how the seed/challenge is chosen).

See:
- `ezkl/src/circuit/ops/region.rs` (defaults + RegionSettings plumbing)

At the CLI/settings layer, these parameters are surfaced via:
- `--execution-mode probabilistic`
- `--prob-k <int>`
- `--prob-ops <csv>`
- `--prob-seed-mode <mode>`

(Exact wiring varies by build/feature set; this doc focuses on the algorithm and its security properties.)

---

## 4) In-circuit randomness: LCG-derived challenge vector

Freivalds requires a “random” vector \( r \). In a zk circuit, we cannot just “pick randomness” off-chain and hope it’s honest: we must **bind the randomness to circuit constraints**.

Dark-EZKL does this by:
1. taking a scalar `seed` (a field element) as input to the gadget, and
2. expanding it into a length-`p` vector \( r \) via a simple **LCG** computed **in-circuit**.

### 4.1 LCG definition used

In `ezkl/src/circuit/ops/probabilistic.rs` the LCG is:

- multiplier: `LCG_A_U64 = 6364136223846793005`
- increment:  `LCG_C_U64 = 1442695040888963407`

We interpret these `u64` constants as field elements and iterate:

- `state0 = seed + domain_sep`
- `state_{i+1} = state_i * A + C`
- `r_i = state_{i+1}`

Where `domain_sep` is a `u64` domain separator (see below).

### 4.2 Why “domain separation” exists

If we repeat the Freivalds check `prob_k` times, we need *fresh-ish* challenges each time.

Instead of having `prob_k` independent seeds, we use:
- the same seed,
- but different `domain_sep` values per repetition (and per batch item).

In `freivalds.rs` this is done with:

- `domain_sep = rep * (batch_size + 1) + batch_index`

So each (repetition, batch-item) pair gets a distinct derived \( r \).

### 4.3 This is not a cryptographic PRG

An LCG is **not cryptographically secure**. We use it because:
- it is cheap to constrain,
- it deterministically expands a seed into many field elements,
- correctness is enforced by constraints (prover cannot lie about r once seed is fixed).

Security therefore depends heavily on:
- how the `seed` is chosen / bound, and
- the threat model (see below).

---

## 5) Soundness and the meaning of `prob_k`

### 5.1 Freivalds soundness (idealized)

Let \( D = AB - C \). If \( AB \neq C \), then \( D \neq 0 \).

Freivalds checks whether:

\[
D r = 0
\]

For “truly random” \( r \in \mathbb{F}^{p} \), the probability that a non-zero matrix maps a random vector to zero is at most:

\[
\Pr[Dr = 0] \le \frac{1}{|\mathbb{F}|}
\]

So one repetition gives soundness error about \( 1/|\mathbb{F}| \) (field-dependent).

If we repeat with independent \( r^{(1)}, \dots, r^{(k)} \), the idealized bound is:

\[
\Pr[\text{all checks pass}] \le \left(\frac{1}{|\mathbb{F}|}\right)^k
\]

In settings like BN254 / Pasta fields, \( |\mathbb{F}| \) is enormous, so even small `k` can be extremely strong *if* the challenge behaves like uniform field randomness.

### 5.2 What `prob_k` means here

In Dark-EZKL:
- `prob_k` controls how many times we run the Freivalds identity check (per eligible op).

Increasing `prob_k`:
- increases constraints roughly linearly in `k` (each repetition needs 3 mat-vec multiplies),
- reduces soundness error (in the idealized model).

---

## 6) Security implications and recommended usage patterns

Freivalds only protects you if the prover **cannot choose or bias** the random challenge \( r \) *after* committing to the alleged incorrect product.

In a zkSNARK, the prover typically controls witness generation, so you must be careful that the “seed” is not effectively prover-chosen.

### 6.1 Seed must be binding / non-malleable

If the prover can pick the seed freely, they can try to construct incorrect \( C \) that passes for that particular \( r \). In the extreme case, if the prover chooses \( r \) after seeing \( D \), Freivalds provides no meaningful protection.

Therefore, the seed should be derived from something the prover cannot adapt to arbitrarily, e.g.:

- **Fiat–Shamir challenge** derived from transcript commitments (binding the prover to earlier commitments), or
- **public randomness** (e.g., verifier-chosen seed, on-chain randomness, or externally sampled randomness that is committed to).

Dark-EZKL supports different seed modes via `prob_seed_mode` (plumbed through settings).

### 6.2 “Grinding” / repeated proof attempts

Even with Fiat–Shamir, a prover can sometimes attempt “grinding”:
- create many candidate proofs/commitments,
- see the derived challenge,
- keep the one that passes.

The cost of grinding grows with the cost to generate candidate proofs, but it exists in principle.
If you expect adversaries to grind, you should consider:
- increasing `prob_k`,
- ensuring the transcript challenge has enough entropy and is derived from binding commitments,
- limiting the ability to cheaply retry.

### 6.3 LCG caveats

Because we use an LCG (not a cryptographic PRG), the derived \( r \) values may have structure. In many proof threat models this is acceptable if:
- the seed is transcript-derived and binding,
- the adversary cannot efficiently exploit LCG structure to increase the chance that \( Dr = 0 \).

If you want a more conservative design, you can replace the LCG expander with:
- a hash-to-field construction inside the circuit (more expensive),
- or a bit-vector challenge (classic Freivalds with \( r \in \{0,1\}^p \)) with boolean constraints.

### 6.4 Practical guidance for choosing `prob_k`

There is no single universal `prob_k`, because it depends on:
- which field is used,
- whether seed randomness is truly unpredictable or merely transcript-derived,
- whether grinding is a concern,
- your acceptable soundness error.

As a rule of thumb for engineering:
- start with a moderate default (Dark-EZKL’s default is currently `40` in `RegionSettings`),
- benchmark constraint growth vs. proving time,
- and increase if the application is high-stakes or grinding is plausible.

---

## 7) Minimal conceptual example (single repetition)

Given `A (m×n)`, `B (n×p)`, `C (m×p)`:

1. `seed` is provided (public or transcript-derived).
2. Circuit derives `r` of length `p` via LCG(seed, domain_sep).
3. Circuit computes:
   - `u = B*r`
   - `v = A*u`
   - `w = C*r`
4. Circuit constrains `v[i] == w[i]` for all `i`.

Repeat steps 2–4 `prob_k` times with distinct `domain_sep`.

---

## 8) Summary

- Freivalds replaces a cubic matrix-multiplication check with a quadratic mat-vec identity check.
- Dark-EZKL implements this in `ezkl/src/circuit/ops/probabilistic.rs` and wires it into einsum matmul-like contractions in `.../einsum/freivalds.rs`.
- Randomness is expanded in-circuit via an LCG, binding `r` to a `seed` under constraints.
- Security depends on the seed being binding/unbiasable and on `prob_k` being chosen appropriately for your threat model.
