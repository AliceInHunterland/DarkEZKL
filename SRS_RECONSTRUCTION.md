# Trusted SRS reconstruction / offline cache (k=26)

If `ezkl get-srs` cannot download a trusted setup SRS, it may fall back to generating a **dummy** SRS.
For large circuits (notably `k=26`) dummy generation is enormous and often gets the process **SIGKILL / OOM-killed**
inside Docker.

The most reliable fix is to **pre-seed the EZKL global SRS cache** with a trusted `kzg26.srs`:

- Host path: `./.ezkl/srs/kzg26.srs`
- Container path (when mounted): `/root/.ezkl/srs/kzg26.srs`

Once that file exists, you can keep:

- `EZKL_SRS_SOURCE=auto` (default), and
- the usual Docker mount: `-v "$PWD/.ezkl:/root/.ezkl"`

and `ezkl get-srs` will hit the cache and avoid both downloads and dummy generation.

---

## Quick path (recommended): use the helper script

From the repo root:

```bash
chmod +x ./scripts/reconstruct_srs_kzg26.sh
./scripts/reconstruct_srs_kzg26.sh
```

This will:
1) download the Hermez `powersOfTau28_hez_final_26.ptau`,
2) verify its BLAKE2b-512 checksum,
3) convert it into Halo2 raw params (`kzg26.srs`),
4) place it in `./.ezkl/srs/kzg26.srs`, and
5) print the resulting SHA256 so you can compare it to EZKL's expected hash.

---

## Manual path: download .ptau → convert → cache

### A) Download the Hermez k=26 .ptau and verify checksum

```bash
mkdir -p srs_work && cd srs_work

# Download (fast path if aria2c is installed; otherwise use curl/wget)
aria2c -c -x 16 -s 16 -k 1M \
  https://hermez.s3-eu-west-1.amazonaws.com/powersOfTau28_hez_final_26.ptau

# Verify BLAKE2b-512 checksum (must match exactly)
printf '%s  %s\n' \
  '418dee4a74b9592198bd8fd02ad1aea76f9cf3085f206dfd7d594c9e264ae919611b1459a1cc920c2f143417744ba9edd7b8d51e44be9452344a225ff7eead19' \
  'powersOfTau28_hez_final_26.ptau' \
  | b2sum -c -
```

### B) Convert to Halo2 raw params (k=26)

We use `halo2-kzg-srs`'s `convert-from-snarkjs` helper:

```bash
git clone https://github.com/han0110/halo2-kzg-srs.git
cd halo2-kzg-srs

mkdir -p ./srs

cargo run --release --bin convert-from-snarkjs \
  ../powersOfTau28_hez_final_26.ptau \
  ./srs/hermez-raw- \
  26

# Output will be:
#   ./srs/hermez-raw-26
```

### C) Put it where EZKL will find it (global cache)

From inside `srs_work/halo2-kzg-srs`:

```bash
mkdir -p ../../.ezkl/srs
cp -f ./srs/hermez-raw-26 ../../.ezkl/srs/kzg26.srs
rm -f ../../.ezkl/srs/kzg26.srs.dummy
```

Optional: verify the resulting SRS SHA256 matches EZKL's pinned expected hash for `k=26`:

```bash
sha256sum ../../.ezkl/srs/kzg26.srs
# expected:
# b198a51d48b88181508d8e4ea9dea39db285e4585663b29b7e4ded0c22a94875  kzg26.srs
```

---

## Docker usage

Once `./.ezkl/srs/kzg26.srs` exists on the host, run containers with:

```bash
-v "$PWD/.ezkl:/root/.ezkl"
```

and keep:

```bash
-e EZKL_SRS_SOURCE=auto
```

If you want strict "never dummy" behavior:

```bash
-e EZKL_SRS_SOURCE=public
```

---

## Alternative reconstruction source: PPOT .ptau (very large)

If the Hermez URL ever becomes unreachable, the Perpetual Powers of Tau (PPOT) ceremony also publishes a k=26 `.ptau`
(often listed as ~72GB).

You can download that `.ptau` and run the same `convert-from-snarkjs ... 26` flow.

