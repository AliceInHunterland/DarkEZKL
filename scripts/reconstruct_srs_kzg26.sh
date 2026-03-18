#!/usr/bin/env bash
set -euo pipefail

K="${1:-26}"
if [[ "$K" != "26" ]]; then
  echo "[error] This helper currently supports k=26 only (got k=$K)." >&2
  echo "        If you need other k values, use the manual flow in SRS_RECONSTRUCTION.md." >&2
  exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK_DIR="${SRS_WORK_DIR:-$REPO_ROOT/srs_work}"

PTAU_FILE="powersOfTau28_hez_final_26.ptau"
PTAU_URL="https://hermez.s3-eu-west-1.amazonaws.com/powersOfTau28_hez_final_26.ptau"
PTAU_B2SUM="418dee4a74b9592198bd8fd02ad1aea76f9cf3085f206dfd7d594c9e264ae919611b1459a1cc920c2f143417744ba9edd7b8d51e44be9452344a225ff7eead19"

# EZKL's pinned expected SHA256 for k=26 SRS (see ezkl/src/srs_sha.rs).
EXPECTED_SRS_SHA256="b198a51d48b88181508d8e4ea9dea39db285e4585663b29b7e4ded0c22a94875"

download() {
  local url="$1"
  local out="$2"

  if command -v aria2c >/dev/null 2>&1; then
    echo "[download] aria2c -> $out"
    aria2c -c -x 16 -s 16 -k 1M -o "$out" "$url"
    return 0
  fi

  if command -v curl >/dev/null 2>&1; then
    echo "[download] curl -> $out"
    # -C - resumes a partial download if present
    curl -L --fail --retry 5 --retry-delay 2 -C - -o "$out" "$url"
    return 0
  fi

  if command -v wget >/dev/null 2>&1; then
    echo "[download] wget -> $out"
    wget -O "$out" "$url"
    return 0
  fi

  echo "[error] need one of: aria2c, curl, wget" >&2
  exit 1
}

verify_blake2b() {
  local expected="$1"
  local file="$2"

  if command -v b2sum >/dev/null 2>&1; then
    printf '%s  %s\n' "$expected" "$(basename "$file")" | (cd "$(dirname "$file")" && b2sum -c -)
    return 0
  fi

  if command -v python3 >/dev/null 2>&1; then
    python3 - <<PY
import hashlib, pathlib, sys
p = pathlib.Path(r"$file")
expected = r"$expected".strip().lower()
h = hashlib.blake2b(digest_size=64)
with p.open("rb") as f:
    for chunk in iter(lambda: f.read(1024 * 1024), b""):
        h.update(chunk)
actual = h.hexdigest()
print("[b2] actual  :", actual)
print("[b2] expected:", expected)
if actual != expected:
    print("[error] blake2b checksum mismatch", file=sys.stderr)
    sys.exit(1)
PY
    return 0
  fi

  echo "[error] need b2sum (coreutils) or python3 to verify BLAKE2b checksum" >&2
  exit 1
}

sha256_file() {
  local file="$1"
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$file" | awk '{print $1}'
    return 0
  fi
  if command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$file" | awk '{print $1}'
    return 0
  fi
  if command -v python3 >/dev/null 2>&1; then
    python3 - <<PY
import hashlib, pathlib
p = pathlib.Path(r"$file")
h = hashlib.sha256()
with p.open("rb") as f:
    for chunk in iter(lambda: f.read(1024 * 1024), b""):
        h.update(chunk)
print(h.hexdigest())
PY
    return 0
  fi
  echo "[error] need sha256sum, shasum, or python3 to compute SHA256" >&2
  exit 1
}

echo "[info] repo_root : $REPO_ROOT"
echo "[info] work_dir  : $WORK_DIR"

mkdir -p "$WORK_DIR"

if [[ ! -f "$WORK_DIR/$PTAU_FILE" ]]; then
  echo "[step] downloading ptau (k=26)"
  download "$PTAU_URL" "$WORK_DIR/$PTAU_FILE"
else
  echo "[skip] ptau already exists: $WORK_DIR/$PTAU_FILE"
fi

echo "[step] verifying ptau BLAKE2b-512 checksum"
verify_blake2b "$PTAU_B2SUM" "$WORK_DIR/$PTAU_FILE"
echo "[ok] ptau checksum verified"

cd "$WORK_DIR"

if [[ ! -d "halo2-kzg-srs" ]]; then
  echo "[step] cloning halo2-kzg-srs"
  git clone --depth 1 https://github.com/han0110/halo2-kzg-srs.git
else
  echo "[skip] halo2-kzg-srs already cloned"
fi

pushd halo2-kzg-srs >/dev/null
mkdir -p ./srs

OUT="./srs/hermez-raw-${K}"
if [[ ! -f "$OUT" ]]; then
  echo "[step] converting .ptau -> halo2 raw params (this can take a while)"
  cargo run --release --bin convert-from-snarkjs \
    "../${PTAU_FILE}" \
    "./srs/hermez-raw-" \
    "${K}"
else
  echo "[skip] already converted: $OUT"
fi
popd >/dev/null

TARGET_DIR="$REPO_ROOT/.ezkl/srs"
TARGET_SRS="$TARGET_DIR/kzg${K}.srs"

echo "[step] installing SRS into EZKL cache: $TARGET_SRS"
mkdir -p "$TARGET_DIR"
cp -f "$WORK_DIR/halo2-kzg-srs/srs/hermez-raw-${K}" "$TARGET_SRS"

# If a dummy marker exists from a prior run, remove it so `EZKL_SRS_SOURCE=public` can treat this as trusted.
rm -f "${TARGET_SRS}.dummy"

echo "[step] verifying SHA256 of cached SRS"
ACTUAL_SHA256="$(sha256_file "$TARGET_SRS")"
echo "[sha256] actual  : $ACTUAL_SHA256"
echo "[sha256] expected: $EXPECTED_SRS_SHA256"

if [[ "$ACTUAL_SHA256" != "$EXPECTED_SRS_SHA256" ]]; then
  cat >&2 <<EOF
[warn] SHA256 mismatch.
       The file may still load, but EZKL_SRS_SOURCE=public strict hash checking may reject it.
       If you are in a production/trust-sensitive environment, investigate before proceeding.
EOF
else
  echo "[ok] SHA256 matches EZKL pinned hash for k=26"
fi

cat <<EOF

Done.

Cached SRS:
  $TARGET_SRS

Use it in Docker by mounting:
  -v "\$PWD/.ezkl:/root/.ezkl"

and keep:
  -e EZKL_SRS_SOURCE=auto

Optional: seed smaller-k aliases from this file without re-conversion:
  ./scripts/seed_srs_aliases.sh 26 24 25

EOF
