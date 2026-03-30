#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EZKL_CACHE_DIR="${EZKL_CACHE_DIR:-$REPO_ROOT/.ezkl}"
SRS_DIR="${EZKL_CACHE_DIR}/srs"

SOURCE_K="${1:-26}"
shift || true

if [[ $# -eq 0 ]]; then
  TARGET_KS=(24 25)
else
  TARGET_KS=("$@")
fi

find_source() {
  local k="$1"
  local candidate
  for candidate in \
    "$SRS_DIR/kzg${k}.srs" \
    "$SRS_DIR/k${k}.srs"
  do
    if [[ -f "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  return 1
}

link_or_copy() {
  local src="$1"
  local dst="$2"

  if [[ -e "$dst" ]]; then
    echo "[skip] already exists: $dst"
    return 0
  fi

  if ln "$src" "$dst" 2>/dev/null; then
    echo "[link] $dst -> $(basename "$src")"
    return 0
  fi

  cp -f "$src" "$dst"
  echo "[copy] $dst <- $(basename "$src")"
}

mkdir -p "$SRS_DIR"

SOURCE_PATH="$(find_source "$SOURCE_K" || true)"
if [[ -z "$SOURCE_PATH" ]]; then
  cat >&2 <<EOF
[error] source SRS not found for k=${SOURCE_K}
        expected one of:
          $SRS_DIR/kzg${SOURCE_K}.srs
          $SRS_DIR/k${SOURCE_K}.srs
EOF
  exit 1
fi

echo "[info] source: $SOURCE_PATH"

for target_k in "${TARGET_KS[@]}"; do
  if ! [[ "$target_k" =~ ^[0-9]+$ ]]; then
    echo "[error] invalid target k: $target_k" >&2
    exit 2
  fi
  if (( target_k <= 0 )); then
    echo "[error] invalid target k: $target_k" >&2
    exit 2
  fi
  if (( target_k > SOURCE_K )); then
    echo "[error] target k=${target_k} is larger than source k=${SOURCE_K}" >&2
    exit 2
  fi

  link_or_copy "$SOURCE_PATH" "$SRS_DIR/k${target_k}.srs"
  link_or_copy "$SOURCE_PATH" "$SRS_DIR/kzg${target_k}.srs"
done

echo
echo "[done] seeded aliases in $SRS_DIR"
ls -li "$SOURCE_PATH" "$SRS_DIR"/k*.srs "$SRS_DIR"/kzg*.srs 2>/dev/null || true
