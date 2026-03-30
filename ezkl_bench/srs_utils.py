from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Set


_SRS_NAME_RE = re.compile(r"^(?:kzg|k)(\d+)\.srs$")


@dataclass(frozen=True)
class SrsResolution:
    requested_logrows: int
    available_logrows: int
    path: Path
    exact_match: bool


def parse_srs_logrows(path: Path) -> Optional[int]:
    match = _SRS_NAME_RE.match(path.name)
    if match is None:
        return None
    return int(match.group(1))


def default_srs_search_dirs() -> List[Path]:
    dirs: List[Path] = []

    raw_dir = (os.environ.get("EZKL_SHARED_SRS_DIR") or "").strip()
    if raw_dir:
        dirs.append(Path(raw_dir))

    env_path = (os.environ.get("EZKL_SHARED_SRS_PATH") or "").strip()
    if env_path:
        p = Path(env_path)
        if p.is_dir():
            dirs.append(p)
        else:
            dirs.append(p.parent)

    dirs.append(Path.home() / ".ezkl" / "srs")
    return dirs


def _iter_candidate_files(
    *,
    preferred_paths: Sequence[Path],
    search_dirs: Sequence[Path],
) -> Iterable[Path]:
    seen: Set[Path] = set()

    for p in preferred_paths:
        try:
            key = p.resolve(strict=False)
        except Exception:
            key = p
        if key in seen:
            continue
        seen.add(key)
        if p.exists() and p.is_file():
            yield p

    for d in search_dirs:
        if not d.exists() or not d.is_dir():
            continue
        for p in sorted(d.glob("*.srs")):
            try:
                key = p.resolve(strict=False)
            except Exception:
                key = p
            if key in seen:
                continue
            seen.add(key)
            if p.is_file():
                yield p


def resolve_compatible_srs(
    requested_logrows: int,
    *,
    preferred_paths: Optional[Sequence[Path]] = None,
    search_dirs: Optional[Sequence[Path]] = None,
) -> Optional[SrsResolution]:
    preferred_paths = list(preferred_paths or [])

    merged_dirs: List[Path] = []
    seen_dirs: Set[Path] = set()
    for d in list(search_dirs or []) + default_srs_search_dirs():
        try:
            key = d.resolve(strict=False)
        except Exception:
            key = d
        if key in seen_dirs:
            continue
        seen_dirs.add(key)
        merged_dirs.append(d)

    preferred_index = {
        str(p.resolve(strict=False) if hasattr(p, "resolve") else p): idx for idx, p in enumerate(preferred_paths)
    }

    best: Optional[SrsResolution] = None
    best_score = None

    for candidate in _iter_candidate_files(preferred_paths=preferred_paths, search_dirs=merged_dirs):
        available_logrows = parse_srs_logrows(candidate)
        if available_logrows is None:
            continue
        if available_logrows < requested_logrows:
            continue

        exact_match = available_logrows == requested_logrows
        try:
            candidate_key = str(candidate.resolve(strict=False))
        except Exception:
            candidate_key = str(candidate)
        score = (
            0 if exact_match else 1,
            available_logrows - requested_logrows,
            preferred_index.get(candidate_key, len(preferred_paths) + 1),
            len(str(candidate)),
            str(candidate),
        )

        if best is None or score < best_score:
            best = SrsResolution(
                requested_logrows=requested_logrows,
                available_logrows=available_logrows,
                path=candidate,
                exact_match=exact_match,
            )
            best_score = score

    return best
