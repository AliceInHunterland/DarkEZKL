from __future__ import annotations

import logging
import shutil
import time
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def _looks_like_text_pointer_or_html(path: Path) -> bool:
    """
    Heuristic guard: the 'header too large' safetensors error almost always means the
    downloaded file isn't a real safetensors binary (git-lfs pointer, HTML, etc.).
    """
    try:
        if not path.exists() or path.stat().st_size < 1024:
            # tiny file => very likely pointer or error page
            return True

        head = path.read_bytes()[:512]
        # git-lfs pointer
        if b"git-lfs" in head and b"spec/v1" in head:
            return True
        # html error page
        if b"<html" in head.lower() or b"<!doctype html" in head.lower():
            return True
    except Exception:
        # If we can't read it, treat as suspicious so we can retry.
        return True
    return False


def _purge_dir(p: Path) -> None:
    try:
        if p.exists():
            shutil.rmtree(p)
    except Exception as e:
        logger.warning("Failed to purge cache dir %s: %s", p, e)


def create_timm_model_with_retries(
    model_name: str,
    *,
    cache_dir: Path,
    attempts: int = 4,
    backoff_s: float = 2.0,
    timm_kwargs: Optional[Dict[str, Any]] = None,
):
    """
    Load a timm model with real pretrained weights, with retries + cache purge
    to handle partially downloaded/corrupt artifacts (common in CI/Docker).

    Strategy:
      - Force a dedicated cache_dir so we can purge it safely.
      - On safetensors-related failures, wipe the cache_dir and retry.
      - Also scan for suspicious .safetensors files (git-lfs pointer / HTML).
    """
    import timm

    timm_kwargs = dict(timm_kwargs or {})
    # timm supports passing cache_dir, which is forwarded to HF hub downloader
    timm_kwargs["cache_dir"] = str(cache_dir)

    cache_dir.mkdir(parents=True, exist_ok=True)

    last_err: Optional[BaseException] = None
    for i in range(1, max(1, int(attempts)) + 1):
        logger.info("Loading timm pretrained weights: model=%s attempt=%s/%s cache_dir=%s", model_name, i, attempts, cache_dir)
        try:
            model = timm.create_model(model_name, pretrained=True, **timm_kwargs)
            model.eval()
            logger.info("Loaded pretrained weights successfully: model=%s", model_name)
            return model
        except BaseException as e:
            last_err = e
            msg = str(e)
            logger.error("Failed to load pretrained weights (attempt %s/%s): %s: %s", i, attempts, type(e).__name__, msg)

            # Try to detect obvious corruption/pointer files and purge cache before retry.
            try:
                safes = list(cache_dir.rglob("*.safetensors"))
                suspicious = [p for p in safes if _looks_like_text_pointer_or_html(p)]
                if suspicious:
                    logger.error("Detected suspicious safetensors files (will purge cache): %s", [str(p) for p in suspicious[:10]])
                    _purge_dir(cache_dir)
                    cache_dir.mkdir(parents=True, exist_ok=True)
            except Exception as scan_err:
                logger.warning("Cache scan failed: %s", scan_err)

            if i < attempts:
                sleep_s = backoff_s * (2 ** (i - 1))
                logger.info("Retrying in %.1fs...", sleep_s)
                time.sleep(sleep_s)

    raise RuntimeError(f"Failed to load timm pretrained model '{model_name}' after {attempts} attempts: {last_err}")
