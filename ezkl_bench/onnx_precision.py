from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import onnx

logger = logging.getLogger(__name__)


@dataclass
class OnnxPrecisionResult:
    requested: str
    applied: str
    ok: bool
    error: Optional[str] = None
    details: Dict[str, Any] = None  # type: ignore


def _safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def convert_onnx_to_fp16(src: Path, dst: Path, *, keep_io_types: bool = True) -> None:
    """
    Converts ONNX float tensors to float16.

    keep_io_types=True is important:
      - Inputs/outputs stay float32 (stable interface)
      - Weights/internal ops become float16 (smaller + often faster)
    """
    from onnxconverter_common import float16  # type: ignore

    logger.info("Converting ONNX to fp16: %s -> %s (keep_io_types=%s)", src, dst, keep_io_types)
    model = onnx.load(str(src))
    model_fp16 = float16.convert_float_to_float16(model, keep_io_types=bool(keep_io_types))
    onnx.save(model_fp16, str(dst))


def quantize_onnx_to_int8_dynamic(src: Path, dst: Path) -> None:
    """
    Best-effort INT8 dynamic quantization via onnxruntime.
    NOTE: ezkl may not support all quantized (Q/DQ) graphs; this is "try and fallback".
    """
    from onnxruntime.quantization import QuantType, quantize_dynamic  # type: ignore

    logger.info("Quantizing ONNX to int8 (dynamic): %s -> %s", src, dst)
    quantize_dynamic(
        model_input=str(src),
        model_output=str(dst),
        weight_type=QuantType.QInt8,
        optimize_model=True,
    )


def maybe_convert_onnx_precision(
    onnx_path: Path,
    *,
    out_dir: Path,
    mode: str,
) -> Tuple[Path, OnnxPrecisionResult]:
    """
    Returns (path_to_use, meta).

    mode:
      - "fp16" (default recommended)
      - "fp32" (no conversion)
      - "int8" (try int8, fallback to fp16, fallback to fp32)
    """
    req = (mode or "fp16").strip().lower()
    _safe_mkdir(out_dir)

    # No-op
    if req in ("fp32", "float32", "f32", "none"):
        return onnx_path, OnnxPrecisionResult(requested=req, applied="fp32", ok=True, details={})

    # INT8 (best-effort)
    if req in ("int8", "i8", "qint8"):
        int8_path = out_dir / "network.int8.onnx"
        try:
            quantize_onnx_to_int8_dynamic(onnx_path, int8_path)
            return int8_path, OnnxPrecisionResult(requested=req, applied="int8", ok=True, details={"path": str(int8_path)})
        except Exception as e:
            logger.warning("INT8 quantization failed; will fallback to fp16. err=%s", e)

    # FP16
    fp16_path = out_dir / "network.fp16.onnx"
    try:
        convert_onnx_to_fp16(onnx_path, fp16_path, keep_io_types=True)
        return fp16_path, OnnxPrecisionResult(requested=req, applied="fp16", ok=True, details={"path": str(fp16_path)})
    except Exception as e:
        logger.warning("FP16 conversion failed; using fp32. err=%s", e)
        return (
            onnx_path,
            OnnxPrecisionResult(
                requested=req,
                applied="fp32",
                ok=False,
                error=f"{type(e).__name__}: {e}",
                details={"fallback": "fp32"},
            ),
        )
