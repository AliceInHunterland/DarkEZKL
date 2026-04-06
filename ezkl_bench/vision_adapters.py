from __future__ import annotations

import hashlib
import inspect
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from .utils import ensure_dir, slugify, write_json, write_json_compact

logger = logging.getLogger(__name__)


def _parameter_count(module: nn.Module) -> int:
    return int(sum(int(p.numel()) for p in module.parameters()))


def _shape_list(tensor: torch.Tensor) -> List[int]:
    return [int(x) for x in tensor.shape]


def _safe_name(name: str) -> str:
    return name.replace("/", "__").replace(".", "__")


def _torch_export_kwargs() -> Dict[str, object]:
    out: Dict[str, object] = {}
    try:
        sig = inspect.signature(torch.onnx.export)
        if "use_external_data_format" in sig.parameters:
            out["use_external_data_format"] = True
        elif "large_model" in sig.parameters:
            out["large_model"] = True
    except Exception:
        pass
    return out


class LinearAdapterBranch(nn.Module):
    def __init__(self, in_features: int, out_features: int, rank: int, alpha: float) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.rank = max(1, min(int(rank), self.in_features, self.out_features))
        self.alpha = float(alpha)
        self.scale = self.alpha / float(self.rank)
        self.down = nn.Linear(self.in_features, self.rank, bias=False)
        self.up = nn.Linear(self.rank, self.out_features, bias=False)
        nn.init.kaiming_uniform_(self.down.weight, a=5**0.5)
        nn.init.kaiming_uniform_(self.up.weight, a=5**0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(self.down(x)) * self.scale


class Conv2dAdapterBranch(nn.Module):
    def __init__(self, module: nn.Conv2d, rank: int, alpha: float) -> None:
        super().__init__()
        if int(module.groups) != 1:
            raise ValueError(f"Conv2d groups={module.groups} is unsupported for adapter export")

        in_channels = int(module.in_channels)
        out_channels = int(module.out_channels)
        self.rank = max(1, min(int(rank), in_channels, out_channels))
        self.alpha = float(alpha)
        self.scale = self.alpha / float(self.rank)

        self.down = nn.Conv2d(
            in_channels=in_channels,
            out_channels=self.rank,
            kernel_size=module.kernel_size,
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            groups=1,
            bias=False,
        )
        self.up = nn.Conv2d(
            in_channels=self.rank,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        nn.init.kaiming_uniform_(self.down.weight, a=5**0.5)
        nn.init.kaiming_uniform_(self.up.weight, a=5**0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(self.down(x)) * self.scale


@dataclass
class AdapterSkip:
    name: str
    reason: str


@dataclass
class AdapterCandidate:
    name: str
    module_type: str
    base_module: nn.Module
    adapter_module: nn.Module
    base_param_count: int
    adapter_param_count: int
    rank: int
    alpha: float


def build_adapter_for_module(module: nn.Module, *, rank: int, alpha: float) -> nn.Module:
    if isinstance(module, nn.Linear):
        return LinearAdapterBranch(
            in_features=int(module.in_features),
            out_features=int(module.out_features),
            rank=int(rank),
            alpha=float(alpha),
        )
    if isinstance(module, nn.Conv2d):
        return Conv2dAdapterBranch(module, rank=int(rank), alpha=float(alpha))
    raise TypeError(f"Unsupported module type for adapters: {type(module).__name__}")


def discover_adapter_candidates(
    model: nn.Module,
    *,
    rank: int,
    alpha: float,
    min_base_params: int = 0,
    max_modules: Optional[int] = None,
    include_module_types: Sequence[str] = ("linear", "conv2d"),
) -> Tuple[List[AdapterCandidate], List[AdapterSkip]]:
    include = {str(x).strip().lower() for x in include_module_types if str(x).strip()}
    candidates: List[AdapterCandidate] = []
    skipped: List[AdapterSkip] = []

    for name, module in model.named_modules():
        if not name:
            continue

        module_type = None
        if isinstance(module, nn.Linear) and "linear" in include:
            module_type = "linear"
        elif isinstance(module, nn.Conv2d) and "conv2d" in include:
            module_type = "conv2d"
        else:
            continue

        base_param_count = _parameter_count(module)
        if base_param_count < int(min_base_params):
            skipped.append(AdapterSkip(name=name, reason=f"below_min_base_params:{base_param_count}"))
            continue

        try:
            adapter_module = build_adapter_for_module(module, rank=rank, alpha=alpha).eval()
        except Exception as exc:
            skipped.append(AdapterSkip(name=name, reason=str(exc)))
            continue

        candidates.append(
            AdapterCandidate(
                name=name,
                module_type=module_type,
                base_module=module,
                adapter_module=adapter_module,
                base_param_count=base_param_count,
                adapter_param_count=_parameter_count(adapter_module),
                rank=int(getattr(adapter_module, "rank", rank)),
                alpha=float(alpha),
            )
        )

    candidates.sort(key=lambda item: (-item.base_param_count, item.name))
    if max_modules is not None and int(max_modules) > 0:
        candidates = candidates[: int(max_modules)]
    return candidates, skipped


def capture_candidate_inputs(
    model: nn.Module,
    dummy_input: torch.Tensor,
    candidates: Sequence[AdapterCandidate],
) -> Dict[str, torch.Tensor]:
    captured: Dict[str, torch.Tensor] = {}
    handles = []

    def make_hook(name: str):
        def hook(_module: nn.Module, inputs: Tuple[object, ...]) -> None:
            if name in captured or not inputs:
                return
            x = inputs[0]
            if torch.is_tensor(x):
                captured[name] = x.detach().cpu().float().clone()

        return hook

    for candidate in candidates:
        handles.append(candidate.base_module.register_forward_pre_hook(make_hook(candidate.name)))

    model_cpu = model.eval().cpu()
    with torch.no_grad():
        model_cpu(dummy_input.detach().cpu())

    for handle in handles:
        handle.remove()

    return captured


def export_adapter_artifacts(
    *,
    candidate: AdapterCandidate,
    example_input: torch.Tensor,
    out_dir: Path,
) -> Dict[str, object]:
    ensure_dir(out_dir)
    adapter_cpu = candidate.adapter_module.eval().cpu()
    input_cpu = example_input.detach().cpu().float()

    safe_name = _safe_name(candidate.name)
    onnx_path = out_dir / f"{safe_name}.onnx"
    input_path = out_dir / f"{safe_name}_input.json"
    metadata_path = out_dir / f"{safe_name}_metadata.json"

    torch.onnx.export(
        adapter_cpu,
        input_cpu,
        str(onnx_path),
        input_names=["input"],
        output_names=["output"],
        opset_version=14,
        do_constant_folding=True,
        **_torch_export_kwargs(),
    )

    try:
        from .onnx_sanitize import sanitize_exported_onnx_inplace

        sanitize_exported_onnx_inplace(onnx_path, rewrite_gemm=False)
    except Exception as exc:
        logger.warning("Skipping ONNX sanitize for %s: %s", onnx_path, exc)

    payload = {"input_data": [np.asarray(input_cpu, dtype=np.float32).reshape(-1).tolist()]}
    write_json_compact(input_path, payload)

    metadata = {
        "name": candidate.name,
        "module_type": candidate.module_type,
        "base_param_count": int(candidate.base_param_count),
        "adapter_param_count": int(candidate.adapter_param_count),
        "rank": int(candidate.rank),
        "alpha": float(candidate.alpha),
        "input_shape": _shape_list(input_cpu),
        "artifact_hash": hashlib.sha256(candidate.name.encode("utf-8")).hexdigest()[:12],
    }
    write_json(metadata_path, metadata)

    return {
        "onnx_path": onnx_path,
        "input_path": input_path,
        "metadata_path": metadata_path,
        "safe_name": safe_name,
        "input_shape": metadata["input_shape"],
    }


def model_parameter_count(model: nn.Module) -> int:
    return _parameter_count(model)


def adapter_coverage_summary(
    *,
    model: nn.Module,
    selected: Sequence[AdapterCandidate],
    skipped: Sequence[AdapterSkip],
) -> Dict[str, object]:
    total_model_params = model_parameter_count(model)
    selected_base = int(sum(item.base_param_count for item in selected))
    selected_adapter = int(sum(item.adapter_param_count for item in selected))
    eligible_names = {item.name for item in selected}
    return {
        "model_param_count": int(total_model_params),
        "selected_module_count": int(len(selected)),
        "selected_base_param_count": selected_base,
        "selected_adapter_param_count": selected_adapter,
        "selected_base_param_ratio": float(selected_base / total_model_params) if total_model_params else 0.0,
        "selected_adapter_param_ratio": float(selected_adapter / total_model_params) if total_model_params else 0.0,
        "skipped_module_count": int(len(skipped)),
        "selected_modules": [item.name for item in selected],
        "selected_module_slug": slugify("-".join(item.name for item in selected)[:120]) if selected else "none",
        "skipped": [{"name": item.name, "reason": item.reason} for item in skipped],
        "eligible_names": sorted(eligible_names),
    }
