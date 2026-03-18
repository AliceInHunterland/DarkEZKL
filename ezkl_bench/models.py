from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn

from .timm_weights import create_timm_model_with_retries

logger = logging.getLogger(__name__)


class LeNet5(nn.Module):
    """
    Classic-ish LeNet-5.
    Modified to support variable input side for benchmarking scaling.
    Standard LeNet-5 is 28x28 input.
    """

    def __init__(self, num_classes: int = 10, input_side: int = 28):
        super().__init__()
        self.input_side = input_side

        s = input_side
        s = s - 4
        s = s // 2
        s = s - 4
        s = s // 2
        self.final_side = s

        if self.final_side < 1:
            raise ValueError(f"Input side {input_side} too small for LeNet5")

        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.AvgPool2d(2, 2)

        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.AvgPool2d(2, 2)

        self.fc1 = nn.Linear(16 * self.final_side * self.final_side, 120)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(x.shape[0], -1)
        x = self.relu3(self.fc1(x))
        x = self.relu4(self.fc2(x))
        return self.fc3(x)


class NanoMLP(nn.Module):
    """
    Tiny MLP for ultra-fast benchmarking sanity checks.
    """

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(16, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 10)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


def _default_cache_dir() -> Path:
    """
    Default cache dir for weights/datasets.

    NOTE: this is intentionally local/relative-by-default to behave well in CI and
    Docker environments where $HOME may be read-only or unset.
    """
    env = os.environ.get("EZKL_BENCH_CACHE_DIR", "").strip()
    if env:
        return Path(env)
    return Path(".cache") / "ezkl_bench"


def _get_mnist_sample(cache_dir: Path, side: int, channels: int = 1) -> torch.Tensor:
    """
    Best-effort MNIST sample fetcher.

    If torchvision isn't installed or MNIST can't be downloaded (offline CI),
    fall back to random data with the correct shape.
    """
    try:
        import torchvision
    except Exception as e:
        logger.warning("torchvision unavailable (%s). Falling back to random input.", e)
        return torch.rand(1, channels, side, side, dtype=torch.float32)

    data_dir = cache_dir / "datasets"
    data_dir.mkdir(parents=True, exist_ok=True)

    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize((side, side)),
            torchvision.transforms.ToTensor(),
        ]
    )

    try:
        ds = torchvision.datasets.MNIST(root=str(data_dir), train=False, download=True, transform=transform)
    except Exception as e:
        logger.error("Failed to download MNIST: %s. Fallback to random data.", e)
        return torch.rand(1, channels, side, side, dtype=torch.float32)

    img = ds[0][0].unsqueeze(0)

    if channels == 3:
        img = img.repeat(1, 3, 1, 1)

    return img


def _train_lenet_quick(ckpt_path: Path, data_dir: Path) -> None:
    try:
        import torchvision  # lazy import
    except Exception as e:
        raise RuntimeError(f"torchvision is required to train LeNet quick checkpoint: {e}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LeNet5(input_side=28).to(device)
    model.train()

    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    train_ds = torchvision.datasets.MNIST(root=str(data_dir), train=True, download=True, transform=transform)

    subset = torch.utils.data.Subset(train_ds, list(range(0, min(5000, len(train_ds)))))
    loader = torch.utils.data.DataLoader(subset, batch_size=64, shuffle=True, num_workers=2, pin_memory=True)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    for _epoch in range(1):
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()

    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt_path)


def get_lenet5_pretrained(cache_dir: Path, input_side: int = 28) -> Tuple[nn.Module, torch.Tensor]:
    """
    Returns a LeNet-5 model and a dummy input.

    - For input_side == 28, tries to use a locally cached quick-trained checkpoint,
      training it if needed (requires torchvision + MNIST download).
    - For other input sizes, returns randomly initialized weights (still valid for
      architectural benchmarking) to avoid forced downloads.
    """
    model = LeNet5(input_side=input_side)

    if input_side == 28:
        ckpt_path = cache_dir / "weights" / "lenet5_mnist_quick.pth"
        data_dir = cache_dir / "datasets"
        try:
            if ckpt_path.exists():
                state = torch.load(ckpt_path, map_location="cpu")
                model.load_state_dict(state)
            else:
                _train_lenet_quick(ckpt_path=ckpt_path, data_dir=data_dir)
                state = torch.load(ckpt_path, map_location="cpu")
                model.load_state_dict(state)
        except Exception as e:
            # Don't fail import/tests in offline environments; keep random init.
            logger.warning("LeNet quick-pretrain unavailable (%s). Using random weights.", e)

    model.eval()
    dummy_input = _get_mnist_sample(cache_dir, input_side, channels=1)
    return model, dummy_input


def get_nano_mlp(cache_dir: Path = None) -> Tuple[nn.Module, torch.Tensor]:
    model = NanoMLP()
    model.eval()
    dummy_input = torch.rand(1, 16, dtype=torch.float32)
    return model, dummy_input


def _create_timm_model_best_effort(model_name: str, *, cache_dir: Path) -> nn.Module:
    """
    Prefer real pretrained weights, but if that fails (offline CI / HF issues),
    fall back to random-init weights so benchmarks/tests can still run.
    """
    weights_cache = cache_dir / "weights" / "timm_hf"
    try:
        return create_timm_model_with_retries(
            model_name,
            cache_dir=weights_cache,
            attempts=4,
            backoff_s=2.0,
        )
    except Exception as e:
        logger.warning(
            "Failed to load timm pretrained weights for %s (%s). Falling back to pretrained=False.",
            model_name,
            e,
        )
        import timm

        model = timm.create_model(model_name, pretrained=False)
        model.eval()
        return model


def get_vit_tiny_pretrained(cache_dir: Path) -> Tuple[nn.Module, torch.Tensor]:
    model = _create_timm_model_best_effort("vit_tiny_patch16_224", cache_dir=cache_dir)
    dummy_input = _get_mnist_sample(cache_dir, 224, channels=3)
    return model, dummy_input


def get_repvgg_a0_pretrained(cache_dir: Path) -> Tuple[nn.Module, torch.Tensor]:
    model = _create_timm_model_best_effort("repvgg_a0", cache_dir=cache_dir)

    if hasattr(model, "switch_to_deploy"):
        model.switch_to_deploy()
    model.eval()

    dummy_input = _get_mnist_sample(cache_dir, 224, channels=3)
    return model, dummy_input


def get_model(name: str, cache_dir: Optional[Path] = None) -> Tuple[nn.Module, torch.Tensor]:
    """
    Factory function for benchmark targets.

    Supports:
      - 'vit'           -> timm vit_tiny_patch16_224
      - 'lenet-5-small' -> local LeNet5 (input_side=16)
      - 'repvgg-a0'     -> timm repvgg_a0

    Note: cache_dir controls where datasets/weights are cached. If not provided,
    defaults to EZKL_BENCH_CACHE_DIR or ./.cache/ezkl_bench.
    """
    cache_dir = Path(cache_dir) if cache_dir is not None else _default_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)

    key = (name or "").strip().lower()

    if key == "vit":
        return get_vit_tiny_pretrained(cache_dir)
    if key in {"lenet-5-small", "lenet5-small", "lenet_small", "lenet-small"}:
        # "small" here matches the prior bench's 16x16 variant.
        return get_lenet5_pretrained(cache_dir, input_side=16)
    if key in {"repvgg-a0", "repvgg_a0", "repvgg"}:
        return get_repvgg_a0_pretrained(cache_dir)

    raise ValueError(
        f"Unknown model '{name}'. Supported: vit, lenet-5-small, repvgg-a0"
    )


@dataclass(frozen=True)
class ModelSpec:
    key: str
    display_name: str
    factory: Callable[[], Tuple[nn.Module, torch.Tensor]]
    input_scale: int = 7
    param_scale: int = 7
    num_inner_cols: int = 2

    enable_onnx_split: bool = False
    split_min_params: Optional[int] = None
    split_max_nodes: Optional[int] = None
    max_segment_logrows: Optional[int] = None
    max_segment_rows: Optional[int] = None
    max_segment_assignments: Optional[int] = None
    precision: Optional[str] = None

    # Post-export ONNX patching knobs
    # - rewrite_gemm: fixes ezkl/tract failures on Gemm for certain exported graphs
    rewrite_gemm: bool = False


def get_model_specs(cache_dir: Path) -> Dict[str, ModelSpec]:
    """
    Existing spec registry used by the older benchmark runner.

    Note: we keep backwards-compatible keys (lenet/repvgg/etc.), and also provide
    aliases matching the new benchmark naming ('lenet-5-small', 'repvgg-a0').
    """
    specs: Dict[str, ModelSpec] = {
        "lenet": ModelSpec(
            key="lenet",
            display_name="LeNet-5-Small",
            factory=lambda: get_lenet5_pretrained(cache_dir, input_side=16),
            input_scale=3,
            param_scale=3,
            enable_onnx_split=False,
            precision="fp32",
            rewrite_gemm=True,
        ),
        "lenet-medium": ModelSpec(
            key="lenet-medium",
            display_name="LeNet-5-Medium",
            factory=lambda: get_lenet5_pretrained(cache_dir, input_side=32),
            input_scale=3,
            param_scale=3,
            enable_onnx_split=False,
            precision="fp32",
            rewrite_gemm=True,
        ),
        "nano": ModelSpec(
            key="nano",
            display_name="Nano-MLP",
            factory=lambda: get_nano_mlp(cache_dir),
            input_scale=7,
            param_scale=7,
            enable_onnx_split=False,
            precision="fp32",
            rewrite_gemm=False,
        ),
        "vit": ModelSpec(
            key="vit",
            display_name="ViT-Tiny",
            factory=lambda: get_vit_tiny_pretrained(cache_dir),
            input_scale=3,
            param_scale=3,
            num_inner_cols=16,
            enable_onnx_split=True,
            precision=None,
            rewrite_gemm=False,
        ),
        "repvgg": ModelSpec(
            key="repvgg",
            display_name="RepVGG-A0",
            factory=lambda: get_repvgg_a0_pretrained(cache_dir),
            input_scale=3,
            param_scale=3,
            num_inner_cols=8,
            enable_onnx_split=True,
            split_min_params=10_000,
            split_max_nodes=12,
            max_segment_logrows=23,
            max_segment_rows=8_000_000,
            max_segment_assignments=64_000_000,
            precision=None,
            rewrite_gemm=False,
        ),
    }

    # Aliases for the new benchmark naming (do not remove old keys).
    specs["lenet-5-small"] = ModelSpec(
        key="lenet-5-small",
        display_name="LeNet-5-Small",
        factory=lambda: get_lenet5_pretrained(cache_dir, input_side=16),
        input_scale=3,
        param_scale=3,
        enable_onnx_split=False,
        precision="fp32",
        rewrite_gemm=True,
    )
    specs["repvgg-a0"] = ModelSpec(
        key="repvgg-a0",
        display_name="RepVGG-A0",
        factory=lambda: get_repvgg_a0_pretrained(cache_dir),
        input_scale=3,
        param_scale=3,
        num_inner_cols=8,
        enable_onnx_split=True,
        split_min_params=10_000,
        split_max_nodes=12,
        max_segment_logrows=23,
        max_segment_rows=8_000_000,
        max_segment_assignments=64_000_000,
        precision=None,
        rewrite_gemm=False,
    )

    return specs
