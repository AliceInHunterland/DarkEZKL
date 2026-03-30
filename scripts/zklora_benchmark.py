#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple


ROOT = Path(__file__).resolve().parents[1]
ZKLORA_SRC = ROOT / "external" / "zkLoRA" / "src"
if str(ZKLORA_SRC) not in sys.path:
    sys.path.insert(0, str(ZKLORA_SRC))

torch = None
AutoModelForCausalLM = None
AutoTokenizer = None
PeftModel = None
export_lora_onnx_json_mpi = None
batch_verify_proofs = None
generate_proofs = None


def _slugify(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-")
    return slug or "run"


def _strip_prefix(raw_name: str) -> str:
    name = raw_name
    for prefix in ("base_model.model.", "base_model.", "model."):
        if name.startswith(prefix):
            name = name[len(prefix) :]
    return name.strip()


def _list_proofable_submodules(
    peft_model: torch.nn.Module, module_suffix: str
) -> Dict[str, torch.nn.Module]:
    out: Dict[str, torch.nn.Module] = {}
    for raw_name, module in peft_model.named_modules():
        has_lora_params = any(
            "lora" in pname.lower() for pname, _ in module.named_parameters()
        )
        if not has_lora_params:
            continue
        sub_name = _strip_prefix(raw_name)
        if not sub_name or "." not in sub_name:
            continue
        if module_suffix and not sub_name.endswith(module_suffix):
            continue
        out[sub_name] = module
    return out


def _capture_module_inputs(
    peft_model: torch.nn.Module,
    tokenizer,
    prompt: str,
    selected: Dict[str, torch.nn.Module],
) -> Dict[str, object]:
    captured: Dict[str, object] = {}
    handles = []

    def make_hook(name: str):
        def hook(_module, inputs):
            if name in captured or not inputs:
                return
            x = inputs[0]
            if torch.is_tensor(x):
                captured[name] = x.detach().cpu().float().numpy()

        return hook

    for sub_name, module in selected.items():
        handles.append(module.register_forward_pre_hook(make_hook(sub_name)))

    encoded = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        peft_model(**encoded)

    for handle in handles:
        handle.remove()
    return captured


def _export_selected_modules(
    run_dir: Path,
    selected: Dict[str, torch.nn.Module],
    captured_inputs: Dict[str, object],
    verbose: bool,
) -> List[str]:
    exported: List[str] = []
    for sub_name, module in selected.items():
        x_data = captured_inputs.get(sub_name)
        if x_data is None:
            continue
        export_lora_onnx_json_mpi(
            sub_name=sub_name,
            x_data=x_data,
            submodule=module,
            output_dir=str(run_dir),
            verbose=verbose,
        )
        exported.append(sub_name)
    return exported


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark zkLoRA proof generation on a configurable subset of LoRA submodules."
    )
    parser.add_argument("--base-model", default="distilgpt2")
    parser.add_argument("--lora-model-id", default="ng0-k1/distilgpt2-finetuned-es")
    parser.add_argument(
        "--prompt", default="Hello World, this is a LoRA benchmark prompt."
    )
    parser.add_argument(
        "--module-suffix",
        default="c_attn",
        help="Only benchmark proofable submodules whose stripped names end with this suffix.",
    )
    parser.add_argument(
        "--max-modules",
        type=int,
        default=1,
        help="Number of proofable submodules to export and benchmark.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory for exported ONNX/JSON/proof artifacts. Defaults under results/zklora.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Run batch proof verification after proof generation.",
    )
    parser.add_argument(
        "--capture-only",
        action="store_true",
        help="Stop after module capture and ONNX/JSON export.",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Enable verbose export/proof logging."
    )
    args = parser.parse_args()

    global torch
    global AutoModelForCausalLM
    global AutoTokenizer
    global PeftModel
    global export_lora_onnx_json_mpi
    global batch_verify_proofs
    global generate_proofs

    import torch as _torch
    from peft import PeftModel as _PeftModel
    from transformers import AutoModelForCausalLM as _AutoModelForCausalLM
    from transformers import AutoTokenizer as _AutoTokenizer
    from zklora.mpi_lora_onnx_exporter import (
        export_lora_onnx_json_mpi as _export_lora_onnx_json_mpi,
    )
    from zklora.zk_proof_generator import (
        batch_verify_proofs as _batch_verify_proofs,
        generate_proofs as _generate_proofs,
    )

    torch = _torch
    AutoModelForCausalLM = _AutoModelForCausalLM
    AutoTokenizer = _AutoTokenizer
    PeftModel = _PeftModel
    export_lora_onnx_json_mpi = _export_lora_onnx_json_mpi
    batch_verify_proofs = _batch_verify_proofs
    generate_proofs = _generate_proofs

    base_slug = _slugify(args.base_model)
    lora_slug = _slugify(args.lora_model_id)
    if args.out_dir:
        run_dir = Path(args.out_dir)
    else:
        run_dir = (
            ROOT
            / "results"
            / "zklora"
            / f"{base_slug}__{lora_slug}__m{args.max_modules}"
        )
    run_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print(f"[zklora-bench] loading base model: {args.base_model}", flush=True)
    base_model = AutoModelForCausalLM.from_pretrained(args.base_model)
    base_model.config.use_cache = False
    base_model.eval()

    print(f"[zklora-bench] loading LoRA adapter: {args.lora_model_id}", flush=True)
    peft_model = PeftModel.from_pretrained(base_model, args.lora_model_id)
    peft_model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    load_s = time.time() - t0

    proofable = _list_proofable_submodules(peft_model, args.module_suffix)
    proofable_names = list(proofable.keys())
    selected_names = proofable_names[: max(args.max_modules, 0)]
    selected = {name: proofable[name] for name in selected_names}

    print(
        f"[zklora-bench] discovered {len(proofable_names)} proofable modules; "
        f"selected {len(selected_names)}",
        flush=True,
    )
    for name in selected_names:
        print(f"[zklora-bench] selected module: {name}", flush=True)

    t1 = time.time()
    captured_inputs = _capture_module_inputs(peft_model, tokenizer, args.prompt, selected)
    capture_s = time.time() - t1
    print(
        f"[zklora-bench] captured inputs for {len(captured_inputs)} selected modules",
        flush=True,
    )

    t2 = time.time()
    exported_names = _export_selected_modules(
        run_dir=run_dir,
        selected=selected,
        captured_inputs=captured_inputs,
        verbose=args.verbose,
    )
    export_s = time.time() - t2
    print(f"[zklora-bench] exported {len(exported_names)} modules", flush=True)

    summary = {
        "base_model": args.base_model,
        "lora_model_id": args.lora_model_id,
        "module_suffix": args.module_suffix,
        "max_modules": args.max_modules,
        "prompt": args.prompt,
        "run_dir": str(run_dir),
        "load_s": load_s,
        "capture_s": capture_s,
        "export_s": export_s,
        "proofable_module_count": len(proofable_names),
        "selected_modules": selected_names,
        "captured_module_count": len(captured_inputs),
        "exported_modules": exported_names,
    }

    if not args.capture_only and exported_names:
        t3 = time.time()
        proof_stats = asyncio.run(
            generate_proofs(
                onnx_dir=str(run_dir),
                json_dir=str(run_dir),
                output_dir=str(run_dir),
                verbose=args.verbose,
            )
        )
        proof_elapsed_s = time.time() - t3
        summary["proof_elapsed_s"] = proof_elapsed_s
        summary["proof_stats"] = proof_stats

        if args.verify:
            t4 = time.time()
            verify_stats = batch_verify_proofs(proof_dir=str(run_dir), verbose=args.verbose)
            summary["verify_elapsed_s"] = time.time() - t4
            summary["verify_stats"] = verify_stats

    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"[zklora-bench] wrote summary to {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
