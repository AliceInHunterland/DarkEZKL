from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import onnx
from onnx import TensorProto, helper

logger = logging.getLogger(__name__)


def _safe_infer_shapes(model: onnx.ModelProto) -> onnx.ModelProto:
    try:
        return onnx.shape_inference.infer_shapes(model)
    except Exception as e:
        logger.warning("ONNX shape inference failed (ignored): %s", e)
        return model


def _get_opset(model: onnx.ModelProto) -> Optional[int]:
    try:
        if model.opset_import:
            return int(model.opset_import[0].version)
    except Exception:
        pass
    return None


def _get_attr_float(node: onnx.NodeProto, name: str, default: float) -> float:
    for a in node.attribute:
        if a.name == name and a.type == onnx.AttributeProto.FLOAT:
            return float(a.f)
    return float(default)


def _get_attr_int(node: onnx.NodeProto, name: str, default: int) -> int:
    for a in node.attribute:
        if a.name == name and a.type == onnx.AttributeProto.INT:
            return int(a.i)
    return int(default)


def _make_scalar_initializer(name: str, value: float) -> onnx.TensorProto:
    return helper.make_tensor(name=name, data_type=TensorProto.FLOAT, dims=[], vals=[float(value)])


def rewrite_gemm_to_matmul_add_inplace(model: onnx.ModelProto) -> Dict[str, Any]:
    """
    Rewrite Gemm nodes to (Transpose?) + MatMul + (Mul alpha?) + (Mul beta? + Add bias?)
    to avoid ezkl/tract Gemm translation issues.
    """
    g = model.graph
    new_nodes: List[onnx.NodeProto] = []
    added_inits: List[onnx.TensorProto] = []

    rewrites = 0

    for node in g.node:
        if node.op_type != "Gemm":
            new_nodes.append(node)
            continue

        # Gemm inputs: A, B, C(optional)
        if len(node.input) < 2 or len(node.output) < 1:
            new_nodes.append(node)
            continue

        a_in = node.input[0]
        b_in = node.input[1]
        c_in = node.input[2] if len(node.input) >= 3 and node.input[2] else ""

        y_out = node.output[0]
        base = node.name or f"gemm_{rewrites}"

        alpha = _get_attr_float(node, "alpha", 1.0)
        beta = _get_attr_float(node, "beta", 1.0)
        transA = _get_attr_int(node, "transA", 0)
        transB = _get_attr_int(node, "transB", 0)

        a_eff = a_in
        if transA == 1:
            a_t = f"{base}__A_T"
            new_nodes.append(helper.make_node("Transpose", inputs=[a_in], outputs=[a_t], name=f"{base}__TransposeA"))
            a_eff = a_t

        b_eff = b_in
        if transB == 1:
            b_t = f"{base}__B_T"
            new_nodes.append(helper.make_node("Transpose", inputs=[b_in], outputs=[b_t], name=f"{base}__TransposeB"))
            b_eff = b_t

        mm = f"{base}__MatMul"
        new_nodes.append(helper.make_node("MatMul", inputs=[a_eff, b_eff], outputs=[mm], name=f"{base}__MatMul"))

        mm2 = mm
        if abs(alpha - 1.0) > 1e-12:
            alpha_name = f"{base}__alpha"
            added_inits.append(_make_scalar_initializer(alpha_name, alpha))
            mm_alpha = f"{base}__MulAlpha"
            new_nodes.append(helper.make_node("Mul", inputs=[mm, alpha_name], outputs=[mm_alpha], name=f"{base}__MulAlpha"))
            mm2 = mm_alpha

        if c_in:
            c_eff = c_in
            if abs(beta - 1.0) > 1e-12:
                beta_name = f"{base}__beta"
                added_inits.append(_make_scalar_initializer(beta_name, beta))
                c_beta = f"{base}__MulBeta"
                new_nodes.append(helper.make_node("Mul", inputs=[c_in, beta_name], outputs=[c_beta], name=f"{base}__MulBeta"))
                c_eff = c_beta

            new_nodes.append(helper.make_node("Add", inputs=[mm2, c_eff], outputs=[y_out], name=f"{base}__AddBias"))
        else:
            # No bias -> just rename output to original
            new_nodes.append(helper.make_node("Identity", inputs=[mm2], outputs=[y_out], name=f"{base}__IdentityOut"))

        rewrites += 1

    if rewrites:
        logger.info("Rewrote Gemm nodes: count=%s", rewrites)
        del g.node[:]
        g.node.extend(new_nodes)
        if added_inits:
            g.initializer.extend(added_inits)

    return {"gemm_rewrites": rewrites, "added_initializers": len(added_inits)}


def sanitize_exported_onnx_inplace(path: Path, *, rewrite_gemm: bool) -> Dict[str, Any]:
    """
    Post-process exported/conversion ONNX to improve stability:
      - shape/type inference (best-effort)
      - optional Gemm rewrite (tract stability)
    """
    if not path.exists():
        return {"ok": False, "error": f"missing: {path}"}

    model = onnx.load(str(path))
    meta: Dict[str, Any] = {"path": str(path), "opset": _get_opset(model)}

    # Inference before rewrite helps some graphs, and also helps us debug.
    model = _safe_infer_shapes(model)

    if rewrite_gemm:
        meta.update(rewrite_gemm_to_matmul_add_inplace(model))
        model = _safe_infer_shapes(model)

    onnx.save(model, str(path))
    meta["ok"] = True
    return meta


def _extract_node_debug(model: onnx.ModelProto, *, idx: Optional[int] = None, name: Optional[str] = None) -> Dict[str, Any]:
    nodes = list(model.graph.node)
    if idx is None and name is not None:
        for i, n in enumerate(nodes):
            if n.name == name:
                idx = i
                break

    if idx is None or idx < 0 or idx >= len(nodes):
        return {"found": False, "reason": "node not found", "requested_idx": idx, "requested_name": name}

    n = nodes[idx]
    attrs = {}
    for a in n.attribute:
        try:
            if a.type == onnx.AttributeProto.INT:
                attrs[a.name] = int(a.i)
            elif a.type == onnx.AttributeProto.FLOAT:
                attrs[a.name] = float(a.f)
            elif a.type == onnx.AttributeProto.STRING:
                attrs[a.name] = a.s.decode("utf-8", errors="replace")
            else:
                attrs[a.name] = f"<type={a.type}>"
        except Exception:
            attrs[a.name] = "<unreadable>"

    return {
        "found": True,
        "idx": idx,
        "name": n.name,
        "op_type": n.op_type,
        "inputs": list(n.input),
        "outputs": list(n.output),
        "attributes": attrs,
    }


def onnx_debug_dump_for_exception(exc: BaseException, candidate_onnx_paths: Iterable[Path]) -> None:
    """
    Print high-signal debug when we see known failure patterns (tract Gemm / ORT type mismatch).
    This is meant to keep logs useful without dumping entire models.
    """
    s = str(exc)

    # Pattern 1: ezkl/tract translation node error
    m = re.search(r'Translating node #(\d+)\s+"([^"]+)"\s+(\w+)', s)
    if m:
        idx = int(m.group(1))
        node_name = m.group(2)
        op_type = m.group(3)
        logger.error("ONNX debug: tract translation failed at node_idx=%s name=%s op=%s", idx, node_name, op_type)

        for p in candidate_onnx_paths:
            if not p.exists():
                continue
            try:
                model = onnx.load(str(p))
                logger.error("ONNX debug: file=%s opset=%s ir_version=%s", p, _get_opset(model), model.ir_version)
                # dump exact and nearby nodes
                dbg = _extract_node_debug(model, idx=idx)
                logger.error("ONNX debug: node=%s", dbg)
                if idx > 0:
                    logger.error("ONNX debug: prev_node=%s", _extract_node_debug(model, idx=idx - 1))
                if idx + 1 < len(model.graph.node):
                    logger.error("ONNX debug: next_node=%s", _extract_node_debug(model, idx=idx + 1))
                break
            except Exception as e:
                logger.warning("ONNX debug load failed for %s: %s", p, e)
        return

    # Pattern 2: ORT type mismatch on output arg
    m2 = re.search(r"Type .* of output arg \(([^)]+)\).*expected type \(([^)]+)\)", s)
    if m2:
        tensor = m2.group(1)
        expected = m2.group(2)
        logger.error("ONNX debug: ORT type mismatch tensor=%s expected=%s", tensor, expected)
        for p in candidate_onnx_paths:
            if not p.exists():
                continue
            try:
                model = onnx.load(str(p))
                # show any matching value_info / output definitions
                outs = [o for o in model.graph.output if o.name == tensor]
                vis = [v for v in model.graph.value_info if v.name == tensor]
                logger.error("ONNX debug: file=%s opset=%s", p, _get_opset(model))
                logger.error("ONNX debug: graph.output matches=%s", [o.name for o in outs])
                logger.error("ONNX debug: graph.value_info matches=%s", [v.name for v in vis])
                break
            except Exception:
                pass
        return


def onnx_debug_dump_for_exception_paths_only(exc: BaseException, candidate_onnx_paths: Iterable[Path]) -> None:
    logger.error("ONNX debug: exception=%s candidate_onnx_paths=%s", exc, [str(p) for p in candidate_onnx_paths])


def onnx_debug_dump_for_exception(exc: BaseException, candidate_onnx_paths: Iterable[Path]) -> None:
    # Keep wrapper stable, even if parsing fails.
    try:
        onnx_debug_dump_for_exception_paths_only(exc, candidate_onnx_paths)
        onnx_debug_dump_for_exception.__wrapped__  # type: ignore[attr-defined]
    except Exception:
        pass
    # Run full debug (best-effort)
    try:
        # call the inner logic defined above (same name, but we keep it simple by re-calling)
        s = str(exc)
        _ = s  # quiet linters
        # actually run the logic block above by re-importing ourselves (no-op),
        # then just call the local parsing function:
        # (we can't easily reference the previous definition after rebinds in some editors)
    except Exception:
        pass

    # The “real” logic is in the earlier function; easiest is to call it via a new name.
    try:
        _onnx_debug_dump_for_exception_impl(exc, candidate_onnx_paths)  # type: ignore[name-defined]
    except NameError:
        # In case an editor/tool collapses definitions, fall back to a minimal dump:
        onnx_debug_dump_for_exception_paths_only(exc, candidate_onnx_paths)


def _onnx_debug_dump_for_exception_impl(exc: BaseException, candidate_onnx_paths: Iterable[Path]) -> None:
    # This is the implementation (identical to the earlier parser) but isolated to ensure it runs.
    s = str(exc)

    m = re.search(r'Translating node #(\d+)\s+"([^"]+)"\s+(\w+)', s)
    if m:
        idx = int(m.group(1))
        node_name = m.group(2)
        op_type = m.group(3)
        logger.error("ONNX debug: tract translation failed at node_idx=%s name=%s op=%s", idx, node_name, op_type)

        for p in candidate_onnx_paths:
            if not p.exists():
                continue
            try:
                model = onnx.load(str(p))
                logger.error("ONNX debug: file=%s opset=%s ir_version=%s", p, _get_opset(model), model.ir_version)
                dbg = _extract_node_debug(model, idx=idx)
                logger.error("ONNX debug: node=%s", dbg)
                if idx > 0:
                    logger.error("ONNX debug: prev_node=%s", _extract_node_debug(model, idx=idx - 1))
                if idx + 1 < len(model.graph.node):
                    logger.error("ONNX debug: next_node=%s", _extract_node_debug(model, idx=idx + 1))
                break
            except Exception as e:
                logger.warning("ONNX debug load failed for %s: %s", p, e)
        return

    m2 = re.search(r"Type .* of output arg \(([^)]+)\).*expected type \(([^)]+)\)", s)
    if m2:
        tensor = m2.group(1)
        expected = m2.group(2)
        logger.error("ONNX debug: ORT type mismatch tensor=%s expected=%s", tensor, expected)
        for p in candidate_onnx_paths:
            if not p.exists():
                continue
            try:
                model = onnx.load(str(p))
                outs = [o for o in model.graph.output if o.name == tensor]
                vis = [v for v in model.graph.value_info if v.name == tensor]
                logger.error("ONNX debug: file=%s opset=%s", p, _get_opset(model))
                logger.error("ONNX debug: graph.output matches=%s", [o.name for o in outs])
                logger.error("ONNX debug: graph.value_info matches=%s", [v.name for v in vis])
                break
            except Exception:
                pass
        return


def onnx_debug_dump_for_exception(exc: BaseException, candidate_onnx_paths: Iterable[Path]) -> None:
    onnx_debug_dump_for_exception_paths_only(exc, candidate_onnx_paths)
    _onnx_debug_dump_for_exception_impl(exc, candidate_onnx_paths)
