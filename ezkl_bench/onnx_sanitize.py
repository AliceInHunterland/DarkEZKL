from __future__ import annotations

import logging
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

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


def _extract_static_shape(value_info: onnx.ValueInfoProto) -> Optional[List[int]]:
    try:
        tensor_type = value_info.type.tensor_type
        if tensor_type is None or tensor_type.elem_type == 0:
            return None
        dims: List[int] = []
        for dim in tensor_type.shape.dim:
            if not dim.HasField("dim_value"):
                return None
            value = int(dim.dim_value)
            if value <= 0:
                return None
            dims.append(value)
        return dims
    except Exception:
        return None


def _collect_static_shapes(model: onnx.ModelProto) -> Dict[str, List[int]]:
    shapes: Dict[str, List[int]] = {}
    for collection in (model.graph.input, model.graph.value_info, model.graph.output):
        for value_info in collection:
            shape = _extract_static_shape(value_info)
            if shape is not None:
                shapes[value_info.name] = shape
    return shapes


def _collect_identity_passthroughs(model: onnx.ModelProto) -> Dict[str, str]:
    passthroughs: Dict[str, str] = {}
    for node in model.graph.node:
        if node.op_type != "Identity" or len(node.input) != 1 or len(node.output) != 1:
            continue
        src = node.input[0]
        dst = node.output[0]
        if src and dst:
            passthroughs[dst] = src
    return passthroughs


def _resolve_initializer_via_identities(
    name: str,
    *,
    initializer_map: Dict[str, onnx.TensorProto],
    identity_passthroughs: Dict[str, str],
) -> Optional[onnx.TensorProto]:
    current = name
    seen: set[str] = set()
    while current and current not in seen:
        init = initializer_map.get(current)
        if init is not None:
            return init
        seen.add(current)
        current = identity_passthroughs.get(current, "")
    return None


def prune_unreachable_nodes_inplace(model: onnx.ModelProto) -> Dict[str, Any]:
    """
    Drop nodes that do not contribute to graph outputs.

    Some export rewrites intentionally replace a node in-place but leave unrelated
    Identity aliases behind. If those dead nodes survive, the ONNX splitter can
    create internal segments with no live outputs, which `onnx.utils.extract_model`
    rejects.
    """
    g = model.graph
    graph_output_names = {out.name for out in g.output if out.name}
    if not graph_output_names:
        return {"pruned_nodes": 0, "pruned_initializers": 0, "pruned_value_info": 0}

    original_nodes = list(g.node)
    live_tensors: set[str] = set(graph_output_names)
    kept_nodes_rev: List[onnx.NodeProto] = []

    for node in reversed(original_nodes):
        node_outputs = [out for out in node.output if out]
        if not node_outputs:
            continue
        if not any(out in live_tensors for out in node_outputs):
            continue
        kept_nodes_rev.append(node)
        for inp in node.input:
            if inp:
                live_tensors.add(inp)

    kept_nodes = list(reversed(kept_nodes_rev))
    removed_nodes = len(original_nodes) - len(kept_nodes)

    used_tensor_names = set(graph_output_names)
    for node in kept_nodes:
        for inp in node.input:
            if inp:
                used_tensor_names.add(inp)
        for out in node.output:
            if out:
                used_tensor_names.add(out)

    kept_initializers = [init for init in g.initializer if init.name in used_tensor_names]
    removed_initializers = len(g.initializer) - len(kept_initializers)

    live_value_info_names = used_tensor_names | {value.name for value in g.input if value.name}
    kept_value_info = [value for value in g.value_info if value.name in live_value_info_names]
    removed_value_info = len(g.value_info) - len(kept_value_info)

    if removed_nodes:
        logger.info("Pruned unreachable ONNX nodes: removed=%s kept=%s", removed_nodes, len(kept_nodes))
        del g.node[:]
        g.node.extend(kept_nodes)

    if removed_initializers:
        del g.initializer[:]
        g.initializer.extend(kept_initializers)

    if removed_value_info:
        del g.value_info[:]
        g.value_info.extend(kept_value_info)

    return {
        "pruned_nodes": removed_nodes,
        "pruned_initializers": removed_initializers,
        "pruned_value_info": removed_value_info,
    }


def rewrite_large_conv_to_channel_chunks_inplace(
    model: onnx.ModelProto,
    *,
    max_macs_per_chunk: int,
) -> Dict[str, Any]:
    """
    Rewrite oversized Conv nodes into smaller Conv chunks split across output channels,
    followed by a Concat on axis=1.

    This gives the ONNX splitter something real to cut when a single large 3x3 Conv
    would otherwise become an unsplittable one-node segment.
    """
    if int(max_macs_per_chunk) <= 0:
        return {"conv_chunk_rewrites": 0, "conv_chunk_max_macs": int(max_macs_per_chunk)}

    g = model.graph
    static_shapes = _collect_static_shapes(model)
    initializer_map = {init.name: init for init in g.initializer}

    new_nodes: List[onnx.NodeProto] = []
    new_initializers: List[onnx.TensorProto] = []
    replaced_initializer_names: set[str] = set()

    rewrites = 0
    chunk_nodes = 0
    skipped_grouped = 0
    skipped_shape = 0

    for node in g.node:
        if node.op_type != "Conv":
            new_nodes.append(node)
            continue
        if len(node.input) < 2 or len(node.output) < 1:
            new_nodes.append(node)
            continue

        weight_name = node.input[1]
        weight_init = initializer_map.get(weight_name)
        if weight_init is None:
            new_nodes.append(node)
            continue

        try:
            weight_arr = numpy_helper.to_array(weight_init)
        except Exception:
            new_nodes.append(node)
            continue

        if weight_arr.ndim < 3:
            new_nodes.append(node)
            continue

        kernel_area = int(math.prod(int(x) for x in weight_arr.shape[2:]))
        if kernel_area <= 1:
            new_nodes.append(node)
            continue

        group = _get_attr_int(node, "group", 1)
        if group != 1:
            skipped_grouped += 1
            new_nodes.append(node)
            continue

        output_name = node.output[0]
        output_shape = static_shapes.get(output_name)
        if not output_shape:
            skipped_shape += 1
            new_nodes.append(node)
            continue

        out_channels = int(weight_arr.shape[0])
        if out_channels <= 1:
            new_nodes.append(node)
            continue

        output_elems = int(math.prod(output_shape))
        macs_total = int(output_elems * int(weight_arr.shape[1]) * kernel_area)
        if macs_total <= int(max_macs_per_chunk):
            new_nodes.append(node)
            continue

        chunk_count = max(2, int(math.ceil(float(macs_total) / float(max_macs_per_chunk))))
        chunk_out_channels = max(1, int(math.ceil(float(out_channels) / float(chunk_count))))
        if chunk_out_channels >= out_channels:
            new_nodes.append(node)
            continue

        bias_name = node.input[2] if len(node.input) >= 3 and node.input[2] else ""
        bias_arr = None
        if bias_name:
            bias_init = initializer_map.get(bias_name)
            if bias_init is None:
                new_nodes.append(node)
                continue
            try:
                bias_arr = numpy_helper.to_array(bias_init)
            except Exception:
                new_nodes.append(node)
                continue
            if getattr(bias_arr, "shape", None) != (out_channels,):
                new_nodes.append(node)
                continue

        attrs = {attr.name: helper.get_attribute_value(attr) for attr in node.attribute}
        base = node.name or output_name or f"conv_{rewrites}"
        chunk_outputs: List[str] = []

        for chunk_idx, start in enumerate(range(0, out_channels, chunk_out_channels)):
            end = min(start + chunk_out_channels, out_channels)
            chunk_weight_name = f"{base}__weight_{chunk_idx:02d}"
            chunk_weight = numpy_helper.from_array(weight_arr[start:end].copy(), name=chunk_weight_name)
            new_initializers.append(chunk_weight)

            chunk_inputs = [node.input[0], chunk_weight_name]
            if bias_arr is not None:
                chunk_bias_name = f"{base}__bias_{chunk_idx:02d}"
                chunk_bias = numpy_helper.from_array(bias_arr[start:end].copy(), name=chunk_bias_name)
                new_initializers.append(chunk_bias)
                chunk_inputs.append(chunk_bias_name)

            chunk_output = f"{base}__out_{chunk_idx:02d}"
            chunk_outputs.append(chunk_output)
            new_nodes.append(
                helper.make_node(
                    "Conv",
                    inputs=chunk_inputs,
                    outputs=[chunk_output],
                    name=f"{base}__chunk_{chunk_idx:02d}",
                    **attrs,
                )
            )

        new_nodes.append(
            helper.make_node(
                "Concat",
                inputs=chunk_outputs,
                outputs=[output_name],
                name=f"{base}__concat",
                axis=1,
            )
        )

        replaced_initializer_names.add(weight_name)
        if bias_name:
            replaced_initializer_names.add(bias_name)

        rewrites += 1
        chunk_nodes += len(chunk_outputs)
        logger.info(
            "Rewrote Conv node into %s channel chunks: name=%s output=%s macs=%s max_macs_per_chunk=%s",
            len(chunk_outputs),
            node.name or "<unnamed>",
            output_name,
            macs_total,
            max_macs_per_chunk,
        )

    if rewrites:
        kept_initializers = [init for init in g.initializer if init.name not in replaced_initializer_names]
        del g.node[:]
        g.node.extend(new_nodes)
        del g.initializer[:]
        g.initializer.extend(kept_initializers)
        g.initializer.extend(new_initializers)

    return {
        "conv_chunk_rewrites": rewrites,
        "conv_chunk_nodes": chunk_nodes,
        "conv_chunk_max_macs": int(max_macs_per_chunk),
        "conv_chunk_skipped_grouped": skipped_grouped,
        "conv_chunk_skipped_shape": skipped_shape,
    }


def rewrite_inference_batchnorm_to_affine_inplace(model: onnx.ModelProto) -> Dict[str, Any]:
    """
    Replace inference BatchNormalization nodes with equivalent affine Mul + Add ops.

    RepVGG exports many inference-only identity BatchNorm branches whose running stats are
    sometimes threaded through Identity aliases. Folding them here prevents tract/ezkl from
    seeing standalone BatchNormalization segments after ONNX splitting.
    """
    g = model.graph
    static_shapes = _collect_static_shapes(model)
    initializer_map = {init.name: init for init in g.initializer}
    identity_passthroughs = _collect_identity_passthroughs(model)

    new_nodes: List[onnx.NodeProto] = []
    new_initializers: List[onnx.TensorProto] = []

    rewrites = 0
    skipped_missing_constants = 0
    skipped_shape = 0
    skipped_unsupported = 0

    for node in g.node:
        if node.op_type != "BatchNormalization":
            new_nodes.append(node)
            continue

        if len(node.input) < 5 or len(node.output) != 1:
            skipped_unsupported += 1
            new_nodes.append(node)
            continue

        training_mode = _get_attr_int(node, "training_mode", 0)
        if training_mode != 0 or len(node.output) != 1:
            skipped_unsupported += 1
            new_nodes.append(node)
            continue

        x_name = node.input[0]
        y_name = node.output[0]
        input_shape = static_shapes.get(x_name) or static_shapes.get(y_name)
        if not input_shape:
            skipped_shape += 1
            new_nodes.append(node)
            continue

        scale_init = _resolve_initializer_via_identities(
            node.input[1], initializer_map=initializer_map, identity_passthroughs=identity_passthroughs
        )
        bias_init = _resolve_initializer_via_identities(
            node.input[2], initializer_map=initializer_map, identity_passthroughs=identity_passthroughs
        )
        mean_init = _resolve_initializer_via_identities(
            node.input[3], initializer_map=initializer_map, identity_passthroughs=identity_passthroughs
        )
        var_init = _resolve_initializer_via_identities(
            node.input[4], initializer_map=initializer_map, identity_passthroughs=identity_passthroughs
        )
        if any(init is None for init in (scale_init, bias_init, mean_init, var_init)):
            skipped_missing_constants += 1
            new_nodes.append(node)
            continue

        try:
            scale_arr = np.asarray(numpy_helper.to_array(scale_init))
            bias_arr = np.asarray(numpy_helper.to_array(bias_init))
            mean_arr = np.asarray(numpy_helper.to_array(mean_init))
            var_arr = np.asarray(numpy_helper.to_array(var_init))
        except Exception:
            skipped_missing_constants += 1
            new_nodes.append(node)
            continue

        if any(arr.ndim != 1 for arr in (scale_arr, bias_arr, mean_arr, var_arr)):
            skipped_shape += 1
            new_nodes.append(node)
            continue

        channel_count = int(scale_arr.shape[0])
        if channel_count <= 0 or any(int(arr.shape[0]) != channel_count for arr in (bias_arr, mean_arr, var_arr)):
            skipped_shape += 1
            new_nodes.append(node)
            continue

        if len(input_shape) == 1:
            if int(input_shape[0]) != channel_count:
                skipped_shape += 1
                new_nodes.append(node)
                continue
            broadcast_shape = [channel_count]
        else:
            if len(input_shape) < 2 or int(input_shape[1]) != channel_count:
                skipped_shape += 1
                new_nodes.append(node)
                continue
            broadcast_shape = [1, channel_count] + [1] * (len(input_shape) - 2)

        epsilon = _get_attr_float(node, "epsilon", 1e-5)
        calc_dtype = np.float32
        out_dtype = np.dtype(scale_arr.dtype)

        scale_f = scale_arr.astype(calc_dtype, copy=False)
        bias_f = bias_arr.astype(calc_dtype, copy=False)
        mean_f = mean_arr.astype(calc_dtype, copy=False)
        var_f = var_arr.astype(calc_dtype, copy=False)

        affine_scale = scale_f / np.sqrt(var_f + float(epsilon))
        affine_bias = bias_f - (mean_f * affine_scale)

        affine_scale = affine_scale.astype(out_dtype, copy=False).reshape(broadcast_shape)
        affine_bias = affine_bias.astype(out_dtype, copy=False).reshape(broadcast_shape)

        base = node.name or y_name or f"batchnorm_{rewrites}"
        scale_name = f"{base}__affine_scale"
        bias_name = f"{base}__affine_bias"
        mul_out = f"{base}__mul_out"

        new_initializers.append(numpy_helper.from_array(affine_scale.copy(), name=scale_name))
        new_initializers.append(numpy_helper.from_array(affine_bias.copy(), name=bias_name))
        new_nodes.append(helper.make_node("Mul", inputs=[x_name, scale_name], outputs=[mul_out], name=f"{base}__Mul"))
        new_nodes.append(helper.make_node("Add", inputs=[mul_out, bias_name], outputs=[y_name], name=f"{base}__Add"))

        rewrites += 1

    if rewrites:
        logger.info("Rewrote inference BatchNormalization nodes: count=%s", rewrites)
        del g.node[:]
        g.node.extend(new_nodes)
        if new_initializers:
            g.initializer.extend(new_initializers)

    return {
        "batchnorm_rewrites": rewrites,
        "batchnorm_skipped_missing_constants": skipped_missing_constants,
        "batchnorm_skipped_shape": skipped_shape,
        "batchnorm_skipped_unsupported": skipped_unsupported,
    }


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


def sanitize_exported_onnx_inplace(
    path: Path,
    *,
    rewrite_gemm: bool,
    rewrite_conv_max_macs: Optional[int] = None,
    rewrite_batchnorm: bool = False,
) -> Dict[str, Any]:
    """
    Post-process exported/conversion ONNX to improve stability:
      - shape/type inference (best-effort)
      - optional large-Conv chunking (split-mode friendliness)
      - optional inference BatchNorm folding (tract stability on split identity branches)
      - optional Gemm rewrite (tract stability)
    """
    if not path.exists():
        return {"ok": False, "error": f"missing: {path}"}

    model = onnx.load(str(path))
    meta: Dict[str, Any] = {"path": str(path), "opset": _get_opset(model)}

    # Inference before rewrite helps some graphs, and also helps us debug.
    model = _safe_infer_shapes(model)

    if rewrite_conv_max_macs is not None:
        meta.update(rewrite_large_conv_to_channel_chunks_inplace(model, max_macs_per_chunk=int(rewrite_conv_max_macs)))
        model = _safe_infer_shapes(model)

    if rewrite_batchnorm:
        meta.update(rewrite_inference_batchnorm_to_affine_inplace(model))
        model = _safe_infer_shapes(model)

    if rewrite_gemm:
        meta.update(rewrite_gemm_to_matmul_add_inplace(model))
        model = _safe_infer_shapes(model)

    meta.update(prune_unreachable_nodes_inplace(model))
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
