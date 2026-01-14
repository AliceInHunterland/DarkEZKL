from __future__ import annotations

import inspect
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import onnx
from onnx import numpy_helper

logger = logging.getLogger(__name__)

# Manifest bump to invalidate caches when we change split rules.
SPLITTER_VERSION = 2


@dataclass(frozen=True)
class SplitSegment:
    idx: int
    onnx_path: str
    input_names: List[str]
    output_names: List[str]
    node_count: int
    param_count: int  # number of scalar parameters (elements) in initializers attributed to nodes in this segment


def _initializer_param_counts(model: onnx.ModelProto) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for init in model.graph.initializer:
        try:
            arr = numpy_helper.to_array(init)
            counts[init.name] = int(arr.size)
        except Exception:
            size = 1
            for d in init.dims:
                size *= int(d)
            counts[init.name] = int(size)
    return counts


def _topo_nodes(model: onnx.ModelProto) -> List[onnx.NodeProto]:
    return list(model.graph.node)


def _collect_tensor_last_use(
    nodes: Sequence[onnx.NodeProto],
    initializer_names: Set[str],
) -> Dict[str, int]:
    last_use: Dict[str, int] = {}
    for i, n in enumerate(nodes):
        for inp in n.input:
            if not inp or inp in initializer_names:
                continue
            last_use[inp] = i
    return last_use


def _collect_tensor_producer_info(
    nodes: Sequence[onnx.NodeProto],
) -> Tuple[Dict[str, int], Dict[str, str]]:
    producer_idx: Dict[str, int] = {}
    producer_op: Dict[str, str] = {}
    for i, n in enumerate(nodes):
        for out in n.output:
            if out:
                producer_idx[out] = i
                producer_op[out] = n.op_type
    return producer_idx, producer_op


def _node_param_count(
    node: onnx.NodeProto,
    initializer_param_count: Dict[str, int],
) -> int:
    c = 0
    for inp in node.input:
        if inp in initializer_param_count:
            c += initializer_param_count[inp]
    return int(c)


def _segment_inputs_for_nodes(
    nodes: Sequence[onnx.NodeProto],
    *,
    seg_start: int,
    producer_idx: Dict[str, int],
    producer_op: Dict[str, str],
    initializer_names: Set[str],
) -> List[str]:
    """
    Inputs are any non-initializer tensors referenced by nodes in this segment that are not
    produced by a node in this segment.

    Optimization: do NOT surface tensors produced by Constant nodes as segment inputs.
    They can be safely re-materialized by extract_model into downstream segments, reducing
    segment IO fan-out dramatically (fixes the "hundreds of Constant_* outputs" issue).
    """
    produced: Set[str] = set()
    for n in nodes:
        for out in n.output:
            if out:
                produced.add(out)

    needed: List[str] = []
    seen: Set[str] = set()
    for n in nodes:
        for inp in n.input:
            if not inp or inp in initializer_names:
                continue
            if inp in produced:
                continue

            # If this tensor is produced by a Constant node earlier in the graph,
            # do not treat it as an external input; let extract_model pull it in.
            pidx = producer_idx.get(inp)
            if pidx is not None and pidx < seg_start and producer_op.get(inp) == "Constant":
                continue

            if inp not in seen:
                needed.append(inp)
                seen.add(inp)
    return needed


def _segment_outputs_for_cut(
    nodes: Sequence[onnx.NodeProto],
    *,
    seg_start: int,
    seg_end: int,
    last_use: Dict[str, int],
    graph_output_names: Set[str],
    producer_op: Dict[str, str],
) -> List[str]:
    """
    Outputs must include:
      - any tensor produced in this segment that is used after seg_end (live-out),
      - any graph outputs produced in this segment.

    Optimization: do NOT include Constant-node outputs as live-outs.
    They can be duplicated into downstream segments without passing huge IO maps.
    """
    produced_in_seg: Set[str] = set()
    for n in nodes[seg_start : seg_end + 1]:
        for out in n.output:
            if out:
                produced_in_seg.add(out)

    outs: List[str] = []
    seen: Set[str] = set()

    for t in produced_in_seg:
        if producer_op.get(t) == "Constant" and t not in graph_output_names:
            continue
        if t in graph_output_names or last_use.get(t, -1) > seg_end:
            if t not in seen:
                outs.append(t)
                seen.add(t)

    ordered_outs: List[str] = []
    seen2: Set[str] = set()
    for n in nodes[seg_start : seg_end + 1]:
        for out in n.output:
            if out and out in seen and out not in seen2:
                ordered_outs.append(out)
                seen2.add(out)

    for t in outs:
        if t not in seen2:
            ordered_outs.append(t)
            seen2.add(t)

    return ordered_outs


def _extract_model_compat(
    src_onnx: Path,
    dst_onnx: Path,
    *,
    input_names: List[str],
    output_names: List[str],
) -> None:
    kwargs = {"input_names": input_names, "output_names": output_names}

    try:
        sig = inspect.signature(onnx.utils.extract_model)  # type: ignore[attr-defined]
        if "check_model" in sig.parameters:
            kwargs["check_model"] = False
    except Exception:
        pass

    onnx.utils.extract_model(  # type: ignore[attr-defined]
        str(src_onnx),
        str(dst_onnx),
        **kwargs,
    )

    # Best-effort: infer shapes/types for stability (helps ORT + other tooling).
    try:
        m = onnx.load(str(dst_onnx))
        m = onnx.shape_inference.infer_shapes(m)
        onnx.save(m, str(dst_onnx))
    except Exception:
        pass


def split_onnx_model(
    onnx_path: Path,
    out_dir: Path,
    *,
    min_params_per_segment: int = 50_000,
    force: bool = False,
) -> List[SplitSegment]:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "split_manifest.json"

    src_stat = None
    try:
        st = onnx_path.stat()
        src_stat = {"size": int(st.st_size), "mtime": float(st.st_mtime)}
    except Exception:
        src_stat = None

    if manifest_path.exists() and not force:
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            if int(payload.get("splitter_version", 0)) != SPLITTER_VERSION:
                raise RuntimeError("split manifest version mismatch")

            prev = payload.get("source_onnx_stat") or {}
            if src_stat and (prev.get("size") != src_stat["size"] or prev.get("mtime") != src_stat["mtime"]):
                raise RuntimeError("split manifest source stat mismatch")

            segs = []
            for s in payload.get("segments", []):
                segs.append(
                    SplitSegment(
                        idx=int(s["idx"]),
                        onnx_path=str(s["onnx_path"]),
                        input_names=list(s["input_names"]),
                        output_names=list(s["output_names"]),
                        node_count=int(s["node_count"]),
                        param_count=int(s["param_count"]),
                    )
                )
            if segs:
                logger.info("Using cached ONNX split manifest: %s (%s segments)", manifest_path, len(segs))
                return segs
        except Exception as e:
            logger.info("Cached split manifest ignored; will re-split (%s): %s", manifest_path, e)

    logger.info("Splitting ONNX model: %s -> %s (min_params_per_segment=%s)", onnx_path, out_dir, min_params_per_segment)

    model = onnx.load(str(onnx_path))

    # Clear value_info to remove stale type/shape information.
    if model.graph.value_info:
        logger.info("Clearing stale ValueInfo from model before splitting.")
        model.graph.value_info.clear()

    # Save the "cleaned" model to disk; extract_model reads from disk.
    clean_onnx_path = onnx_path.with_suffix(".clean.tmp.onnx")
    try:
        # Best-effort inference after clearing (re-populates with consistent types where possible).
        try:
            model = onnx.shape_inference.infer_shapes(model)
        except Exception:
            pass
        onnx.save(model, str(clean_onnx_path))

        nodes = _topo_nodes(model)
        initializer_names = {i.name for i in model.graph.initializer}
        init_param_count = _initializer_param_counts(model)
        last_use = _collect_tensor_last_use(nodes, initializer_names)
        producer_idx, producer_op = _collect_tensor_producer_info(nodes)
        graph_output_names = {o.name for o in model.graph.output}

        node_params = [_node_param_count(n, init_param_count) for n in nodes]
        total_params = int(sum(node_params))
        logger.info("ONNX nodes=%s total_params=%.3fM", len(nodes), total_params / 1_000_000.0)

        segments: List[Tuple[int, int]] = []
        seg_start = 0
        acc = 0
        for i, pc in enumerate(node_params):
            acc += int(pc)
            if acc >= min_params_per_segment and i < len(nodes) - 1:
                segments.append((seg_start, i))
                seg_start = i + 1
                acc = 0
        segments.append((seg_start, len(nodes) - 1))

        out: List[SplitSegment] = []

        for idx, (a, b) in enumerate(segments):
            seg_nodes = nodes[a : b + 1]
            seg_param_count = int(sum(node_params[a : b + 1]))

            input_names = _segment_inputs_for_nodes(
                seg_nodes,
                seg_start=a,
                producer_idx=producer_idx,
                producer_op=producer_op,
                initializer_names=initializer_names,
            )
            output_names = _segment_outputs_for_cut(
                nodes=nodes,
                seg_start=a,
                seg_end=b,
                last_use=last_use,
                graph_output_names=graph_output_names,
                producer_op=producer_op,
            )

            if idx == 0 and not input_names:
                input_names = [i.name for i in model.graph.input]

            if idx == len(segments) - 1:
                output_names = [o.name for o in model.graph.output]

            seg_path = out_dir / f"seg_{idx:03d}.onnx"
            _extract_model_compat(
                clean_onnx_path,
                seg_path,
                input_names=input_names,
                output_names=output_names,
            )

            out.append(
                SplitSegment(
                    idx=idx,
                    onnx_path=str(seg_path),
                    input_names=input_names,
                    output_names=output_names,
                    node_count=len(seg_nodes),
                    param_count=seg_param_count,
                )
            )

            logger.info(
                "Segment %s: nodes=%s params=%.3fM inputs=%s outputs=%s path=%s",
                idx,
                len(seg_nodes),
                seg_param_count / 1_000_000.0,
                len(input_names),
                len(output_names),
                seg_path,
            )

        payload = {
            "splitter_version": SPLITTER_VERSION,
            "source_onnx": str(onnx_path),
            "source_onnx_stat": src_stat,
            "min_params_per_segment": int(min_params_per_segment),
            "notes": {"constant_liveouts_suppressed": True},
            "segments": [asdict(s) for s in out],
        }
        manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logger.info("Wrote split manifest: %s", manifest_path)

        return out

    finally:
        if clean_onnx_path.exists():
            try:
                clean_onnx_path.unlink()
            except Exception:
                pass


def run_onnx_segment(
    onnx_path: Path,
    inputs: Dict[str, np.ndarray],
    *,
    output_names: Optional[List[str]] = None,
) -> Dict[str, np.ndarray]:
    """
    Execute an ONNX model segment to materialize intermediate activations needed as
    inputs for the next segment.
    """
    import onnxruntime as ort

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    # Keep ORT logs reasonable by default; only go very verbose when our logger is DEBUG.
    sess_options.log_severity_level = 0 if logger.isEnabledFor(logging.DEBUG) else 2

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    try:
        sess = ort.InferenceSession(str(onnx_path), sess_options=sess_options, providers=providers)
    except Exception as e:
        logger.warning("Failed to create ORT session with CUDA provider: %s. Fallback to CPU.", e)
        sess = ort.InferenceSession(str(onnx_path), sess_options=sess_options, providers=["CPUExecutionProvider"])

    logger.info("ONNXRuntime session providers (actual): %s", sess.get_providers())

    if output_names is None:
        output_names = [o.name for o in sess.get_outputs()]

    model_inputs = sess.get_inputs()
    ort_inputs: Dict[str, np.ndarray] = {}

    for inp in model_inputs:
        name = inp.name
        if name not in inputs:
            continue

        val = inputs[name]
        if not isinstance(val, np.ndarray):
            val = np.asarray(val)

        target_type = inp.type or ""

        # Fix: promotion path must cast float16 -> float32 (previously incorrectly re-cast to float16).
        if "float16" in target_type and val.dtype == np.float32:
            val = val.astype(np.float16)
        elif "tensor(float)" in target_type and val.dtype == np.float16:
            val = val.astype(np.float32)
        elif "tensor(float)" in target_type and val.dtype == np.float64:
            val = val.astype(np.float32)

        ort_inputs[name] = val

    outs = sess.run(output_names, ort_inputs)

    out_map: Dict[str, np.ndarray] = {}
    for name, arr in zip(output_names, outs):
        out_map[name] = np.asarray(arr)
    return out_map
