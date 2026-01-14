import logging
import onnx
from onnx import numpy_helper
import os
from pathlib import Path
from typing import List, Tuple, Dict, Any

logger = logging.getLogger(__name__)

class OnnxSplitter:
    """
    Splits an ONNX model into sequential segments based on parameter count threshold.
    """

    def __init__(self, onnx_path: Path, threshold_params: int = 50000):
        self.onnx_path = onnx_path
        self.threshold = threshold_params
        self.model = onnx.load(str(onnx_path))
        self.graph = self.model.graph
        
        # Map initializer names to their size (num elements)
        self.initializer_sizes = {}
        for init in self.graph.initializer:
            size = 1
            for dim in init.dims:
                size *= dim
            self.initializer_sizes[init.name] = size

    def _get_node_params(self, node) -> int:
        """Calculate number of parameters (initializers) used by this node."""
        count = 0
        for input_name in node.input:
            if input_name in self.initializer_sizes:
                count += self.initializer_sizes[input_name]
        return count

    def split(self, output_dir: Path) -> List[Path]:
        """
        Splits the model and saves segments to output_dir.
        Returns a list of paths to the split ONNX files.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Traverse and find cut points
        cut_points = [] # List of output names that form boundaries
        
        current_params = 0
        node_indices = []
        
        # We will split strictly by node index ranges for simplicity and robustness
        # using onnx.utils.extract_model which takes input/output names.
        # However, extract_model is topological. We need to define input/output tensors.
        
        # Strategy:
        # Walk nodes. Accumulate params.
        # If > threshold, the current node's output becomes a cut point.
        # The next segment starts with that output as input.
        
        segments = []
        
        # Top-level inputs
        model_inputs = [i.name for i in self.graph.input]
        
        # We need an ordered list of nodes. The graph.node is usually topological.
        # We'll identify the tensor names that flow between segments.
        
        current_start_inputs = [i.name for i in self.graph.input if i.name not in self.initializer_sizes]
        
        # Sanity: if model has no graph inputs (rare), handle it
        if not current_start_inputs:
            logger.warning("Model has no graph inputs?")
        
        current_segment_nodes = []
        segment_idx = 0
        
        # Build a map of "tensor produced by node"
        # and "tensors consumed by node" is node.input
        
        total_nodes = len(self.graph.node)
        logger.info(f"Splitting model with {total_nodes} nodes, threshold={self.threshold} params")

        processed_nodes = 0
        
        # Determine boundaries
        boundaries = [] # (start_node_idx, end_node_idx, output_tensor_name)
        
        start_idx = 0
        current_count = 0
        
        # A mapped structure to easily find what creates what
        node_outputs = {}
        for idx, node in enumerate(self.graph.node):
            for out in node.output:
                node_outputs[out] = idx
        
        for idx, node in enumerate(self.graph.node):
            p = self._get_node_params(node)
            current_count += p
            
            # Check threshold (ensure at least 1 node per segment)
            if current_count >= self.threshold and idx < total_nodes - 1:
                # Found a cut point
                # The output of this node will be the output of this segment
                # and input of the next.
                # We pick the first output of the node as the cut tensor.
                # (Complex nodes might have multiple, we take the first for simplicity)
                cut_tensor = node.output[0]
                
                boundaries.append({
                    "start": start_idx,
                    "end": idx, # inclusive
                    "output": cut_tensor,
                    "params": current_count
                })
                
                logger.info(f"Segment {len(boundaries)-1}: nodes {start_idx}-{idx}, params={current_count}, cut={cut_tensor}")
                
                start_idx = idx + 1
                current_count = 0
        
        # Add final segment
        if start_idx < total_nodes:
            # Final output is the graph output
            final_outputs = [o.name for o in self.graph.output]
            boundaries.append({
                "start": start_idx,
                "end": total_nodes - 1,
                "output": final_outputs, # List for final
                "params": current_count
            })
            logger.info(f"Segment {len(boundaries)-1} (Final): nodes {start_idx}-{total_nodes-1}, params={current_count}")

        if len(boundaries) <= 1:
            logger.info("Model was not split (threshold not met or single segment).")
            # Just return original if we didn't split, or save a copy?
            # Returns list containing just the original if no split
            out_file = output_dir / "network_0.onnx"
            onnx.save(self.model, str(out_file))
            return [out_file]

        # 2. Extract segments
        # We use onnx.utils.extract_model. 
        # Note: extract_model takes input_names and output_names. It includes all nodes 
        # required to compute outputs from inputs. This might duplicate nodes if we aren't careful.
        # But if we define inputs as the cut tensors, it should be fine.
        
        generated_files = []
        
        prev_output_names = current_start_inputs
        
        for i, b in enumerate(boundaries):
            seg_path = output_dir / f"network_{i}.onnx"
            
            target_outputs = b["output"]
            if isinstance(target_outputs, str):
                target_outputs = [target_outputs]
                
            # Input names for this segment are the outputs of the previous (or graph inputs for first)
            input_names = prev_output_names
            
            logger.info(f"Extracting Segment {i}: Inputs={input_names} -> Outputs={target_outputs}")
            
            try:
                # onnx.utils.extract_model is available in standard install
                onnx.utils.extract_model(
                    str(self.onnx_path),
                    str(seg_path),
                    input_names=input_names,
                    output_names=target_outputs,
                    check_model=False # Disable check to speed up and avoid shape inference errors on partials
                )
                generated_files.append(seg_path)
            except Exception as e:
                logger.error(f"Failed to extract segment {i}: {e}")
                raise e
            
            # Prepare for next
            prev_output_names = target_outputs

        return generated_files

def split_onnx(onnx_path: Path, threshold: int) -> List[Path]:
    splitter = OnnxSplitter(onnx_path, threshold)
    return splitter.split(onnx_path.parent)

