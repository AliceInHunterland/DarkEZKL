use std::collections::{HashMap, HashSet};

use itertools::Itertools;

use crate::circuit::{
    einsum::reduction_planner::{self, Reduction},
    region::RegionSettings,
    CircuitError,
};
use crate::commands::ExecutionMode;

use super::freivalds::MatMulEquation;

///
#[derive(Debug, Clone)]
pub struct EinsumAnalysis {
    /// max size of input tensors
    pub max_input_size: usize,
    /// max size of output tensors
    pub max_output_size: usize,
    /// max number of input tensors
    pub max_num_inputs: usize,
    /// max number of output axes
    pub max_num_output_axes: usize,
    /// the sum of the lengths of dot product to compute all the reductions
    pub reduction_length: usize,
}

/// The strategy to use for einsum
#[derive(Debug, Clone)]
pub enum EinsumStrategy {
    /// Use only base ops
    BaseOps,
    /// Use Freivalds' argument
    Freivalds,
}

///
#[derive(Debug, Clone)]
pub struct SingleEquationAnalysis {
    ///
    pub equation: String,
    ///
    pub num_inputs: usize,
    ///
    pub max_input_size: usize,
    ///
    pub output_size: usize,
    ///
    pub num_output_axes: usize,
    ///
    pub output_indices: Vec<char>,
    /// the length of dot product to compute all the reductions
    pub reduction_length: usize,
    /// the strategy to use for einsum
    pub strategy: EinsumStrategy,
}

/// Analyze a set of einsum equations for universal configuration, taking into account region settings.
///
/// In probabilistic execution mode, we may prefer `EinsumStrategy::Freivalds` for matmul-like einsums
/// so that later stages can lower these into the probabilistic Freivalds path (instead of the
/// base-ops accumulation path).
pub fn analyze_einsum_usage(
    equations: &HashMap<(usize, String), HashMap<char, usize>>,
    settings: &RegionSettings,
) -> Result<EinsumAnalysis, CircuitError> {
    let mut max_num_inputs = 0;
    let mut max_input_size = 0;
    let mut max_output_size = 0;
    let mut max_num_output_axes = 0;
    let mut reduction_length = 0;

    for ((_, equation), input_axes_to_dim) in equations.iter() {
        let analysis =
            analyze_single_equation_with_settings(equation, input_axes_to_dim, settings)?;
        max_input_size = max_input_size.max(analysis.max_input_size);
        max_output_size = max_output_size.max(analysis.output_size);
        max_num_inputs = max_num_inputs.max(analysis.num_inputs);
        max_num_output_axes = max_num_output_axes.max(analysis.num_output_axes);
        reduction_length += analysis.reduction_length;
    }

    Ok(EinsumAnalysis {
        max_input_size,
        max_output_size,
        max_num_inputs,
        max_num_output_axes,
        reduction_length,
    })
}

/// Backwards-compatible helper when settings are not available.
/// This preserves prior behavior (i.e., no probabilistic strategy overrides).
pub fn analyze_einsum_usage_without_settings(
    equations: &HashMap<(usize, String), HashMap<char, usize>>,
) -> Result<EinsumAnalysis, CircuitError> {
    // Decomposition params don't matter for analysis; choose safe non-zero defaults.
    let settings = RegionSettings::all_false(2, 2);
    analyze_einsum_usage(equations, &settings)
}

///
pub fn analyze_single_equation(
    equation: &str,
    input_axes_to_dim: &HashMap<char, usize>,
) -> Result<SingleEquationAnalysis, CircuitError> {
    analyze_single_equation_inner(equation, input_axes_to_dim, None)
}

/// Analyze a single equation with settings-aware strategy selection.
///
/// If probabilistic mode is enabled and the equation is a candidate matmul (per freivalds.rs parsing
/// and matmul-like shape rules), we select `EinsumStrategy::Freivalds` even if we'd otherwise dispatch
/// to the base-ops einsum path.
pub fn analyze_single_equation_with_settings(
    equation: &str,
    input_axes_to_dim: &HashMap<char, usize>,
    settings: &RegionSettings,
) -> Result<SingleEquationAnalysis, CircuitError> {
    analyze_single_equation_inner(equation, input_axes_to_dim, Some(settings))
}

fn analyze_single_equation_inner(
    equation: &str,
    input_axes_to_dim: &HashMap<char, usize>,
    settings: Option<&RegionSettings>,
) -> Result<SingleEquationAnalysis, CircuitError> {
    // Sanitise equation to remove trivial axes
    let equation = {
        let (inputs_str, output_str) = equation.split_once("->").unwrap();
        let input_equations: Vec<&str> = inputs_str.split(',').collect();

        let inputs: Vec<String> = input_equations
            .iter()
            .map(|input| {
                input
                    .chars()
                    .filter(|char| *input_axes_to_dim.get(char).unwrap() > 1)
                    .collect()
            })
            .collect();

        let output = output_str
            .chars()
            .filter(|c| {
                input_axes_to_dim.get(c).is_some() && *input_axes_to_dim.get(c).unwrap() > 1
            })
            .collect();

        [inputs.join(","), output].join("->")
    };

    let (inputs_eq, output_eq) = equation.split_once("->").unwrap();
    let input_equations: Vec<&str> = inputs_eq.split(',').collect();

    let max_input_size = input_equations
        .iter()
        .map(|eqn| {
            eqn.chars()
                .map(|c| input_axes_to_dim.get(&c).unwrap())
                .product()
        })
        .max()
        .unwrap();

    let output_indices: Vec<char> = output_eq.chars().collect();
    let output_dims = output_indices
        .iter()
        .map(|c| input_axes_to_dim.get(&c).unwrap());
    let output_size = output_dims.clone().product();

    let output_reduction_length = {
        let mut output_dims = output_dims.rev().cloned().collect_vec();
        let mut total_length = 0;
        for _ in 0..output_dims.len() {
            let dot_product_len = output_dims.remove(0);
            let num_dot_products: usize = output_dims.iter().product();
            total_length += dot_product_len * num_dot_products;
        }
        total_length
    };

    let input_reductions_length = {
        let input_reductions = reduction_planner::input_reductions(&equation)?;
        input_reductions
            .into_iter()
            .map(|reduction| {
                let (_, output_expr) = reduction.expression().split_once("->").unwrap();
                let num_inputs = reduction.input_indices().len();
                let dot_product_len = match reduction {
                    Reduction::RLC { axis, .. } => *input_axes_to_dim.get(&axis).unwrap(),
                    Reduction::Contraction { axis, .. } => *axis
                        .and_then(|axis| input_axes_to_dim.get(&axis))
                        .unwrap_or(&1),
                };
                let num_dot_products: usize = output_expr
                    .chars()
                    .map(|c| input_axes_to_dim.get(&c).unwrap())
                    .product();
                // since `multi_dot` does pairwise mult between input pairs and final summation
                if num_inputs <= 2 {
                    num_dot_products * dot_product_len
                } else {
                    num_dot_products * (dot_product_len * num_inputs)
                }
            })
            .sum::<usize>()
    };

    let dispatch_to_einsum_with_base_ops = {
        let mut seen = HashSet::new();
        let mut common_indices_to_inputs = vec![];
        for input in input_equations.iter() {
            for c in input.chars() {
                if !seen.contains(&c) {
                    seen.insert(c);
                } else {
                    common_indices_to_inputs.push(c);
                }
            }
        }
        let non_common_indices = input_axes_to_dim
            .keys()
            .filter(|&x| {
                !common_indices_to_inputs.contains(x)
                    && input_axes_to_dim.get(x).cloned().unwrap() > 1
            })
            .collect::<Vec<_>>();
        !(output_indices.len() > 0
            && common_indices_to_inputs.len() > 0
            && non_common_indices.len() > 1)
    };

    let matmul_candidate_for_freivalds = is_matmul_candidate_for_freivalds(&equation);

    let mut strategy = if dispatch_to_einsum_with_base_ops || !matmul_candidate_for_freivalds {
        EinsumStrategy::BaseOps
    } else {
        EinsumStrategy::Freivalds
    };

    // --- Step 4: settings-aware override for probabilistic matmul candidates ---
    if let Some(settings) = settings {
        if settings.execution_mode == ExecutionMode::Probabilistic
            && settings.prob_k > 0
            && (prob_ops_contains(settings, "MatMul") || prob_ops_contains(settings, "Gemm"))
            && matmul_candidate_for_freivalds
        {
            strategy = EinsumStrategy::Freivalds;
        }
    }

    Ok(SingleEquationAnalysis {
        output_size,
        max_input_size,
        equation: equation.to_string(),
        num_inputs: input_equations.len(),
        num_output_axes: output_indices.len(),
        output_indices,
        reduction_length: output_reduction_length + input_reductions_length,
        strategy,
    })
}

fn prob_ops_contains(settings: &RegionSettings, op_name: &str) -> bool {
    settings
        .prob_ops
        .iter()
        .any(|s| s.eq_ignore_ascii_case(op_name))
}

/// Return true if an einsum equation is a (supported) matmul-like contraction suitable for Freivalds.
///
/// This intentionally mirrors the *structural* checks used by the Freivalds wrapper in `freivalds.rs`,
/// but without needing actual tensors (we only have the equation here).
pub(crate) fn is_matmul_candidate_for_freivalds(equation: &str) -> bool {
    let eq = match MatMulEquation::parse(equation) {
        Some(eq) => eq,
        None => return false,
    };

    let a_axes: Vec<char> = eq.a_indices.chars().collect();
    let b_axes: Vec<char> = eq.b_indices.chars().collect();
    let c_axes: Vec<char> = eq.c_indices.chars().collect();

    // Reject duplicate axes within an operand (diagonal / trace semantics).
    if has_duplicate_chars(&a_axes) || has_duplicate_chars(&b_axes) || has_duplicate_chars(&c_axes)
    {
        return false;
    }

    let a_set: HashSet<char> = a_axes.iter().copied().collect();
    let b_set: HashSet<char> = b_axes.iter().copied().collect();
    let c_set: HashSet<char> = c_axes.iter().copied().collect();

    // Shared axes between A and B
    let shared: HashSet<char> = a_set.intersection(&b_set).copied().collect();

    // contract: shared but not in output
    let contract_set: HashSet<char> = shared.difference(&c_set).copied().collect();
    // batch: shared and in output
    let batch_set: HashSet<char> = shared.intersection(&c_set).copied().collect();

    // left: in output and A but not in B
    let left_set: HashSet<char> = c_set
        .difference(&b_set)
        .copied()
        .filter(|ch| a_set.contains(ch))
        .collect();

    // right: in output and B but not in A
    let right_set: HashSet<char> = c_set
        .difference(&a_set)
        .copied()
        .filter(|ch| b_set.contains(ch))
        .collect();

    // Current Freivalds lowering is restricted to classic matmul / batched matmul style shapes.
    // Conv-like projections such as `NIHW,OI->NOHW` flatten multiple independent "left" axes into
    // the matrix row dimension and have been observed to fail later during proving. We also keep
    // pure batched inner products (no left/right axes) on the exact path.
    if left_set.len() > 1 || right_set.len() > 1 || (left_set.is_empty() && right_set.is_empty()) {
        return false;
    }

    // Validate output indices: must be exactly batch ∪ left ∪ right, and exclude contract.
    let mut expected_out: HashSet<char> = HashSet::new();
    expected_out.extend(batch_set.iter().copied());
    expected_out.extend(left_set.iter().copied());
    expected_out.extend(right_set.iter().copied());

    if expected_out != c_set {
        return false;
    }

    // Validate there are no "single-input reduction indices":
    // any axis that is in A but not in output must also be in B (so it's contracted),
    // and similarly for B.
    for ch in a_set.difference(&c_set) {
        if !b_set.contains(ch) {
            return false;
        }
    }
    for ch in b_set.difference(&c_set) {
        if !a_set.contains(ch) {
            return false;
        }
    }

    // Need at least one contracted axis for Freivalds-style matmul checking.
    !contract_set.is_empty()
}

/// Returns true if `axes` contains any duplicate char.
fn has_duplicate_chars(axes: &[char]) -> bool {
    let mut seen = HashSet::with_capacity(axes.len());
    for ch in axes.iter() {
        if !seen.insert(*ch) {
            return true;
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::is_matmul_candidate_for_freivalds;

    #[test]
    fn accepts_classic_batched_matmul_shapes() {
        assert!(is_matmul_candidate_for_freivalds("ij,jk->ik"));
        assert!(is_matmul_candidate_for_freivalds("bij,bjk->bik"));
    }

    #[test]
    fn rejects_conv_like_projected_matmul_shapes() {
        assert!(!is_matmul_candidate_for_freivalds("NIHW,OI->NOHW"));
    }

    #[test]
    fn rejects_pure_batched_inner_products() {
        assert!(!is_matmul_candidate_for_freivalds("abcde,abcde->abcd"));
    }

    #[test]
    fn conv_like_projected_shapes_use_base_ops_strategy() {
        use std::collections::HashMap;

        let axes_to_dim = HashMap::from([
            ('N', 1usize),
            ('I', 3usize),
            ('H', 8usize),
            ('W', 8usize),
            ('O', 16usize),
        ]);

        let analysis = super::analyze_single_equation("NIHW,OI->NOHW", &axes_to_dim).unwrap();
        assert!(matches!(analysis.strategy, super::EinsumStrategy::BaseOps));
    }
}
