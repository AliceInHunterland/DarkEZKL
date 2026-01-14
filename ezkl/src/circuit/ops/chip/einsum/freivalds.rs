use std::collections::{HashMap, HashSet};

use crate::circuit::ops::chip::einsum::Einsums;
use crate::circuit::ops::probabilistic as probabilistic_ops;
use crate::circuit::{region::RegionCtx, CheckMode, CircuitError};
use crate::tensor::{Tensor, TensorType, ValTensor, ValType};
use halo2_proofs::circuit::Value;
use halo2curves::ff::PrimeField;

#[derive(Clone, Debug)]
pub struct MatMulEquation {
    pub a_indices: String,
    pub b_indices: String,
    pub c_indices: String,
}

impl MatMulEquation {
    /// Parses an einsum equation string (e.g. "ij,jk->ik") into a MatMulEquation.
    /// Returns None if the equation does not represent a supported 2-input contraction form.
    ///
    /// Note: this parser is intentionally permissive w.r.t. rank; validation that the
    /// equation corresponds to a matmul-like contraction happens later (when we have
    /// the actual tensor shapes).
    pub fn parse(equation: &str) -> Option<Self> {
        let (lhs, rhs) = equation.split_once("->")?;
        let inputs: Vec<&str> = lhs.split(',').collect();
        if inputs.len() != 2 {
            return None;
        }

        let a = inputs[0].trim().to_string();
        let b = inputs[1].trim().to_string();
        let c = rhs.trim().to_string();

        if a.is_empty() || b.is_empty() || c.is_empty() {
            return None;
        }

        Some(MatMulEquation {
            a_indices: a,
            b_indices: b,
            c_indices: c,
        })
    }
}

/// Parsed matmul-like structure for a 2-input einsum equation.
///
/// We support the common contraction shape:
///   A(batch, left, contract) * B(batch, contract, right) = C(batch, left, right)
///
/// Where:
/// - `batch` indices appear in both inputs and the output,
/// - `contract` indices appear in both inputs but not the output,
/// - `left` indices appear in (A ∩ output) but not B,
/// - `right` indices appear in (B ∩ output) but not A.
///
/// We intentionally do *not* support "single-input reduction indices" (indices that
/// appear in only one input and not the output), since that is not a matmul-like
/// contraction and cannot be checked by Freivalds in this wrapper.
#[derive(Clone, Debug)]
struct EinsumMatMulSpec {
    // Original axes order
    a_axes: Vec<char>,
    b_axes: Vec<char>,
    c_axes: Vec<char>,

    // Grouped indices (canonical order)
    batch: Vec<char>,
    left: Vec<char>,
    contract: Vec<char>,
    right: Vec<char>,

    // Desired tensor axis orders
    a_order: Vec<char>, // batch + left + contract
    b_order: Vec<char>, // batch + contract + right
    c_order: Vec<char>, // batch + left + right

    // Axis dimensions
    axis_dims: HashMap<char, usize>,

    // Flattened sizes
    batch_dims: Vec<usize>,
    m: usize,
    n: usize,
    p: usize,
}

impl EinsumMatMulSpec {
    fn from_equation_and_tensors<F: PrimeField + TensorType + PartialOrd + std::hash::Hash>(
        eq: &MatMulEquation,
        a: &ValTensor<F>,
        b: &ValTensor<F>,
        c: &ValTensor<F>,
    ) -> Result<Self, CircuitError> {
        let a_axes: Vec<char> = eq.a_indices.chars().collect();
        let b_axes: Vec<char> = eq.b_indices.chars().collect();
        let c_axes: Vec<char> = eq.c_indices.chars().collect();

        // Reject duplicate axes within an operand (diagonal / trace semantics).
        // (We can add support later, but it complicates the matmul reduction.)
        if has_duplicate_chars(&a_axes)
            || has_duplicate_chars(&b_axes)
            || has_duplicate_chars(&c_axes)
        {
            return Err(CircuitError::UnsupportedOp);
        }

        // Ensure ranks match axes lengths.
        if a.dims().len() != a_axes.len()
            || b.dims().len() != b_axes.len()
            || c.dims().len() != c_axes.len()
        {
            return Err(CircuitError::UnsupportedOp);
        }

        // Map axis -> dim, and ensure shared axes have equal dimensions across operands.
        let mut axis_dims: HashMap<char, usize> = HashMap::new();

        for (i, ch) in a_axes.iter().enumerate() {
            let dim = a.dims()[i];
            match axis_dims.get(ch) {
                Some(&existing) if existing != dim => return Err(CircuitError::UnsupportedOp),
                _ => {
                    axis_dims.insert(*ch, dim);
                }
            }
        }
        for (i, ch) in b_axes.iter().enumerate() {
            let dim = b.dims()[i];
            match axis_dims.get(ch) {
                Some(&existing) if existing != dim => return Err(CircuitError::UnsupportedOp),
                _ => {
                    axis_dims.insert(*ch, dim);
                }
            }
        }
        for (i, ch) in c_axes.iter().enumerate() {
            let dim = c.dims()[i];
            match axis_dims.get(ch) {
                Some(&existing) if existing != dim => return Err(CircuitError::UnsupportedOp),
                _ => {
                    axis_dims.insert(*ch, dim);
                }
            }
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

        // left: in A and output but not in B (equivalently: in output and not in B, but must be in A)
        let left_set: HashSet<char> = c_set
            .difference(&b_set)
            .copied()
            .filter(|ch| a_set.contains(ch))
            .collect();

        // right: in B and output but not in A
        let right_set: HashSet<char> = c_set
            .difference(&a_set)
            .copied()
            .filter(|ch| b_set.contains(ch))
            .collect();

        // Validate output indices: must be exactly batch ∪ left ∪ right, and exclude contract.
        let mut expected_out: HashSet<char> = HashSet::new();
        expected_out.extend(batch_set.iter().copied());
        expected_out.extend(left_set.iter().copied());
        expected_out.extend(right_set.iter().copied());

        if expected_out != c_set {
            return Err(CircuitError::UnsupportedOp);
        }

        // Validate there are no "single-input reduction indices":
        // any axis that is in A but not in output must also be in B (so it's contracted),
        // and similarly for B.
        for ch in a_set.difference(&c_set) {
            if !b_set.contains(ch) {
                return Err(CircuitError::UnsupportedOp);
            }
        }
        for ch in b_set.difference(&c_set) {
            if !a_set.contains(ch) {
                return Err(CircuitError::UnsupportedOp);
            }
        }

        // Need at least one contracted axis for Freivalds-style matmul checking.
        if contract_set.is_empty() {
            return Err(CircuitError::UnsupportedOp);
        }

        // Canonical ordering:
        // - batch/left/right follow output order (so that reshaping matches C's semantics)
        // - contract follows A's order, and B is permuted to match
        let batch: Vec<char> = c_axes
            .iter()
            .copied()
            .filter(|ch| batch_set.contains(ch))
            .collect();
        let left: Vec<char> = c_axes
            .iter()
            .copied()
            .filter(|ch| left_set.contains(ch))
            .collect();
        let right: Vec<char> = c_axes
            .iter()
            .copied()
            .filter(|ch| right_set.contains(ch))
            .collect();
        let contract: Vec<char> = a_axes
            .iter()
            .copied()
            .filter(|ch| contract_set.contains(ch))
            .collect();

        let mut a_order = vec![];
        a_order.extend(batch.iter().copied());
        a_order.extend(left.iter().copied());
        a_order.extend(contract.iter().copied());

        let mut b_order = vec![];
        b_order.extend(batch.iter().copied());
        b_order.extend(contract.iter().copied());
        b_order.extend(right.iter().copied());

        let mut c_order = vec![];
        c_order.extend(batch.iter().copied());
        c_order.extend(left.iter().copied());
        c_order.extend(right.iter().copied());

        // Ensure orders are permutations of original axes.
        if !is_permutation(&a_axes, &a_order)
            || !is_permutation(&b_axes, &b_order)
            || !is_permutation(&c_axes, &c_order)
        {
            return Err(CircuitError::UnsupportedOp);
        }

        let batch_dims: Vec<usize> = batch.iter().map(|ch| axis_dims[ch]).collect();
        let m = product_usize(left.iter().map(|ch| axis_dims[ch]));
        let n = product_usize(contract.iter().map(|ch| axis_dims[ch]));
        let p = product_usize(right.iter().map(|ch| axis_dims[ch]));

        // Allow empty left/right => size 1 (vector/scalar edge cases).
        let m = if left.is_empty() { 1 } else { m };
        let p = if right.is_empty() { 1 } else { p };

        Ok(Self {
            a_axes,
            b_axes,
            c_axes,
            batch,
            left,
            contract,
            right,
            a_order,
            b_order,
            c_order,
            axis_dims,
            batch_dims,
            m,
            n,
            p,
        })
    }

    fn batch_size(&self) -> usize {
        if self.batch_dims.is_empty() {
            1
        } else {
            self.batch_dims.iter().product()
        }
    }

    fn batch_rank(&self) -> usize {
        self.batch_dims.len()
    }
}

pub struct FreivaldsCheck;

impl FreivaldsCheck {
    pub fn verify<F: PrimeField + TensorType + PartialOrd + std::hash::Hash>(
        einsums: &Einsums<F>,
        region: &mut RegionCtx<F>,
        a: &ValTensor<F>,
        b: &ValTensor<F>,
        c: &ValTensor<F>,
        eq: &MatMulEquation,
        seed: Value<F>,
        k_repetitions: usize,
        check_mode: &CheckMode,
    ) -> Result<(), CircuitError> {
        let spec = EinsumMatMulSpec::from_equation_and_tensors(eq, a, b, c)?;

        // Permute A, B, C into canonical axis order so we can reshape them into rank-2 matrices.
        let a_perm = permute_valtensor(a, &spec.a_axes, &spec.a_order)?;
        let b_perm = permute_valtensor(b, &spec.b_axes, &spec.b_order)?;
        let c_perm = permute_valtensor(c, &spec.c_axes, &spec.c_order)?;

        // Provide the (scalar) seed as a ValTensor so the probabilistic gadget can derive r
        // *in-circuit* (i.e., constrained).
        let seed_tensor = Tensor::<ValType<F>>::new(Some(&[ValType::from(seed)]), &[1])
            .map_err(|_| CircuitError::ConstrainError)?;
        let seed_vt: ValTensor<F> = ValTensor::from(seed_tensor);

        let batch_size = spec.batch_size();
        let batch_rank = spec.batch_rank();

        for rep in 0..k_repetitions {
            // Run the Freivalds gadget per batch item.
            for b_idx in 0..batch_size {
                let batch_multi = if batch_rank == 0 {
                    vec![]
                } else {
                    unravel_index(b_idx, &spec.batch_dims)?
                };

                let a_mat = slice_batch_to_matrix(&a_perm, batch_rank, &batch_multi, spec.m, spec.n)?;
                let b_mat = slice_batch_to_matrix(&b_perm, batch_rank, &batch_multi, spec.n, spec.p)?;
                let c_mat = slice_batch_to_matrix(&c_perm, batch_rank, &batch_multi, spec.m, spec.p)?;

                // Domain separation so each repetition / batch uses a distinct r.
                let domain_sep = (rep as u64)
                    .wrapping_mul(batch_size as u64 + 1)
                    .wrapping_add(b_idx as u64);

                let check = probabilistic_ops::FreivaldsCheck::new(&a_mat, &b_mat, &c_mat);
                check.layout(einsums, region, &seed_vt, domain_sep, check_mode)?;
            }
        }

        Ok(())
    }
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

/// Returns true iff `b` is a permutation of `a` (same elements, same multiplicity).
fn is_permutation(a: &[char], b: &[char]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let sa: HashSet<char> = a.iter().copied().collect();
    let sb: HashSet<char> = b.iter().copied().collect();
    sa == sb
}

/// Compute product of an iterator of usizes. Empty iterator => 1.
fn product_usize<I: Iterator<Item = usize>>(it: I) -> usize {
    it.fold(1usize, |acc, x| acc.saturating_mul(x))
}

/// Permute a ValTensor's axes from `current_axes` to `desired_axes`, preserving underlying `ValType` cell refs.
///
/// This clones the underlying `ValType`s into a new tensor in permuted order.
fn permute_valtensor<F: PrimeField + TensorType + PartialOrd + std::hash::Hash>(
    t: &ValTensor<F>,
    current_axes: &[char],
    desired_axes: &[char],
) -> Result<ValTensor<F>, CircuitError> {
    if current_axes.len() != desired_axes.len() {
        return Err(CircuitError::ConstrainError);
    }

    // Build mapping char -> position in current
    let mut pos: HashMap<char, usize> = HashMap::with_capacity(current_axes.len());
    for (i, ch) in current_axes.iter().enumerate() {
        pos.insert(*ch, i);
    }

    let curr_dims: Vec<usize> = t.dims().to_vec();
    if curr_dims.len() != current_axes.len() {
        return Err(CircuitError::ConstrainError);
    }

    let desired_positions: Vec<usize> = desired_axes
        .iter()
        .map(|ch| *pos.get(ch).ok_or(CircuitError::ConstrainError)?)
        .collect();

    let new_dims: Vec<usize> = desired_positions.iter().map(|&i| curr_dims[i]).collect();

    // Read old values in row-major order.
    let inner = t.get_inner_tensor().map_err(|_| CircuitError::ConstrainError)?;
    let old_vals: Vec<ValType<F>> = inner.iter().cloned().collect();

    // Compute strides.
    let old_strides = compute_strides(&curr_dims);
    let new_len = new_dims.iter().product::<usize>();
    if new_len != old_vals.len() {
        return Err(CircuitError::ConstrainError);
    }

    let mut new_vals: Vec<ValType<F>> = Vec::with_capacity(new_len);

    for lin in 0..new_len {
        // multi-index in new order
        let new_multi = unravel_index(lin, &new_dims)?;
        // map into old multi-index
        let mut old_multi = vec![0usize; curr_dims.len()];
        for (new_axis_pos, &old_axis_pos) in desired_positions.iter().enumerate() {
            old_multi[old_axis_pos] = new_multi[new_axis_pos];
        }
        let old_lin = dot_usize(&old_multi, &old_strides)?;
        new_vals.push(old_vals[old_lin].clone());
    }

    let new_tensor =
        Tensor::<ValType<F>>::new(Some(&new_vals), &new_dims).map_err(|_| CircuitError::ConstrainError)?;
    Ok(ValTensor::from(new_tensor))
}

/// Slice a permuted tensor (with batch axes first) to a particular batch item and reshape it into a rank-2 matrix.
///
/// `t` is expected to have shape:
///   [batch..., rows_axes..., cols_axes...]
///
/// We slice all batch axes to a single element and take full ranges on the remaining axes.
/// The slice is then flattened and reshaped into [rows, cols].
fn slice_batch_to_matrix<F: PrimeField + TensorType + PartialOrd + std::hash::Hash>(
    t: &ValTensor<F>,
    batch_rank: usize,
    batch_multi: &[usize],
    rows: usize,
    cols: usize,
) -> Result<ValTensor<F>, CircuitError> {
    if batch_rank != batch_multi.len() {
        return Err(CircuitError::ConstrainError);
    }

    let dims = t.dims().to_vec();
    if batch_rank > dims.len() {
        return Err(CircuitError::ConstrainError);
    }

    let mut ranges = Vec::with_capacity(dims.len());
    for (i, dim) in dims.iter().enumerate() {
        if i < batch_rank {
            let idx = batch_multi[i];
            if idx >= *dim {
                return Err(CircuitError::ConstrainError);
            }
            ranges.push(idx..idx + 1);
        } else {
            ranges.push(0..*dim);
        }
    }

    let mut sliced = t.get_slice(&ranges).map_err(|_| CircuitError::ConstrainError)?;
    sliced.flatten();

    let inner = sliced.get_inner_tensor().map_err(|_| CircuitError::ConstrainError)?;
    let vals: Vec<ValType<F>> = inner.iter().cloned().collect();

    if vals.len() != rows * cols {
        return Err(CircuitError::ConstrainError);
    }

    let mat_tensor =
        Tensor::<ValType<F>>::new(Some(&vals), &[rows, cols]).map_err(|_| CircuitError::ConstrainError)?;
    Ok(ValTensor::from(mat_tensor))
}

/// Compute row-major strides for dims.
fn compute_strides(dims: &[usize]) -> Vec<usize> {
    if dims.is_empty() {
        return vec![];
    }
    let mut strides = vec![1usize; dims.len()];
    for i in (0..dims.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1].saturating_mul(dims[i + 1]);
    }
    strides
}

/// Unravel a linear index into a multi-index under row-major layout.
fn unravel_index(mut idx: usize, dims: &[usize]) -> Result<Vec<usize>, CircuitError> {
    if dims.is_empty() {
        return Ok(vec![]);
    }
    let mut out = vec![0usize; dims.len()];
    for i in (0..dims.len()).rev() {
        let d = dims[i];
        if d == 0 {
            return Err(CircuitError::ConstrainError);
        }
        out[i] = idx % d;
        idx /= d;
    }
    Ok(out)
}

/// Dot-product for usize vectors (used for converting multi-index -> linear index with strides).
fn dot_usize(a: &[usize], b: &[usize]) -> Result<usize, CircuitError> {
    if a.len() != b.len() {
        return Err(CircuitError::ConstrainError);
    }
    Ok(a.iter().zip(b.iter()).map(|(x, y)| x.saturating_mul(*y)).sum())
}
