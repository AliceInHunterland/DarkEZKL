/*!
Probabilistic verification gadgets.

Step 3 (this file):
- Implement a Freivalds-style check gadget that produces Halo2 constraints for verifying
  a single matrix product relation via a mat-vec identity:

For A (m×n), B (n×p), C (m×p), and random r (derived deterministically from a seed), constrain:

    A * (B * r)  ==  C * r

This check is typically repeated `k` times (outside this gadget) with independent `domain_sep`
values to drive soundness error down.

Important:
- This gadget *derives r in-circuit* from a provided scalar `seed` using a simple LCG PRG
  over the circuit field, so that r is bound to the seed by constraints.
- `r` is a vector of field elements (not boolean). Soundness is then per-check ≈ 1/|F|
  (assuming seed is uniformly random / unpredictable to the prover and the PRG behaves
  as a random oracle for the threat model). If you require classic Freivalds 0/1-vectors
  with 2^-k bounds, you must implement bit extraction + booleanity constraints separately.

Implementation strategy:
- Generate r = PRG(seed, domain_sep) as p field elements:
    state0 = seed + domain_sep
    state_{i+1} = state_i * A + C
    r_i = state_{i+1}
- Use existing einsum dot gadgets to compute the needed matrix-vector products:
    u = B r, v = A u, w = C r
  and constrain v == w elementwise.

Phase note:
- This gadget intentionally uses *second-phase* einsum columns for all internal arithmetic
  (phases = [1, 1]). This is compatible with deriving `seed` from Halo2 challenges (Fiat-Shamir),
  which are only available after first-phase commitments.
*/

use crate::circuit::base::BaseOp;
use crate::circuit::ops::chip::einsum::{layouts as einsum_layouts, Einsums};
use crate::circuit::region::RegionCtx;
use crate::circuit::{CheckMode, CircuitError};
use crate::tensor::{Tensor, TensorType, ValTensor, ValType};
use halo2curves::ff::PrimeField;

/// LCG multiplier (u64 interpreted in the circuit field).
const LCG_A_U64: u64 = 6364136223846793005;
/// LCG increment (u64 interpreted in the circuit field).
const LCG_C_U64: u64 = 1442695040888963407;

/// A Freivalds-style probabilistic check for a single matrix product relation.
///
/// Verifies (probabilistically) that `A * B == C` by checking the mat-vec identity:
/// `A * (B * r) == C * r` where `r` is derived deterministically in-circuit from `seed` and
/// `domain_sep`.
///
/// Shapes:
/// - `a`: (m, n)
/// - `b`: (n, p)
/// - `c`: (m, p)
///
/// Seed:
/// - `seed` must be a scalar ValTensor (len == 1). In secure deployments this should be
///   public/instance-backed or otherwise not choosable by the prover after seeing A/B/C.
#[derive(Debug, Clone)]
pub struct FreivaldsCheck<'a, F: PrimeField + TensorType + PartialOrd + std::hash::Hash> {
    /// Left matrix A (m x n)
    pub a: &'a ValTensor<F>,
    /// Right matrix B (n x p)
    pub b: &'a ValTensor<F>,
    /// Claimed product C (m x p)
    pub c: &'a ValTensor<F>,
}

impl<'a, F: PrimeField + TensorType + PartialOrd + std::hash::Hash> FreivaldsCheck<'a, F> {
    /// Create a new Freivalds check gadget borrowing `a`, `b`, and `c`.
    pub fn new(a: &'a ValTensor<F>, b: &'a ValTensor<F>, c: &'a ValTensor<F>) -> Self {
        Self { a, b, c }
    }

    /// Lay out Halo2 constraints for the Freivalds identity:
    /// `A * (B * r) == C * r`, with `r` derived in-circuit from `seed` and `domain_sep`.
    ///
    /// - `seed`: scalar ValTensor (len == 1)
    /// - `domain_sep`: public domain separator (e.g. repetition counter, batch counter, etc.)
    pub fn layout(
        &self,
        einsums: &Einsums<F>,
        region: &mut RegionCtx<F>,
        seed: &ValTensor<F>,
        domain_sep: u64,
        check_mode: &CheckMode,
    ) -> Result<(), CircuitError> {
        let (m, n, p) = infer_matmul_dims(self.a, self.b, self.c)?;

        // Derive r deterministically from seed and domain_sep.
        let r_vec = derive_lcg_vector(einsums, region, seed, domain_sep, p)?;

        // Optional "early fail" check during witness generation (best-effort).
        //
        // IMPORTANT: Do NOT hard-fail synthesis here, because:
        // - this is not a substitute for constraints, and
        // - negative tests (e.g., "soundness" tests) intentionally provide bad witnesses and must
        //   reach the proof verification stage.
        if region.witness_gen() {
            let _ = self.check_in_clear(&r_vec);
        }

        // u = B * r  (length n)
        let u_scalars = mat_vec_mul_vt(einsums, region, self.b, &r_vec, check_mode)?;
        if u_scalars.len() != n {
            return Err(CircuitError::ConstrainError);
        }
        let u_vec = scalars_to_vector(&u_scalars)?;

        // v = A * u  (length m)
        let v_scalars = mat_vec_mul_vt(einsums, region, self.a, &u_vec, check_mode)?;
        if v_scalars.len() != m {
            return Err(CircuitError::ConstrainError);
        }

        // w = C * r  (length m)
        let w_scalars = mat_vec_mul_vt(einsums, region, self.c, &r_vec, check_mode)?;
        if w_scalars.len() != m {
            return Err(CircuitError::ConstrainError);
        }

        // Constrain v == w elementwise
        for (v_i, w_i) in v_scalars.iter().zip(w_scalars.iter()) {
            region.constrain_equal(v_i, w_i)?;
        }

        Ok(())
    }

    /// Optional in-clear check (best-effort).
    ///
    /// If all of A/B/C/r can be fully evaluated to concrete field elements, compute
    /// and compare `A(B r)` and `C r` directly and return an error on mismatch.
    ///
    /// If any value is unknown (or instance-backed), this function returns Ok(()).
    fn check_in_clear(&self, r_vec: &ValTensor<F>) -> Result<(), CircuitError> {
        let (m, n, p) = infer_matmul_dims(self.a, self.b, self.c)?;

        let a = match try_eval_fully(self.a) {
            Some(t) => t,
            None => return Ok(()),
        };
        let b = match try_eval_fully(self.b) {
            Some(t) => t,
            None => return Ok(()),
        };
        let c = match try_eval_fully(self.c) {
            Some(t) => t,
            None => return Ok(()),
        };

        let mut r_val = match try_eval_fully(r_vec) {
            Some(t) => t,
            None => return Ok(()),
        };
        r_val.flatten();
        if r_val.len() != p {
            return Err(CircuitError::ConstrainError);
        }
        let r_plain: Vec<F> = r_val.into_iter().collect();

        // u = B * r
        let u = mat_vec_mul_plain(&b, &r_plain)?;
        if u.len() != n {
            return Err(CircuitError::ConstrainError);
        }

        // v = A * u
        let v = mat_vec_mul_plain(&a, &u)?;
        if v.len() != m {
            return Err(CircuitError::ConstrainError);
        }

        // w = C * r
        let w = mat_vec_mul_plain(&c, &r_plain)?;
        if w.len() != m {
            return Err(CircuitError::ConstrainError);
        }

        if v != w {
            return Err(CircuitError::ConstrainError);
        }

        Ok(())
    }
}

/// Derive a length-`len` vector of field elements from a scalar `seed` and `domain_sep`
/// using a simple LCG, with constraints enforced via existing einsum pairwise gadgets.
///
/// r generation:
///   state0 = seed + domain_sep
///   state_{i+1} = state_i * LCG_A + LCG_C
///   r_i = state_{i+1}
fn derive_lcg_vector<F: PrimeField + TensorType + PartialOrd + std::hash::Hash>(
    einsums: &Einsums<F>,
    region: &mut RegionCtx<F>,
    seed: &ValTensor<F>,
    domain_sep: u64,
    len: usize,
) -> Result<ValTensor<F>, CircuitError> {
    // Ensure seed is scalar (len == 1).
    let mut seed_scalar = seed.clone();
    seed_scalar.flatten();
    if seed_scalar.len() != 1 {
        return Err(CircuitError::ConstrainError);
    }

    let a_const = scalar_constant::<F>(field_from_u64::<F>(LCG_A_U64))?;
    let c_const = scalar_constant::<F>(field_from_u64::<F>(LCG_C_U64))?;
    let ds_const = scalar_constant::<F>(field_from_u64::<F>(domain_sep))?;

    // state0 = seed + domain_sep
    let mut state = einsum_layouts::pairwise(
        &einsums.contraction_gate,
        region,
        &[&seed_scalar, &ds_const],
        BaseOp::Add,
        &[1, 1],
    )?;

    let mut scalars: Vec<ValTensor<F>> = Vec::with_capacity(len);

    for _ in 0..len {
        // state = state * A + C
        let state_mul_a = einsum_layouts::pairwise(
            &einsums.contraction_gate,
            region,
            &[&state, &a_const],
            BaseOp::Mult,
            &[1, 1],
        )?;

        let next = einsum_layouts::pairwise(
            &einsums.contraction_gate,
            region,
            &[&state_mul_a, &c_const],
            BaseOp::Add,
            &[1, 1],
        )?;

        scalars.push(next.clone());
        state = next;
    }

    scalars_to_vector(&scalars)
}

/// Infer (m,n,p) for A(m×n) * B(n×p) == C(m×p).
fn infer_matmul_dims<F: PrimeField + TensorType + PartialOrd + std::hash::Hash>(
    a: &ValTensor<F>,
    b: &ValTensor<F>,
    c: &ValTensor<F>,
) -> Result<(usize, usize, usize), CircuitError> {
    let ad = a.dims();
    let bd = b.dims();
    let cd = c.dims();

    if ad.len() != 2 || bd.len() != 2 || cd.len() != 2 {
        return Err(CircuitError::ConstrainError);
    }

    let (m, n) = (ad[0], ad[1]);
    let (n2, p) = (bd[0], bd[1]);
    let (m2, p2) = (cd[0], cd[1]);

    if n != n2 || m != m2 || p != p2 {
        return Err(CircuitError::ConstrainError);
    }

    Ok((m, n, p))
}

/// Convert a scalar ValTensor (length 1) into its underlying ValType (preserving cell refs).
fn scalar_to_valtype<F: PrimeField + TensorType + PartialOrd>(
    t: &ValTensor<F>,
) -> Result<ValType<F>, CircuitError> {
    let inner = t
        .get_inner_tensor()
        .map_err(|_| CircuitError::ConstrainError)?;
    if inner.len() != 1usize {
        return Err(CircuitError::ConstrainError);
    }
    Ok(inner[0].clone())
}

/// Convert a list of scalar ValTensors into a 1D ValTensor vector.
fn scalars_to_vector<F: PrimeField + TensorType + PartialOrd + std::hash::Hash>(
    scalars: &[ValTensor<F>],
) -> Result<ValTensor<F>, CircuitError> {
    let mut out: Vec<ValType<F>> = Vec::with_capacity(scalars.len());
    for s in scalars {
        out.push(scalar_to_valtype(s)?);
    }
    Ok(ValTensor::from(out))
}

/// Constraint-based matrix-vector multiply for rank-2 matrices using the einsum dot gadget.
///
/// `mat`: shape (rows, cols)
/// `vec`: any shape accepted; flattened internally; must have `len() == cols`
///
/// Returns a Vec of scalar ValTensors of length `rows`.
fn mat_vec_mul_vt<F: PrimeField + TensorType + PartialOrd + std::hash::Hash>(
    einsums: &Einsums<F>,
    region: &mut RegionCtx<F>,
    mat: &ValTensor<F>,
    vec: &ValTensor<F>,
    check_mode: &CheckMode,
) -> Result<Vec<ValTensor<F>>, CircuitError> {
    let d = mat.dims();
    if d.len() != 2 {
        return Err(CircuitError::ConstrainError);
    }
    let rows = d[0];
    let cols = d[1];

    let mut vec_flat = vec.clone();
    vec_flat.flatten();
    if vec_flat.len() != cols {
        return Err(CircuitError::ConstrainError);
    }

    let mut out = Vec::with_capacity(rows);

    for i in 0..rows {
        // Take the i-th row: mat[i, 0..cols]
        let mut row = mat
            .get_slice(&[i..i + 1, 0..cols])
            .map_err(|_| CircuitError::ConstrainError)?;
        row.flatten();

        // dot(row, vec)
        let res = einsum_layouts::dot(
            &einsums.contraction_gate,
            region,
            &[&row, &vec_flat],
            &[1, 1],
            check_mode,
        )?;

        out.push(res);
    }

    Ok(out)
}

/// Best-effort evaluation of a ValTensor into a concrete Tensor<F>.
///
/// Returns None if:
/// - tensor is instance-backed, or
/// - any element is unknown.
fn try_eval_fully<F: PrimeField + TensorType + PartialOrd + std::hash::Hash>(
    t: &ValTensor<F>,
) -> Option<Tensor<F>> {
    if t.is_instance() {
        return None;
    }

    let inner = t.get_inner_tensor().ok()?;
    let mut out: Vec<F> = Vec::with_capacity(inner.len());

    for v in inner.iter() {
        out.push(v.get_felt_eval()?);
    }

    let mut res: Tensor<F> = out.into_iter().into();
    res.reshape(t.dims()).ok()?;
    Some(res)
}

/// Plain matrix-vector multiply for rank-2 tensors.
///
/// `mat` is shape (rows, cols) and `vec` is length `cols`.
fn mat_vec_mul_plain<F: PrimeField + TensorType>(
    mat: &Tensor<F>,
    vec: &[F],
) -> Result<Vec<F>, CircuitError> {
    let d = mat.dims();
    if d.len() != 2 {
        return Err(CircuitError::ConstrainError);
    }
    let rows = d[0];
    let cols = d[1];
    if vec.len() != cols {
        return Err(CircuitError::ConstrainError);
    }

    let mut out = vec![F::ZERO; rows];
    for i in 0..rows {
        let mut acc = F::ZERO;
        for j in 0..cols {
            // row-major idx = i*cols + j
            let mij = mat[i * cols + j];
            acc = acc + (mij * vec[j]);
        }
        out[i] = acc;
    }
    Ok(out)
}

/// Create a scalar constant ValTensor.
fn scalar_constant<F: PrimeField + TensorType + PartialOrd + std::hash::Hash>(
    c: F,
) -> Result<ValTensor<F>, CircuitError> {
    let t = Tensor::<ValType<F>>::new(Some(&[ValType::Constant(c)]), &[1])
        .map_err(|_| CircuitError::ConstrainError)?;
    Ok(ValTensor::from(t))
}

/// Convert a u64 into a PrimeField element by filling the low 8 bytes of the canonical repr.
///
/// This assumes the field modulus is > 2^64 (true for typical halo2 fields like BN254, Pasta, etc.).
fn field_from_u64<F: PrimeField>(x: u64) -> F {
    let mut repr = <F as PrimeField>::Repr::default();
    let bytes = x.to_le_bytes();
    let repr_bytes = repr.as_mut();
    let take = core::cmp::min(8, repr_bytes.len());
    repr_bytes[..take].copy_from_slice(&bytes[..take]);
    Option::<F>::from(F::from_repr(repr)).unwrap_or(F::ZERO)
}

#[cfg(test)]
mod tests {
    use super::*;
    use halo2curves::bn256::Fr;

    #[test]
    fn field_from_u64_roundtrip_small() {
        let x = 123u64;
        let f = field_from_u64::<Fr>(x);
        assert_eq!(f, Fr::from(x));
    }

    #[test]
    fn lcg_constants_nonzero() {
        use halo2curves::ff::Field;
        let a = field_from_u64::<Fr>(LCG_A_U64);
        let c = field_from_u64::<Fr>(LCG_C_U64);
        assert_ne!(a, Fr::zero());
        assert_ne!(c, Fr::zero());
    }
}
