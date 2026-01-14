use crate::circuit::base::BaseOp;
use crate::circuit::layouts::einsum_with_base_ops;
use crate::circuit::region::RegionCtx;
use crate::circuit::{BaseConfig, CheckMode, CircuitError};
use crate::graph::config::{GLOBAL_SETTINGS, ProbOp, ProbOps};
use crate::tensor::{Tensor, TensorError, TensorType, ValTensor, ValType, VarTensor};
use crate::{ExecutionMode, ProbabilisticSettings};
use analysis::{analyze_single_equation, EinsumAnalysis};
use halo2_proofs::circuit::Value;
use halo2_proofs::plonk::{Challenge, ConstraintSystem, Constraints, Expression, FirstPhase, Selector};
use halo2curves::ff::PrimeField;
use itertools::Itertools;
use layouts::{dot, multi_dot, pairwise, prod, sum};
use reduction_planner::Reduction;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::marker::PhantomData;
use std::ops::Range;

mod freivalds;
use crate::circuit::ops::probabilistic::FreivaldsCheck as ProbFreivaldsCheck;
use freivalds::MatMulEquation;

///
pub mod analysis;
///
pub mod circuit_params;
pub(crate) mod layouts;
mod reduction_planner;

/// The maximum number of einsum challenges
pub const NUM_MAX_EINSUM_CHALLENGES: usize = 10;

/// Default seed used when no challenge / public seed is available (e.g. in some
/// keygen / dummy contexts).
const DEFAULT_FREIVALDS_SEED_U64: u64 = 42;

/// A struct representing reductions for the einsums
#[derive(Clone, Debug, Default)]
pub struct Einsums<F: PrimeField + TensorType + PartialOrd> {
    /// custom gate to constrain tensor contractions
    pub(crate) contraction_gate: ContractionConfig<F>,
    /// custom gate to constrain random linear combinations used by Freivalds' argument
    rlc_gates: Vec<RLCConfig<F>>,
}

impl<F: PrimeField + TensorType + PartialOrd + std::hash::Hash> Einsums<F> {
    ///
    pub fn dummy(col_size: usize, num_inner_cols: usize) -> Self {
        let dummy_var = VarTensor::dummy(col_size, num_inner_cols);
        let dummy_contraction_gate = ContractionConfig {
            inputs: [
                [dummy_var.clone(), dummy_var.clone()],
                [dummy_var.clone(), dummy_var.clone()],
            ],
            outputs: [dummy_var.clone(), dummy_var.clone()],
            selectors: BTreeMap::default(),
            _marker: PhantomData,
        };
        Self {
            contraction_gate: dummy_contraction_gate,
            rlc_gates: (0..NUM_MAX_EINSUM_CHALLENGES)
                .map(|_| RLCConfig::dummy(&dummy_var))
                .collect(),
        }
    }

    /// Configure the columns based on universal Einsum analysis
    pub fn configure_universal(
        meta: &mut ConstraintSystem<F>,
        analysis: &EinsumAnalysis,
        num_inner_cols: usize,
        logrows: usize,
    ) -> Self {
        let capacity = analysis.reduction_length;
        let inputs: [VarTensor; 4] = [
            VarTensor::new_advice(meta, logrows, num_inner_cols, capacity),
            VarTensor::new_advice(meta, logrows, num_inner_cols, capacity),
            VarTensor::new_advice_in_second_phase(meta, logrows, num_inner_cols, capacity),
            VarTensor::new_advice_in_second_phase(meta, logrows, num_inner_cols, capacity),
        ];
        let outputs = [
            VarTensor::new_advice(meta, logrows, num_inner_cols, capacity),
            VarTensor::new_advice_in_second_phase(meta, logrows, num_inner_cols, capacity),
        ];
        let contraction_gate = ContractionConfig::new(
            meta,
            &[&[&inputs[0], &inputs[1]], &[&inputs[2], &inputs[3]]],
            &[&outputs[0], &outputs[1]],
        );

        let mut rlc_gates = vec![];
        for _ in 0..analysis.max_num_output_axes {
            let rlc_gate = RLCConfig::new(meta, &[inputs[0].clone(), inputs[2].clone()], &outputs[1]);
            rlc_gates.push(rlc_gate);
        }

        Self {
            contraction_gate,
            rlc_gates,
        }
    }

    /// In dummy layout phase, calling this function will return error
    pub fn challenges(&self) -> Result<Vec<Challenge>, CircuitError> {
        self.rlc_gates
            .iter()
            .map(|gate| gate.challenge.ok_or(CircuitError::ChallengeNotSet))
            .collect::<Result<Vec<_>, _>>()
    }

    fn execution_mode_prob_contract() -> (ExecutionMode, ProbabilisticSettings, ProbOps) {
        GLOBAL_SETTINGS.with(|gs| {
            gs.borrow()
                .as_ref()
                .map(|s| (s.execution_mode, s.probabilistic_settings, s.prob_ops.clone()))
                .unwrap_or((
                    ExecutionMode::Exact,
                    ProbabilisticSettings::default(),
                    ProbOps::default(),
                ))
        })
    }

    /// Retrieve a seed for Freivalds randomness.
    ///
    /// Preferred behavior (public_seed spec):
    /// - seed should come from a public instance cell chosen by the verifier / protocol
    ///   (e.g. block hash), so the prover cannot adaptively choose it.
    ///
    /// Current implementation detail:
    /// - We derive the seed from the first available Halo2 challenge value (Fiat-Shamir),
    ///   falling back to a fixed constant if no challenge is available.
    ///
    /// This keeps proving/keygen deterministic and avoids requiring additional per-op
    /// instance plumbing inside the einsum chip.
    fn freivalds_seed_value(
        _base_config: &BaseConfig<F>,
        region: &mut RegionCtx<F>,
        _prob_settings: &ProbabilisticSettings,
    ) -> Value<F> {
        region
            .challenges()
            .first()
            .copied()
            .unwrap_or_else(|| Value::known(F::from(DEFAULT_FREIVALDS_SEED_U64)))
    }

    ///
    pub fn assign_einsum(
        &self,
        base_config: &BaseConfig<F>,
        region: &mut RegionCtx<F>,
        input_tensors: &[&ValTensor<F>],
        output_tensor: &ValTensor<F>,
        equation: &str,
        check_mode: &CheckMode,
    ) -> Result<(), CircuitError> {
        region.set_num_einsum_inner_cols(self.contraction_gate.block_width());

        let (input_exprs, _) = equation.split_once("->").unwrap();
        let input_exprs = input_exprs.split(",").collect_vec();
        assert_eq!(input_exprs.len(), input_tensors.len());

        let mut input_tensors = input_tensors.iter().copied().cloned().collect_vec();
        let mut output_tensor = output_tensor.clone();

        let mut input_axes_to_dim: HashMap<char, usize> = HashMap::new();
        input_exprs
            .iter()
            .zip(input_tensors.iter())
            .for_each(|(indices, tensor)| {
                indices.chars().zip(tensor.dims()).for_each(|(index, dim)| {
                    if let std::collections::hash_map::Entry::Vacant(e) = input_axes_to_dim.entry(index)
                    {
                        e.insert(*dim);
                    }
                });
            });

        let equation_analysis = analyze_single_equation(&equation, &input_axes_to_dim)?;
        let equation = equation_analysis.equation;

        // Remove trivial axes from tensors
        input_tensors
            .iter_mut()
            .map(|tensor| tensor.remove_trivial_axes())
            .collect::<Result<Vec<_>, TensorError>>()?;
        output_tensor.remove_trivial_axes()?;

        if matches!(equation_analysis.strategy, analysis::EinsumStrategy::BaseOps) {
            let _ = einsum_with_base_ops(
                base_config,
                region,
                &input_tensors.iter().collect_vec(),
                &equation,
            )?;
            return Ok(());
        }

        // Step 5: probabilistic matmul lowering for Strategy::Freivalds.
        //
        // In probabilistic execution mode, for matmul-like einsums:
        //  - do NOT constrain the full tensor contraction (n^3-ish),
        //  - treat the output tensor `C` as an unrestricted witness,
        //  - enforce correctness via k Freivalds-style randomized matmul checks (mat-vec form).
        //
        // If the equation is not a supported matmul-like einsum, we fall back to the exact einsum path.
        let (execution_mode, prob_settings, prob_ops) = Self::execution_mode_prob_contract();
        let prob_enabled_for_matmul = execution_mode == ExecutionMode::Probabilistic
            && (prob_ops.contains(ProbOp::MatMul) || prob_ops.contains(ProbOp::Gemm))
            && prob_settings.k_repetitions > 0;

        if prob_enabled_for_matmul
            && matches!(equation_analysis.strategy, analysis::EinsumStrategy::Freivalds)
            && input_tensors.len() == 2
        {
            if let Some(matmul_eq) = MatMulEquation::parse(&equation) {
                let seed_val = Self::freivalds_seed_value(base_config, region, &prob_settings);
                let seed: ValTensor<F> = vec![ValType::from(seed_val)].into();

                match layout_probabilistic_freivalds_matmul(
                    self,
                    region,
                    &input_tensors[0],
                    &input_tensors[1],
                    &output_tensor,
                    &matmul_eq,
                    &input_axes_to_dim,
                    &seed,
                    prob_settings.k_repetitions as usize,
                    check_mode,
                ) {
                    Ok(()) => return Ok(()),
                    Err(CircuitError::UnsupportedOp) => {
                        // Fallback to exact path for unsupported einsum patterns / shapes.
                    }
                    Err(e) => return Err(e),
                }
            }
        }

        let output_shape = equation_analysis
            .output_indices
            .iter()
            .map(|c| input_axes_to_dim.get(c).copied().unwrap())
            .collect_vec();

        let squashed_output = self.assign_output(region, &output_tensor, output_shape, check_mode)?;

        // reorder the reduction of input tensors and reduce
        let reordered_input_reductions = reduction_planner::input_reductions(&equation).unwrap();
        let mut tensors = input_tensors;
        let mut reduced_input_phase = 0;

        for reduction in reordered_input_reductions.iter() {
            let (input_expr, output_expr) = reduction.expression().split_once("->").unwrap();
            let input_exprs = input_expr.split(",").collect_vec();

            let remaining_axes = output_expr.chars().collect_vec();
            let mut remaining_axes_indices = remaining_axes
                .iter()
                .map(|c| 0..input_axes_to_dim[c])
                .multi_cartesian_product()
                .collect_vec();

            // Dummy value to ensure the for loop runs at least once
            if remaining_axes.is_empty() {
                remaining_axes_indices.push(vec![]);
            }

            let input_tensors = reduction
                .input_indices()
                .iter()
                .map(|idx| tensors[*idx].clone())
                .collect_vec();

            let mut flattened_input_tensors: Vec<Vec<ValTensor<F>>> = vec![vec![]; input_tensors.len()];
            for remaining_axes_indices in remaining_axes_indices {
                // corresponds to 1 running sum of input tensors
                for (i, (input_tensor, input_expr)) in
                    input_tensors.iter().zip(input_exprs.iter()).enumerate()
                {
                    let mut sliced_dim = vec![];
                    input_expr.chars().for_each(|axis| {
                        if let Some(pos) = remaining_axes.iter().position(|c| *c == axis) {
                            sliced_dim.push(remaining_axes_indices[pos]..remaining_axes_indices[pos] + 1);
                        } else {
                            // common axis
                            sliced_dim.push(0..input_axes_to_dim[&axis]);
                        }
                    });
                    let mut sliced_input_tensor = input_tensor.get_slice(&sliced_dim)?;
                    sliced_input_tensor.flatten();
                    flattened_input_tensors[i].push(sliced_input_tensor);
                }
            }
            let flattened_input_tensors = flattened_input_tensors
                .into_iter()
                .map(|tensors| {
                    ValTensor::from(
                        tensors
                            .into_iter()
                            .flat_map(|t| t.get_inner_tensor().unwrap().clone().into_iter())
                            .collect_vec(),
                    )
                })
                .collect_vec();

            let output_dims = output_expr.chars().map(|c| input_axes_to_dim[&c]).collect_vec();

            let contracted_output = match reduction {
                Reduction::RLC {
                    axis,
                    input_phase,
                    challenge_index,
                    ..
                } => {
                    assert_eq!(flattened_input_tensors.len(), 1);
                    let rlc_len = input_axes_to_dim[axis];
                    let mut result = self.rlc_gates[*challenge_index].assign_rlc(
                        region,
                        &flattened_input_tensors[0],
                        region.challenges()[*challenge_index],
                        rlc_len,
                        *input_phase,
                        check_mode,
                    )?;
                    result.reshape(&output_dims)?;
                    result
                }
                Reduction::Contraction { axis, input_phases, .. } => match axis {
                    Some(axis) => {
                        let dot_product_len = input_axes_to_dim[axis];
                        assign_input_contraction(
                            &self.contraction_gate,
                            region,
                            flattened_input_tensors,
                            dot_product_len,
                            &output_dims,
                            input_phases,
                            check_mode,
                        )?
                    }
                    None => {
                        let mut result = assign_pairwise_mult(
                            &self.contraction_gate,
                            region,
                            flattened_input_tensors,
                            input_phases,
                        )?;
                        result.reshape(&output_dims)?;
                        result
                    }
                },
            };
            tensors.push(contracted_output);
            reduced_input_phase = reduction.output_phase();
        }
        tensors.retain(|tensor| tensor.is_singleton());

        let scalars: ValTensor<F> = tensors
            .into_iter()
            .map(|t| t.get_inner_tensor().unwrap().get_scalar())
            .collect_vec()
            .into();
        let squashed_input = prod(
            &self.contraction_gate,
            region,
            &[&scalars],
            reduced_input_phase,
            check_mode,
        )?;

        region.constrain_equal(&squashed_input, &squashed_output)
    }

    fn assign_output(
        &self,
        region: &mut RegionCtx<F>,
        output: &ValTensor<F>,
        output_shape: Vec<usize>,
        check_mode: &CheckMode,
    ) -> Result<ValTensor<F>, CircuitError> {
        let mut intermediate_values = output.clone();

        let challenges = region
            .challenges()
            .iter()
            .take(output_shape.len())
            .copied()
            .collect_vec();

        // Loop over the output axes
        for (idx, (rlc_config, challenge)) in self
            .rlc_gates
            .iter()
            .take(output_shape.len())
            .zip(challenges.iter())
            .rev()
            .enumerate()
        {
            let rlc_len = output_shape[output_shape.len() - idx - 1];
            intermediate_values.flatten();
            let phase = if idx > 0 { 1 } else { 0 };
            intermediate_values =
                rlc_config.assign_rlc(region, &intermediate_values, *challenge, rlc_len, phase, check_mode)?;
        }

        let phase = if challenges.len() > 0 { 1 } else { 0 };
        let output_var = self.contraction_gate.get_output_var([phase].as_slice().into());
        let res = region.assign_einsum(output_var, &intermediate_values)?;
        region.increment_einsum_col_coord(1);

        Ok(res)
    }
}

fn assign_pairwise_mult<F: PrimeField + TensorType + PartialOrd + std::hash::Hash>(
    config: &ContractionConfig<F>,
    region: &mut RegionCtx<F>,
    flattened_tensors: Vec<ValTensor<F>>,
    input_phases: &[usize],
) -> Result<ValTensor<F>, CircuitError> {
    assert_eq!(flattened_tensors.len(), input_phases.len());
    let (result, _) = flattened_tensors
        .into_iter()
        .zip(input_phases.iter().cloned())
        .reduce(|(acc, acc_phase), (input, phase)| {
            (
                pairwise(config, region, &[&acc, &input], BaseOp::Mult, &[acc_phase, phase]).unwrap(),
                std::cmp::max(acc_phase, phase),
            )
        })
        .unwrap();
    Ok(result)
}

fn assign_input_contraction<F: PrimeField + TensorType + PartialOrd + std::hash::Hash>(
    config: &ContractionConfig<F>,
    region: &mut RegionCtx<F>,
    flattened_tensors: Vec<ValTensor<F>>,
    dot_product_len: usize,
    output_shape: &[usize],
    input_phases: &[usize],
    check_mode: &CheckMode,
) -> Result<ValTensor<F>, CircuitError> {
    assert_eq!(flattened_tensors.len(), input_phases.len());
    let num_dot_products = output_shape.iter().product();
    let mut dot_product_results = vec![];
    for chunk_idx in 0..num_dot_products {
        let start = chunk_idx * dot_product_len;
        let tensors: Vec<_> = flattened_tensors
            .iter()
            .map(|tensor| tensor.get_slice(&[start..(start + dot_product_len)]))
            .collect::<Result<Vec<_>, TensorError>>()?;
        let result = if tensors.len() == 1 {
            sum(config, region, &[&tensors[0]], input_phases[0], check_mode)?
        } else if tensors.len() == 2 {
            dot(config, region, &[&tensors[0], &tensors[1]], &[input_phases[0], input_phases[1]], check_mode)?
        } else {
            multi_dot(
                config,
                region,
                tensors.iter().collect_vec().as_slice(),
                input_phases,
                check_mode,
            )?
        };
        dot_product_results.push(result.get_inner_tensor()?.get_scalar());
    }
    let mut tensor = ValTensor::from(dot_product_results);
    tensor.reshape(output_shape)?;
    Ok(tensor)
}

#[derive(Debug, Clone, Copy, Ord, PartialOrd, Eq, PartialEq, Hash)]
enum InputPhases {
    FirstPhase,
    SecondPhase,
    BothFirstPhase,  // [0, 0]
    Mixed,           // [0, 1] or [1, 0]
    BothSecondPhase, // [1, 1]
}

impl From<&[usize]> for InputPhases {
    fn from(phases: &[usize]) -> Self {
        match phases {
            [0] => Self::FirstPhase,
            [1] => Self::SecondPhase,
            [0, 0] => Self::BothFirstPhase,
            [0, 1] | [1, 0] => Self::Mixed,
            [1, 1] => Self::BothSecondPhase,
            _ => panic!("Invalid phase combination"),
        }
    }
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct BaseOpInfo {
    pub op_kind: BaseOp,
    pub input_phases: InputPhases,
}

/// `ContractionConfig` is the custom gate to constrain tensor contractions
#[derive(Clone, Debug, Default)]
pub(crate) struct ContractionConfig<F: PrimeField + TensorType + PartialOrd> {
    // [[first phase, first phase], [second phase, second phase]]
    inputs: [[VarTensor; 2]; 2],
    // [first phase, second phase]
    outputs: [VarTensor; 2],
    // (BaseOpInfo, block index, inner column index) -> selector
    selectors: BTreeMap<(BaseOpInfo, usize, usize), Selector>,
    _marker: PhantomData<F>,
}

impl<F: PrimeField + TensorType + PartialOrd> ContractionConfig<F> {
    fn get_input_vars(&self, input_phases: InputPhases) -> Vec<&VarTensor> {
        match input_phases {
            InputPhases::FirstPhase => vec![&self.inputs[0][0]],
            InputPhases::SecondPhase => vec![&self.inputs[1][0]],
            InputPhases::BothFirstPhase => vec![&self.inputs[0][0], &self.inputs[0][1]],
            InputPhases::Mixed => vec![&self.inputs[0][0], &self.inputs[1][0]],
            InputPhases::BothSecondPhase => vec![&self.inputs[1][0], &self.inputs[1][1]],
        }
    }

    fn get_output_var(&self, input_phases: InputPhases) -> &VarTensor {
        match input_phases {
            InputPhases::FirstPhase => &self.outputs[0],
            InputPhases::SecondPhase => &self.outputs[1],
            InputPhases::BothFirstPhase => &self.outputs[0],
            InputPhases::Mixed => &self.outputs[1],
            InputPhases::BothSecondPhase => &self.outputs[1],
        }
    }

    fn block_width(&self) -> usize {
        self.outputs[0].num_inner_cols()
    }

    fn new(
        meta: &mut ConstraintSystem<F>,
        inputs: &[&[&VarTensor; 2]; 2],
        outputs: &[&VarTensor; 2],
    ) -> Self {
        let mut selectors = BTreeMap::new();
        let num_blocks = outputs[0].num_blocks();
        let block_width = outputs[0].num_inner_cols();
        for input_phases in [
            InputPhases::BothFirstPhase,
            InputPhases::Mixed,
            InputPhases::BothSecondPhase,
        ] {
            for i in 0..num_blocks {
                for j in 0..block_width {
                    selectors.insert(
                        (
                            BaseOpInfo {
                                op_kind: BaseOp::Mult,
                                input_phases,
                            },
                            i,
                            j,
                        ),
                        meta.selector(),
                    );
                    selectors.insert(
                        (
                            BaseOpInfo {
                                op_kind: BaseOp::Add,
                                input_phases,
                            },
                            i,
                            j,
                        ),
                        meta.selector(),
                    );
                }
                for i in 0..num_blocks {
                    selectors.insert(
                        (
                            BaseOpInfo {
                                op_kind: BaseOp::DotInit,
                                input_phases,
                            },
                            i,
                            0,
                        ),
                        meta.selector(),
                    );
                    selectors.insert(
                        (
                            BaseOpInfo {
                                op_kind: BaseOp::Dot,
                                input_phases,
                            },
                            i,
                            0,
                        ),
                        meta.selector(),
                    );
                }
            }
        }

        for input_phases in [InputPhases::FirstPhase, InputPhases::SecondPhase] {
            for i in 0..num_blocks {
                selectors.insert(
                    (
                        BaseOpInfo {
                            op_kind: BaseOp::SumInit,
                            input_phases,
                        },
                        i,
                        0,
                    ),
                    meta.selector(),
                );
                selectors.insert(
                    (
                        BaseOpInfo {
                            op_kind: BaseOp::Sum,
                            input_phases,
                        },
                        i,
                        0,
                    ),
                    meta.selector(),
                );
                selectors.insert(
                    (
                        BaseOpInfo {
                            op_kind: BaseOp::CumProdInit,
                            input_phases,
                        },
                        i,
                        0,
                    ),
                    meta.selector(),
                );
                selectors.insert(
                    (
                        BaseOpInfo {
                            op_kind: BaseOp::CumProd,
                            input_phases,
                        },
                        i,
                        0,
                    ),
                    meta.selector(),
                );
            }
        }
        for ((base_op, block_idx, inner_col_idx), selector) in selectors.iter() {
            let inputs = match base_op.input_phases {
                InputPhases::FirstPhase => vec![inputs[0][0]],
                InputPhases::SecondPhase => vec![inputs[1][0]],
                InputPhases::BothFirstPhase => vec![inputs[0][0], inputs[0][1]],
                InputPhases::Mixed => vec![inputs[0][0], inputs[1][0]],
                InputPhases::BothSecondPhase => vec![inputs[1][0], inputs[1][1]],
            };
            let output = match base_op.input_phases {
                InputPhases::FirstPhase => outputs[0],
                InputPhases::SecondPhase => outputs[1],
                InputPhases::BothFirstPhase => outputs[0],
                InputPhases::Mixed => outputs[1],
                InputPhases::BothSecondPhase => outputs[1],
            };
            assert_eq!(inputs.len(), base_op.op_kind.num_inputs());
            match base_op.op_kind {
                BaseOp::Mult | BaseOp::Add => {
                    meta.create_gate(base_op.op_kind.as_str(), |meta| {
                        let selector = meta.query_selector(*selector);

                        let zero = Expression::<F>::Constant(F::ZERO);
                        let mut qis = vec![zero; 2];
                        for (q_i, input) in qis.iter_mut().zip(inputs) {
                            *q_i = input
                                .query_rng(meta, *block_idx, *inner_col_idx, 0, 1)
                                .expect("contraction config: input query failed")[0]
                                .clone()
                        }
                        // Get output expressions for each input channel
                        let (rotation_offset, rng) = base_op.op_kind.query_offset_rng();
                        let constraints = {
                            let expected_output: Tensor<Expression<F>> = output
                                .query_rng(meta, *block_idx, *inner_col_idx, rotation_offset, rng)
                                .expect("contraction config: output query failed");

                            let res = base_op.op_kind.nonaccum_f((qis[0].clone(), qis[1].clone()));
                            vec![expected_output[base_op.op_kind.constraint_idx()].clone() - res]
                        };
                        Constraints::with_selector(selector, constraints)
                    });
                }
                _ => {
                    meta.create_gate(base_op.op_kind.as_str(), |meta| {
                        let selector = meta.query_selector(*selector);
                        let mut qis = vec![vec![]; 2];
                        for (q_i, input) in qis.iter_mut().zip(inputs) {
                            *q_i = input
                                .query_whole_block(meta, *block_idx, 0, 1)
                                .expect("contraction config: input query failed")
                                .into_iter()
                                .collect()
                        }
                        // Get output expressions for each input channel
                        let (rotation_offset, rng) = base_op.op_kind.query_offset_rng();
                        let expected_output: Tensor<Expression<F>> = output
                            .query_rng(meta, *block_idx, 0, rotation_offset, rng)
                            .expect("contraction config: output query failed");

                        let res = base_op.op_kind.accum_f(
                            expected_output[0].clone(),
                            qis[1].clone(),
                            qis[0].clone(),
                        );
                        let constraints = vec![expected_output[base_op.op_kind.constraint_idx()].clone() - res];

                        Constraints::with_selector(selector, constraints)
                    });
                }
            }
        }

        let first_phase_inputs: [VarTensor; 2] =
            inputs[0].iter().copied().cloned().collect_vec().try_into().unwrap();
        let second_phase_inputs: [VarTensor; 2] =
            inputs[1].iter().copied().cloned().collect_vec().try_into().unwrap();

        Self {
            inputs: [first_phase_inputs, second_phase_inputs],
            outputs: [outputs[0].clone(), outputs[1].clone()],
            selectors,
            _marker: PhantomData,
        }
    }
}

/// `RLCConfig` is the custom gate used for random linear combination with the specific challenge
#[derive(Clone, Debug)]
struct RLCConfig<F: PrimeField + TensorType + PartialOrd> {
    /// The challenge used for the random linear combination
    /// Challenge is `None` in the dummy configuration
    pub challenge: Option<Challenge>,
    /// [first phase, second phase]
    pub inputs: [VarTensor; 2],
    pub output: VarTensor,
    /// (phase of input, block index) -> (init selector, acc selector)
    pub selectors: BTreeMap<(usize, usize), (Selector, Selector)>,
    _marker: PhantomData<F>,
}

impl<F: PrimeField + TensorType + PartialOrd + std::hash::Hash> RLCConfig<F> {
    fn dummy(dummy_var: &VarTensor) -> Self {
        let challenge = None;
        let inputs = [dummy_var.clone(), dummy_var.clone()];
        let output = dummy_var.clone();
        let selectors = BTreeMap::new();
        Self {
            challenge,
            inputs,
            output,
            selectors,
            _marker: PhantomData,
        }
    }

    fn new(meta: &mut ConstraintSystem<F>, inputs: &[VarTensor; 2], output: &VarTensor) -> Self {
        let challenge = meta.challenge_usable_after(FirstPhase);

        let mut selectors = BTreeMap::new();
        for (phase, _input) in inputs.iter().enumerate() {
            for block_idx in 0..output.num_blocks() {
                let selector = (meta.selector(), meta.selector());
                selectors.insert((phase, block_idx), selector);
            }
        }
        let block_width = output.num_inner_cols();
        let powers_of_challenge = (0..block_width)
            .scan(Expression::Constant(F::ONE), |r_power, _| {
                *r_power = r_power.clone() * challenge.expr();
                Some(r_power.clone())
            })
            .collect_vec();
        for ((phase, block_idx), (init_selector, acc_selector)) in selectors.iter() {
            meta.create_gate("init", |meta| {
                let selector = meta.query_selector(*init_selector);
                let input_exprs = inputs[*phase]
                    .query_whole_block(meta, *block_idx, 0, 1)
                    .expect("rlc config: input query failed")
                    .into_iter()
                    .collect();
                let constraints = {
                    let expected_output: Tensor<Expression<F>> = output
                        .query_rng(meta, *block_idx, 0, 0, 1)
                        .expect("rlc config: output query failed");

                    let res = BaseOp::Dot.accum_f(
                        Expression::Constant(F::ZERO),
                        powers_of_challenge.iter().cloned().rev().collect_vec(),
                        input_exprs,
                    );
                    vec![expected_output[0].clone() - res]
                };
                Constraints::with_selector(selector, constraints)
            });
            meta.create_gate("acc", |meta| {
                let selector = meta.query_selector(*acc_selector);
                let input_exprs = inputs[*phase]
                    .query_whole_block(meta, *block_idx, 0, 1)
                    .expect("rlc config: input query failed")
                    .into_iter()
                    .collect();
                let constraints = {
                    let expected_output: Tensor<Expression<F>> = output
                        .query_rng(meta, *block_idx, 0, -1, 2)
                        .expect("rlc config: output query failed");

                    let res = BaseOp::Dot.accum_f(
                        expected_output[0].clone() * powers_of_challenge.last().cloned().unwrap(),
                        powers_of_challenge.iter().cloned().rev().collect_vec(),
                        input_exprs,
                    );
                    vec![expected_output[1].clone() - res]
                };
                Constraints::with_selector(selector, constraints)
            });
        }
        Self {
            inputs: inputs.clone(),
            output: output.clone(),
            selectors,
            challenge: Some(challenge),
            _marker: PhantomData,
        }
    }

    fn assign_rlc(
        &self,
        region: &mut RegionCtx<F>,
        flattened_input: &ValTensor<F>,
        challenge: Value<F>,
        rlc_len: usize,
        phase: usize,
        check_mode: &CheckMode,
    ) -> Result<ValTensor<F>, CircuitError> {
        region.flush_einsum()?;
        let block_width = self.output.num_inner_cols();
        let powers_of_challenge = (0..block_width)
            .scan(Value::known(F::ONE), |challenge_power, _| {
                *challenge_power = challenge_power.clone() * challenge;
                Some(challenge_power.clone())
            })
            .collect_vec();
        let mut rlc_results: Vec<ValType<F>> = vec![];
        for tensor in flattened_input.get_inner_tensor()?.chunks_exact(rlc_len) {
            let running_sums = tensor
                .iter()
                .chunks(block_width)
                .into_iter()
                .scan(Value::known(F::ZERO), |state, val| {
                    let curr_sum: Value<F> = val
                        .into_iter()
                        .zip(powers_of_challenge.iter().rev())
                        .map(|(v, c_power)| {
                            c_power.and_then(|c_power| {
                                v.get_felt_eval()
                                    .and_then(|v| Some(Value::known(c_power * v)))
                                    .unwrap_or(Value::unknown())
                            })
                        })
                        .reduce(|acc, v| acc + v)
                        .unwrap();
                    *state = *state * powers_of_challenge.last().unwrap() + curr_sum;
                    Some(*state)
                })
                .collect_vec();

            let assigned_len = {
                let mut input: ValTensor<F> = tensor.iter().collect_vec().into();
                input.pad_to_zero_rem(block_width, ValType::Constant(F::ZERO))?;
                let (_, len) = region.assign_einsum_with_duplication_unconstrained(&self.inputs[phase], &input)?;
                len
            };
            let (assigned_output, assigned_output_len) = {
                let running_sums = running_sums.into_iter().map(ValType::from).collect_vec();
                region.assign_einsum_with_duplication_constrained(&self.output, &running_sums.into(), check_mode)?
            };

            (0..assigned_output_len)
                .map(|i| {
                    let (block_idx, _, z) =
                        self.output.cartesian_coord(region.einsum_col_coord() + i * block_width);
                    if z == 0 && i > 0 {
                        return Ok(());
                    }
                    let selector = if i == 0 {
                        self.selectors.get(&(phase, block_idx)).map(|(init, _)| init)
                    } else {
                        self.selectors.get(&(phase, block_idx)).map(|(_, acc)| acc)
                    };
                    region.enable(selector, z)?;
                    Ok(())
                })
                .collect::<Result<Vec<_>, CircuitError>>()?;
            rlc_results.push(assigned_output.last()?.get_inner_tensor()?.get_scalar());

            region.increment_einsum_col_coord(assigned_len);
        }
        Ok(rlc_results.into())
    }
}

/// Layout probabilistic Freivalds-style matmul checks for an einsum that is matmul-like.
///
/// This is a best-effort lowering:
/// - Supports pure (non-batched) matmul and batched matmul where the batch axes appear in A/B/C.
/// - Any remaining non-batch axes are flattened into (m,n,p) matrices per batch.
/// - If the pattern is not supported, returns `CircuitError::UnsupportedOp` so the caller can fall back.
fn layout_probabilistic_freivalds_matmul<F: PrimeField + TensorType + PartialOrd + std::hash::Hash>(
    einsums: &Einsums<F>,
    region: &mut RegionCtx<F>,
    a: &ValTensor<F>,
    b: &ValTensor<F>,
    c: &ValTensor<F>,
    eq: &MatMulEquation,
    axes_to_dim: &HashMap<char, usize>,
    seed: &ValTensor<F>,
    k_repetitions: usize,
    check_mode: &CheckMode,
) -> Result<(), CircuitError> {
    let a_axes: Vec<char> = eq.a_indices.chars().collect();
    let b_axes: Vec<char> = eq.b_indices.chars().collect();
    let c_axes: Vec<char> = eq.c_indices.chars().collect();

    // Basic sanity: rank matches number of indices.
    if a.dims().len() != a_axes.len() || b.dims().len() != b_axes.len() || c.dims().len() != c_axes.len() {
        return Err(CircuitError::UnsupportedOp);
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

    if contract_set.is_empty() {
        return Err(CircuitError::UnsupportedOp);
    }

    // Fixed axis orders (important for consistent flattening):
    let batch_axes = c_axes
        .iter()
        .copied()
        .filter(|ch| batch_set.contains(ch))
        .collect_vec();
    let left_axes = c_axes
        .iter()
        .copied()
        .filter(|ch| left_set.contains(ch))
        .collect_vec();
    let right_axes = c_axes
        .iter()
        .copied()
        .filter(|ch| right_set.contains(ch))
        .collect_vec();
    let contract_axes = a_axes
        .iter()
        .copied()
        .filter(|ch| contract_set.contains(ch))
        .collect_vec();

    // Validate "matmul-like": after removing batch axes, A should be (left, contract),
    // B should be (contract, right), C should be (left, right) (up to reordering).
    {
        let a_non_batch: HashSet<char> = a_set.difference(&batch_set).copied().collect();
        let b_non_batch: HashSet<char> = b_set.difference(&batch_set).copied().collect();
        let c_non_batch: HashSet<char> = c_set.difference(&batch_set).copied().collect();

        let expected_a: HashSet<char> = left_set.union(&contract_set).copied().collect();
        let expected_b: HashSet<char> = right_set.union(&contract_set).copied().collect();
        let expected_c: HashSet<char> = left_set.union(&right_set).copied().collect();

        if a_non_batch != expected_a || b_non_batch != expected_b || c_non_batch != expected_c {
            return Err(CircuitError::UnsupportedOp);
        }
    }

    let dim = |ch: char| -> Result<usize, CircuitError> {
        axes_to_dim
            .get(&ch)
            .copied()
            .ok_or(CircuitError::UnsupportedOp)
    };

    let m = left_axes.iter().copied().map(dim).try_fold(1usize, |acc, v| Ok(acc * v?))?;
    let n = contract_axes.iter().copied().map(dim).try_fold(1usize, |acc, v| Ok(acc * v?))?;
    let p = right_axes.iter().copied().map(dim).try_fold(1usize, |acc, v| Ok(acc * v?))?;

    // Batch cartesian product
    let mut batch_indices = batch_axes
        .iter()
        .copied()
        .map(|ch| Ok(0..dim(ch)?))
        .collect::<Result<Vec<Range<usize>>, CircuitError>>()?
        .into_iter()
        .multi_cartesian_product()
        .collect_vec();

    if batch_axes.is_empty() {
        batch_indices = vec![vec![]];
    }

    // For each batch slice, reshape into matrices and run k Freivalds repetitions.
    for (batch_flat, batch_idx) in batch_indices.iter().enumerate() {
        let a_slice = slice_by_batch(a, &a_axes, &batch_axes, batch_idx, axes_to_dim)?;
        let b_slice = slice_by_batch(b, &b_axes, &batch_axes, batch_idx, axes_to_dim)?;
        let c_slice = slice_by_batch(c, &c_axes, &batch_axes, batch_idx, axes_to_dim)?;

        let a_axes_nb = a_axes.iter().copied().filter(|ch| !batch_set.contains(ch)).collect_vec();
        let b_axes_nb = b_axes.iter().copied().filter(|ch| !batch_set.contains(ch)).collect_vec();
        let c_axes_nb = c_axes.iter().copied().filter(|ch| !batch_set.contains(ch)).collect_vec();

        let a_desired = left_axes.iter().copied().chain(contract_axes.iter().copied()).collect_vec();
        let b_desired = contract_axes.iter().copied().chain(right_axes.iter().copied()).collect_vec();
        let c_desired = left_axes.iter().copied().chain(right_axes.iter().copied()).collect_vec();

        let mut a_mat = reorder_axes(&a_slice, &a_axes_nb, &a_desired, axes_to_dim)?;
        a_mat.flatten();
        a_mat.reshape(&[m, n]).map_err(|_| CircuitError::ConstrainError)?;

        let mut b_mat = reorder_axes(&b_slice, &b_axes_nb, &b_desired, axes_to_dim)?;
        b_mat.flatten();
        b_mat.reshape(&[n, p]).map_err(|_| CircuitError::ConstrainError)?;

        let mut c_mat = reorder_axes(&c_slice, &c_axes_nb, &c_desired, axes_to_dim)?;
        c_mat.flatten();
        c_mat.reshape(&[m, p]).map_err(|_| CircuitError::ConstrainError)?;

        let domain_base = (batch_flat as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15u64);

        for rep in 0..k_repetitions {
            let domain_sep = domain_base ^ (rep as u64);
            let checker = ProbFreivaldsCheck::new(&a_mat, &b_mat, &c_mat);
            checker.layout(einsums, region, seed, domain_sep, check_mode)?;
        }
    }

    Ok(())
}

fn slice_by_batch<F: PrimeField + TensorType + PartialOrd + std::hash::Hash>(
    t: &ValTensor<F>,
    t_axes: &[char],
    batch_axes: &[char],
    batch_idx: &[usize],
    axes_to_dim: &HashMap<char, usize>,
) -> Result<ValTensor<F>, CircuitError> {
    if batch_axes.len() != batch_idx.len() {
        return Err(CircuitError::UnsupportedOp);
    }

    let mut ranges: Vec<Range<usize>> = Vec::with_capacity(t_axes.len());
    for ch in t_axes.iter().copied() {
        let d = axes_to_dim.get(&ch).copied().ok_or(CircuitError::UnsupportedOp)?;
        if let Some(pos) = batch_axes.iter().position(|b| *b == ch) {
            let i = batch_idx[pos];
            ranges.push(i..i + 1);
        } else {
            ranges.push(0..d);
        }
    }

    let mut sliced = t.get_slice(&ranges).map_err(|_| CircuitError::ConstrainError)?;
    sliced
        .remove_trivial_axes()
        .map_err(|_| CircuitError::ConstrainError)?;
    Ok(sliced)
}

/// Reorder the axes of `t` from `current_axes` order to `desired_axes` order (no batching),
/// producing a new ValTensor with dims matching `desired_axes`.
fn reorder_axes<F: PrimeField + TensorType + PartialOrd + std::hash::Hash>(
    t: &ValTensor<F>,
    current_axes: &[char],
    desired_axes: &[char],
    axes_to_dim: &HashMap<char, usize>,
) -> Result<ValTensor<F>, CircuitError> {
    if current_axes.len() != t.dims().len() {
        return Err(CircuitError::ConstrainError);
    }
    if current_axes.len() != desired_axes.len() {
        return Err(CircuitError::UnsupportedOp);
    }

    // Validate permutation
    {
        let s1: HashSet<char> = current_axes.iter().copied().collect();
        let s2: HashSet<char> = desired_axes.iter().copied().collect();
        if s1 != s2 {
            return Err(CircuitError::UnsupportedOp);
        }
    }

    let mut pos_in_current: HashMap<char, usize> = HashMap::with_capacity(current_axes.len());
    for (i, ch) in current_axes.iter().copied().enumerate() {
        pos_in_current.insert(ch, i);
    }

    let current_dims = t.dims();
    let desired_dims = desired_axes
        .iter()
        .copied()
        .map(|ch| axes_to_dim.get(&ch).copied().ok_or(CircuitError::UnsupportedOp))
        .collect::<Result<Vec<_>, _>>()?;

    let inner = t.get_inner_tensor().map_err(|_| CircuitError::ConstrainError)?;

    let mut desired_indices = desired_dims
        .iter()
        .map(|d| 0..*d)
        .multi_cartesian_product()
        .collect_vec();
    if desired_dims.is_empty() {
        desired_indices = vec![vec![]];
    }

    let mut out_vals: Vec<ValType<F>> = Vec::with_capacity(inner.len());
    for idxs_desired in desired_indices.into_iter() {
        let mut idxs_current = vec![0usize; current_axes.len()];
        for (k, axis) in desired_axes.iter().copied().enumerate() {
            let pos = *pos_in_current.get(&axis).ok_or(CircuitError::UnsupportedOp)?;
            idxs_current[pos] = idxs_desired[k];
        }

        let lin = linear_index(&idxs_current, &current_dims)?;
        out_vals.push(inner[lin].clone());
    }

    let mut out: ValTensor<F> = out_vals.into();
    out.reshape(&desired_dims).map_err(|_| CircuitError::ConstrainError)?;
    Ok(out)
}

fn linear_index(idxs: &[usize], dims: &[usize]) -> Result<usize, CircuitError> {
    if idxs.len() != dims.len() {
        return Err(CircuitError::ConstrainError);
    }
    let mut acc = 0usize;
    for (i, d) in idxs.iter().zip(dims.iter()) {
        if *i >= *d {
            return Err(CircuitError::ConstrainError);
        }
        acc = acc * (*d) + (*i);
    }
    Ok(acc)
}
