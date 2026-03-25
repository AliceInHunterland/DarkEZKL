use std::fs::File;
use std::io::{BufReader, BufWriter};

use halo2_proofs::arithmetic::Field;
use halo2_proofs::circuit::{Layouter, SimpleFloorPlanner, Value};
use halo2_proofs::plonk::Circuit;
use halo2_proofs::plonk::VerifyingKey;
use halo2_proofs::plonk::{ConstraintSystem, Error};
use halo2_proofs::poly::commitment::CommitmentScheme;

use halo2curves::bn256::{Fr as Fp, G1Affine};
use itertools::Itertools;
use serde::{Deserialize, Serialize};

use crate::circuit::region::ConstantsMap;
use crate::circuit::table::Range;
use crate::circuit::CheckMode;
use crate::fieldutils::IntegerRep;
use crate::tensor::{Tensor, ValTensor};

use super::errors::GraphError;
use super::input::{FileSourceInner, GraphData};
use super::model::{effective_run_args_for_model, Model, ModelConfig};
use super::modules::{GraphModules, ModuleConfigs, ModuleForwardResult, ModuleSizes};
use super::vars::VarVisibility;
use super::{GraphSettings, GraphWitness};

#[derive(Clone, Debug)]
pub struct GraphCircuitConfig {
    pub model: ModelConfig,
    pub modules: ModuleConfigs,
}

/// A compiled circuit (model + settings). Witness is loaded separately at runtime.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GraphCircuit {
    pub model: Model,
    pub settings: GraphSettings,

    #[serde(skip)]
    witness: Option<GraphWitness>,
}

fn sync_settings_execution_contract(settings: &mut GraphSettings) {
    // Treat top-level settings as the canonical execution contract, then mirror them
    // into run_args for older codepaths that still read from run_args.
    settings.run_args.execution_mode = settings.execution_mode;
    settings.run_args.prob_k = settings.prob_k as _;
    settings.run_args.prob_seed_mode = settings.probabilistic_settings.seed_mode;
    settings.run_args.prob_ops = settings.prob_ops.clone();
    settings.run_args.check_mode = settings.check_mode;

    settings.execution_mode = settings.run_args.execution_mode;
    settings.prob_k = settings.run_args.prob_k as u32;
    settings.prob_ops = settings.run_args.prob_ops.clone();
    settings.probabilistic_settings.k_repetitions = settings.prob_k;
    settings.probabilistic_settings.seed_mode = settings.run_args.prob_seed_mode;
    settings.check_mode = settings.run_args.check_mode;
}

fn normalize_settings_for_model(
    settings: &mut GraphSettings,
    model: &Model,
) -> Result<(), GraphError> {
    let effective_run_args = effective_run_args_for_model(model, &settings.run_args);
    let run_args_changed = effective_run_args != settings.run_args;

    if run_args_changed {
        log::info!(
            "recomputing loaded circuit settings against actual model/segment: execution_mode={} -> {}, disable_freivalds={} -> {}, prob_ops={}",
            settings.run_args.execution_mode,
            effective_run_args.execution_mode,
            settings.run_args.disable_freivalds,
            effective_run_args.disable_freivalds,
            effective_run_args.prob_ops,
        );

        let prev_num_blinding_factors = settings.num_blinding_factors;
        let prev_timestamp = settings.timestamp;

        let mut refreshed = model.gen_params(&effective_run_args, settings.check_mode)?;
        refreshed.num_blinding_factors =
            prev_num_blinding_factors.or(refreshed.num_blinding_factors);
        refreshed.timestamp = prev_timestamp.or(refreshed.timestamp);

        *settings = refreshed;
    }

    settings.module_sizes = GraphModules::num_constraints_and_instances(
        model.graph.input_shapes()?,
        model.const_shapes(),
        model.graph.output_shapes()?,
        model.visibility.clone(),
    );

    sync_settings_execution_contract(settings);
    Ok(())
}

fn format_error_with_sources(err: &dyn std::error::Error) -> String {
    let mut parts = vec![err.to_string()];
    let mut current = err.source();
    while let Some(src) = current {
        parts.push(src.to_string());
        current = src.source();
    }
    parts.join(" | caused by: ")
}

impl GraphCircuit {
    pub fn settings(&self) -> &GraphSettings {
        &self.settings
    }

    /// Params passed to halo2 vk/pk read() calls (must match Circuit::Params).
    pub fn params(&self) -> GraphSettings {
        self.settings.clone()
    }

    /// Compute and set the minimum `logrows` (k) required for the current circuit/settings.
    ///
    /// This mutates:
    /// - `self.settings.run_args.lookup_range` (expanded by `lookup_safety_margin`)
    /// - `self.settings.run_args.logrows`
    ///
    /// It also keeps the thread-local `GLOBAL_SETTINGS` in sync, since `GraphCircuit::configure`
    /// reads sizing parameters from there.
    pub fn calc_min_logrows(
        &mut self,
        observed_lookup_range: Range,
        observed_max_range_abs: IntegerRep,
        max_logrows: Option<u32>,
        lookup_safety_margin: f64,
    ) -> Result<(), GraphError> {
        let has_lookups = !self.settings.required_lookups.is_empty();

        // Normalize (min,max).
        let (mut min_l, mut max_l) = observed_lookup_range;
        if min_l > max_l {
            std::mem::swap(&mut min_l, &mut max_l);
        }

        // Expand lookup range by safety margin (symmetric expansion around the observed span).
        //
        // IMPORTANT: If the circuit does not require any lookup tables at all, ignore the observed
        // lookup range entirely. In practice, dummy/forward passes may leave sentinel min/max values
        // (i128::MAX / i128::MIN) when no lookup op was executed, which can otherwise cause
        // overflow/wrapping and make calibration fail.
        if !has_lookups {
            min_l = 0;
            max_l = 0;
            self.settings.run_args.lookup_range = (0, 0);
        } else {
            // Clamp safety margin to >= 1.0. (Values < 1.0 would shrink ranges which is unsafe.)
            let safety = if lookup_safety_margin.is_finite() && lookup_safety_margin >= 1.0 {
                lookup_safety_margin
            } else {
                1.0
            };

            // Use overflow-safe unsigned arithmetic for span computations.
            let span_u: u128 = max_l.abs_diff(min_l);

            let expanded_span_u: u128 = if span_u == 0 || safety == 1.0 {
                span_u
            } else {
                let expanded_f = (span_u as f64) * safety;
                if !expanded_f.is_finite() {
                    u128::MAX
                } else {
                    // Note: `u128::MAX as f64` is lossy but sufficient as an upper clamp.
                    let max_f = u128::MAX as f64;
                    expanded_f.ceil().min(max_f) as u128
                }
            };

            let extra_u: u128 = expanded_span_u.saturating_sub(span_u);
            let extra_lo_u: u128 = extra_u / 2;
            let extra_hi_u: u128 = extra_u - extra_lo_u;

            let extra_lo_i: IntegerRep = i128::try_from(extra_lo_u).unwrap_or(i128::MAX);
            let extra_hi_i: IntegerRep = i128::try_from(extra_hi_u).unwrap_or(i128::MAX);

            min_l = min_l.saturating_sub(extra_lo_i);
            max_l = max_l.saturating_add(extra_hi_i);

            self.settings.run_args.lookup_range = (min_l, max_l);
        }

        // Conservative range-check expansion based on the observed maximum absolute value.
        // (This is a minimal, data-driven calibration hook.)
        let max_abs_u: u128 = observed_max_range_abs
            .checked_abs()
            .map(|x| x as u128)
            .unwrap_or((IntegerRep::MAX as u128) + 1); // abs(i128::MIN)
        if max_abs_u > 0 {
            let bound_u: u128 = max_abs_u.saturating_mul(super::RANGE_MULTIPLIER as u128);
            let bound_i: IntegerRep = if bound_u > (IntegerRep::MAX as u128) {
                IntegerRep::MAX
            } else {
                bound_u as IntegerRep
            };

            let rc = (-bound_i, bound_i);
            if !self.settings.required_range_checks.contains(&rc) {
                self.settings.required_range_checks.push(rc);
            }
        }

        // Base k requirement from the model & aux columns.
        let model_rows = self.settings.num_rows + super::RESERVED_BLINDING_ROWS;
        let mut required_k = (model_rows as f64).log2().ceil() as u32;
        required_k = required_k.max(self.settings.module_constraint_logrows_with_blinding());
        required_k = required_k.max(self.settings.log2_total_instances_with_blinding());
        required_k = required_k.max(
            self.settings
                .dynamic_lookup_and_shuffle_logrows_with_blinding(),
        );
        required_k = required_k.max(self.settings.range_check_log_rows_with_blinding());
        required_k = required_k.max(super::MIN_LOGROWS);

        let mut max_k = super::MAX_PUBLIC_SRS;
        if let Some(user_max) = max_logrows {
            max_k = max_k.min(user_max);
        }

        if required_k > max_k {
            return Err(GraphError::ExtendedKTooLarge(required_k));
        }

        // If there are no lookup tables, we can skip the lookup-range sizing logic entirely.
        if !has_lookups {
            self.settings.run_args.logrows = required_k;
            super::GLOBAL_SETTINGS.with(|gs| {
                *gs.borrow_mut() = Some(self.settings.clone());
            });
            return Ok(());
        }

        // Enforce the lookup-table column limit by finding the smallest k such that
        // `num_cols_required(range_len, col_size) <= MAX_NUM_LOOKUP_COLS`.
        //
        // NOTE: `range_len` is the *difference* (max-min). The number of represented values is
        // `range_len + 1`, hence the `+ 1` column accounting below.
        let range_len_u: u128 = max_l.abs_diff(min_l);

        for k in required_k..=max_k {
            let col_size_u: u128 =
                (1u128 << (k as u32)).saturating_sub(super::RESERVED_BLINDING_ROWS as u128);
            if col_size_u == 0 {
                continue;
            }

            let cols_needed_u: u128 = (range_len_u / col_size_u) + 1;
            if cols_needed_u <= super::MAX_NUM_LOOKUP_COLS as u128 {
                self.settings.run_args.logrows = k;

                super::GLOBAL_SETTINGS.with(|gs| {
                    *gs.borrow_mut() = Some(self.settings.clone());
                });
                return Ok(());
            }
        }

        let range_len_usize = usize::try_from(range_len_u).unwrap_or(usize::MAX);
        Err(GraphError::LookupRangeTooLarge(range_len_usize))
    }

    pub fn save(&self, path: std::path::PathBuf) -> Result<(), GraphError> {
        let f = File::create(&path).map_err(|e| {
            GraphError::ReadWriteFileError(path.display().to_string(), e.to_string())
        })?;
        let writer = BufWriter::new(f);
        bincode::serialize_into(writer, self)?;
        Ok(())
    }

    pub fn load(path: std::path::PathBuf) -> Result<Self, GraphError> {
        let f = File::open(&path).map_err(|e| {
            GraphError::ReadWriteFileError(path.display().to_string(), e.to_string())
        })?;
        let reader = BufReader::new(f);
        let mut c: GraphCircuit = bincode::deserialize_from(reader)?;

        normalize_settings_for_model(&mut c.settings, &c.model)?;

        // Keep GLOBAL_SETTINGS in sync for subsequent `Circuit::configure` calls.
        super::GLOBAL_SETTINGS.with(|gs| {
            *gs.borrow_mut() = Some(c.settings.clone());
        });

        Ok(c)
    }

    pub fn from_run_args(
        run_args: &crate::RunArgs,
        model_path: &std::path::Path,
    ) -> Result<Self, GraphError> {
        let model = Model::from_run_args(run_args, model_path)?;

        let mut settings = model.gen_params(run_args, run_args.check_mode)?;
        normalize_settings_for_model(&mut settings, &model)?;

        // Keep GLOBAL_SETTINGS in sync for subsequent `Circuit::configure` calls.
        super::GLOBAL_SETTINGS.with(|gs| {
            *gs.borrow_mut() = Some(settings.clone());
        });

        Ok(Self {
            model,
            settings,
            witness: None,
        })
    }

    pub fn from_settings(
        settings: &GraphSettings,
        model_path: &std::path::Path,
        check_mode: CheckMode,
    ) -> Result<Self, GraphError> {
        let mut settings = settings.clone();
        settings.check_mode = check_mode;

        let model = Model::from_run_args(&settings.run_args, model_path)?;
        normalize_settings_for_model(&mut settings, &model)?;

        // Keep GLOBAL_SETTINGS in sync for subsequent `Circuit::configure` calls.
        super::GLOBAL_SETTINGS.with(|gs| {
            *gs.borrow_mut() = Some(settings.clone());
        });

        Ok(Self {
            model,
            settings,
            witness: None,
        })
    }

    pub fn load_graph_witness(&mut self, witness: &GraphWitness) -> Result<(), GraphError> {
        self.witness = Some(witness.clone());
        Ok(())
    }

    pub fn load_graph_input(&self, data: &GraphData) -> Result<Vec<Tensor<Fp>>, GraphError> {
        let input_shapes = self.model.graph.input_shapes()?;

        if !data.input_data.is_empty() && data.input_data.len() != input_shapes.len() {
            return Err(GraphError::InvalidDims(
                0,
                format!(
                    "input_data len {} does not match model inputs {}",
                    data.input_data.len(),
                    input_shapes.len()
                ),
            ));
        }

        let mut out = Vec::with_capacity(input_shapes.len());

        for (i, shape) in input_shapes.iter().enumerate() {
            let flat: Vec<FileSourceInner> =
                data.input_data.get(i).cloned().unwrap_or_else(|| vec![]);

            let scale = self
                .settings
                .model_input_scales
                .get(i)
                .cloned()
                .unwrap_or(self.settings.run_args.input_scale);

            let vals = flat.iter().map(|x| x.to_field(scale)).collect_vec();
            let mut t = Tensor::<Fp>::new(Some(&vals), shape)?;
            t.set_scale(scale);
            t.set_visibility(&self.settings.run_args.input_visibility);
            out.push(t);
        }

        Ok(out)
    }

    pub fn forward<Scheme: CommitmentScheme<Scalar = Fp, Curve = G1Affine>>(
        &mut self,
        inputs: &mut Vec<Tensor<Fp>>,
        vk: Option<&VerifyingKey<G1Affine>>,
        srs: Option<&Scheme::ParamsProver>,
        region_settings: crate::circuit::region::RegionSettings,
    ) -> Result<GraphWitness, GraphError> {
        let res = self
            .model
            .forward(inputs, &self.settings.run_args, region_settings)?;

        let witness_inputs = inputs
            .iter()
            .map(|t| t.clone().into_iter().collect())
            .collect();
        let witness_outputs: Vec<Vec<Fp>> = res
            .outputs
            .iter()
            .map(|t| t.clone().into_iter().collect())
            .collect();

        let params = self.model.get_all_params();

        let processed_inputs: Option<ModuleForwardResult> =
            if self.model.visibility.input.requires_processing() {
                Some(GraphModules::forward::<Scheme>(
                    &inputs.iter().cloned().collect_vec(),
                    &self.model.visibility.input,
                    vk,
                    srs,
                )?)
            } else {
                None
            };

        let processed_params: Option<ModuleForwardResult> =
            if self.model.visibility.params.requires_processing() {
                Some(GraphModules::forward::<Scheme>(
                    &params,
                    &self.model.visibility.params,
                    vk,
                    srs,
                )?)
            } else {
                None
            };

        let processed_outputs: Option<ModuleForwardResult> =
            if self.model.visibility.output.requires_processing() {
                Some(GraphModules::forward::<Scheme>(
                    &res.outputs,
                    &self.model.visibility.output,
                    vk,
                    srs,
                )?)
            } else {
                None
            };

        Ok(GraphWitness {
            inputs: witness_inputs,
            outputs: witness_outputs,
            pretty_elements: None,
            processed_inputs,
            processed_params,
            processed_outputs,
            max_lookup_inputs: res.max_lookup_inputs,
            min_lookup_inputs: res.min_lookup_inputs,
            max_range_size: res.max_range_size,
            version: Some(crate::version().to_string()),
        })
    }

    pub fn prepare_public_inputs(&self, witness: &GraphWitness) -> Result<Vec<Fp>, GraphError> {
        let mut instances: Vec<Fp> = vec![];

        // 1) module public commitments/hashes FIRST (layout uses instance_offset from 0)
        if self.model.visibility.input.is_hashed_public() {
            if let Some(proc) = &witness.processed_inputs {
                if let Some(h) = &proc.poseidon_hash {
                    instances.extend(h.iter().cloned());
                }
            }
        }
        if self.model.visibility.params.is_hashed_public() {
            if let Some(proc) = &witness.processed_params {
                if let Some(h) = &proc.poseidon_hash {
                    instances.extend(h.iter().cloned());
                }
            }
        }
        if self.model.visibility.output.is_hashed_public() {
            if let Some(proc) = &witness.processed_outputs {
                if let Some(h) = &proc.poseidon_hash {
                    instances.extend(h.iter().cloned());
                }
            }
        }

        // 2) model public inputs/outputs (same order as Model::instance_shapes()).
        if self.model.visibility.input.is_public() {
            for v in &witness.inputs {
                instances.extend(v.iter().cloned());
            }
        }
        if self.model.visibility.output.is_public() {
            for v in &witness.outputs {
                instances.extend(v.iter().cloned());
            }
        }

        Ok(instances)
    }

    pub fn pretty_public_inputs(
        &self,
        witness: &GraphWitness,
    ) -> Result<crate::pfsys::PrettyElements, GraphError> {
        let mut w = witness.clone();
        w.generate_rescaled_elements(
            self.settings.model_input_scales.clone(),
            self.settings.model_output_scales.clone(),
            self.model.visibility.clone(),
        );
        Ok(w.pretty_elements.clone().unwrap_or_default())
    }
}

impl Circuit<Fp> for GraphCircuit {
    type Config = GraphCircuitConfig;
    type FloorPlanner = SimpleFloorPlanner;
    type Params = GraphSettings;

    fn without_witnesses(&self) -> Self {
        let mut c = self.clone();
        c.witness = None;
        c
    }

    fn configure(meta: &mut ConstraintSystem<Fp>) -> Self::Config {
        // GraphCircuit config depends on GraphSettings; we thread it through a
        // thread-local (GLOBAL_SETTINGS) because Halo2's Circuit::configure has no `&self`.
        let params = super::GLOBAL_SETTINGS
            .with(|gs| gs.borrow().clone())
            .unwrap_or_else(|| {
                panic!(
                "GLOBAL_SETTINGS not set. Load settings via GraphSettings::load() or construct \
                 GraphCircuit via GraphCircuit::load()/from_run_args()/from_settings() before \
                 invoking Halo2 keygen/prove/verify."
            )
            });

        // Configure modules (poseidon/polycommit) if visibility requires it.
        let module_sizes: ModuleSizes = params.module_sizes.clone();
        let mut module_configs =
            ModuleConfigs::from_visibility(meta, &module_sizes, params.run_args.logrows as usize);

        let visibility =
            VarVisibility::from_args(&params.run_args).unwrap_or_else(|_| VarVisibility::default());
        module_configs.configure_complex_modules(meta, &visibility, &module_sizes);

        // Configure model vars + base gates.
        let mut vars = super::vars::ModelVars::<Fp>::new(meta, &params);

        // If model needs instances OR modules created an instance column, bind them.
        if visibility.input.is_public()
            || visibility.output.is_public()
            || module_configs.instance.is_some()
        {
            vars.instantiate_instance(
                meta,
                params.model_instance_shapes.clone(),
                0,
                module_configs.instance,
            );
        }

        let base = Model::configure(meta, &vars, &params)
            .unwrap_or_else(|e| panic!("Model::configure failed: {e}"));

        GraphCircuitConfig {
            model: ModelConfig { base, vars },
            modules: module_configs,
        }
    }

    fn synthesize(
        &self,
        config: Self::Config,
        mut layouter: impl Layouter<Fp>,
    ) -> Result<(), Error> {
        let GraphCircuitConfig {
            model: model_cfg,
            modules: _modules_cfg,
        } = config;

        // Build per-run vars state (idx/offset changes during layout).
        let mut vars = model_cfg.vars.clone();

        // If we later wire module layouts in, they should consume instance rows first and then
        // set vars.set_initial_instance_offset(instance_offset). For now we keep 0.
        vars.set_initial_instance_offset(0);

        let mut constants: ConstantsMap<Fp> = Default::default();

        // Prepare input ValTensors.
        let input_shapes = self
            .model
            .graph
            .input_shapes()
            .map_err(|_| Error::Synthesis)?;

        let witness_opt = self.witness.as_ref();

        let mut inputs_vt: Vec<ValTensor<Fp>> = vec![];
        for (i, shape) in input_shapes.iter().enumerate() {
            let len = shape.iter().product::<usize>();

            let vt = if let Some(w) = witness_opt {
                let flat = w
                    .inputs
                    .get(i)
                    .cloned()
                    .unwrap_or_else(|| vec![Fp::ZERO; len]);
                let mut t = Tensor::<Fp>::new(Some(&flat), shape).map_err(|_| Error::Synthesis)?;
                t.set_visibility(&self.settings.run_args.input_visibility);
                ValTensor::try_from(t).map_err(|_| Error::Synthesis)?
            } else {
                let mut t: Tensor<Value<Fp>> = (0..len).map(|_| Value::unknown()).collect();
                t.reshape(shape).map_err(|_| Error::Synthesis)?;
                ValTensor::from(t)
            };

            inputs_vt.push(vt);
        }

        // Prepare witnessed outputs only if outputs are FIXED.
        let mut witnessed_outputs: Vec<ValTensor<Fp>> = vec![];
        if self.settings.run_args.output_visibility.is_fixed() {
            let out_shapes = self
                .model
                .graph
                .output_shapes()
                .map_err(|_| Error::Synthesis)?;
            for (i, shape) in out_shapes.iter().enumerate() {
                let len = shape.iter().product::<usize>();

                let vt = if let Some(w) = witness_opt {
                    let flat = w
                        .outputs
                        .get(i)
                        .cloned()
                        .unwrap_or_else(|| vec![Fp::ZERO; len]);
                    let mut t =
                        Tensor::<Fp>::new(Some(&flat), shape).map_err(|_| Error::Synthesis)?;
                    t.set_visibility(&self.settings.run_args.output_visibility);
                    ValTensor::try_from(t).map_err(|_| Error::Synthesis)?
                } else {
                    let mut t: Tensor<Value<Fp>> = (0..len).map(|_| Value::unknown()).collect();
                    t.reshape(shape).map_err(|_| Error::Synthesis)?;
                    ValTensor::from(t)
                };

                witnessed_outputs.push(vt);
            }
        }

        // Layout the model.
        self.model
            .layout(
                model_cfg,
                &mut layouter,
                &self.settings.run_args,
                &inputs_vt,
                &mut vars,
                &witnessed_outputs,
                &mut constants,
            )
            .map_err(|e| {
                let detailed = format_error_with_sources(&e);
                log::error!("GraphCircuit synthesize failed: {detailed}");
                eprintln!("GraphCircuit synthesize failed: {detailed}");
                Error::Synthesis
            })?;

        Ok(())
    }
}
