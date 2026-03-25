use super::errors::GraphError;
use super::modules::ModuleSizes;
use crate::circuit::lookup::LookupOp;
use crate::circuit::table::{Range, RESERVED_BLINDING_ROWS_PAD};
use crate::circuit::{CheckMode, InputType};
use crate::fieldutils::IntegerRep;
use crate::{ProbabilisticSettings, RunArgs, EZKL_BUF_CAPACITY};

use halo2curves::bn256::{self};
use halo2curves::ff::PrimeField;
use log::{error, warn};
use serde::{Deserialize, Serialize};
use std::cell::RefCell;
use std::collections::HashMap;

/// The safety factor for the range of the lookup table.
pub const RANGE_MULTIPLIER: IntegerRep = 2;

/// The maximum number of columns in a lookup table.
pub const MAX_NUM_LOOKUP_COLS: usize = 12;

/// 26
pub const MAX_PUBLIC_SRS: u32 = bn256::Fr::S - 2;

/// Max representation of a lookup table input
pub const MAX_LOOKUP_ABS: IntegerRep =
    (MAX_NUM_LOOKUP_COLS as IntegerRep) * 2_i128.pow(MAX_PUBLIC_SRS);

///
pub const ASSUMED_BLINDING_FACTORS: usize = 5;
/// The minimum number of rows in the grid
pub const MIN_LOGROWS: u32 = 6;

/// 26 - (a few rows)
pub const RESERVED_BLINDING_ROWS: usize = ASSUMED_BLINDING_FACTORS + RESERVED_BLINDING_ROWS_PAD;

thread_local!(
    /// This is a global variable that holds the settings for the graph
    /// This is used to pass settings to the layouter and other parts of the circuit without needing to heavily modify the Halo2 API in a new fork
    pub static GLOBAL_SETTINGS: RefCell<Option<GraphSettings>> = const { RefCell::new(None) }
);

/// Execution mode of the circuit: exact or probabilistic.
///
/// This is part of the "execution contract" and must be auditable from settings.json.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, PartialOrd, Default)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionMode {
    #[default]
    #[serde(alias = "Exact", alias = "EXACT")]
    Exact,
    #[serde(
        alias = "Probabilistic",
        alias = "PROBABILISTIC",
        alias = "probabilistic"
    )]
    Probabilistic,
}

/// Operators eligible for probabilistic execution (Freivalds-style checks).
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum ProbOp {
    #[serde(
        rename = "matmul",
        alias = "mat_mul",
        alias = "MatMul",
        alias = "MATMUL"
    )]
    MatMul,
    #[serde(rename = "gemm", alias = "Gemm", alias = "GEMM")]
    Gemm,
    #[serde(rename = "conv", alias = "Conv", alias = "CONV")]
    Conv,
}

/// A transparent "set" of probabilistic operators for settings.json.
///
/// This is intentionally lightweight for Step 1; later steps can evolve this into
/// a per-op config map without changing the top-level field name (`prob_ops`).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(transparent)]
pub struct ProbOps(pub Vec<ProbOp>);

impl Default for ProbOps {
    fn default() -> Self {
        // Default policy: if `execution_mode` is set to probabilistic, apply to the main
        // heavy linear operators by default.
        Self(vec![ProbOp::MatMul, ProbOp::Gemm, ProbOp::Conv])
    }
}

impl ProbOps {
    /// Returns true if `op` is enabled in this set.
    pub fn contains(&self, op: ProbOp) -> bool {
        self.0.contains(&op)
    }

    /// Insert an op if it is not already present. Returns true if the set changed.
    pub fn insert(&mut self, op: ProbOp) -> bool {
        if self.contains(op) {
            false
        } else {
            self.0.push(op);
            true
        }
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
/// Parameters for dynamic lookups
/// serde should flatten this struct
pub struct DynamicLookupParams {
    /// total dynamic column size
    pub total_dynamic_col_size: usize,
    /// max dynamic column input length
    pub max_dynamic_input_len: usize,
    /// number of dynamic lookups
    pub num_dynamic_lookups: usize,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
/// Parameters for shuffle operations
pub struct ShuffleParams {
    /// number of shuffles
    pub num_shuffles: usize,
    /// total shuffle column size
    pub total_shuffle_col_size: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
/// Parameters for einsum operations
pub struct EinsumParams {
    /// einsum equations
    pub equations: Vec<(String, HashMap<char, usize>)>,
    /// total einsum column size
    pub total_einsum_col_size: usize,
}

/// model parameters
#[derive(Clone, Debug, PartialEq)]
pub struct GraphSettings {
    /// run args
    pub run_args: RunArgs,

    /// execution mode (exact vs probabilistic)
    ///
    /// NOTE: This is duplicated (mirrored) from `run_args.execution_mode` so it is easy to audit
    /// at the top-level of settings.json.
    pub execution_mode: ExecutionMode,

    /// Number of Freivalds repetitions (soundness error per check ≤ 2^-k for classic 0/1 Freivalds).
    ///
    /// NOTE: This is duplicated (mirrored) from both:
    /// * `run_args.prob_k` and
    /// * `probabilistic_settings.k_repetitions`
    /// so it is easy to audit at the top-level of settings.json.
    pub prob_k: u32,

    /// Which ops should use probabilistic verification (Freivalds-style checks).
    ///
    /// NOTE: This is duplicated (mirrored) from `run_args.prob_ops` so it is easy to audit
    /// at the top-level of settings.json.
    pub prob_ops: ProbOps,

    /// probabilistic execution settings (e.g. Freivalds repetitions, seed mode)
    ///
    /// NOTE: This remains as a nested object for future expansion (seed mode, etc).
    pub probabilistic_settings: ProbabilisticSettings,

    /// the potential number of rows used by the circuit
    pub num_rows: usize,
    /// total linear coordinate of assignments
    pub total_assignments: usize,
    /// total const size
    pub total_const_size: usize,
    /// dynamic lookup parameters, flattened for backwards compatibility, serialize and deserialize flattened for backwards compatibility
    pub dynamic_lookup_params: DynamicLookupParams,
    /// shuffle parameters, flattened for backwards compatibility
    pub shuffle_params: ShuffleParams,
    /// einsum parameters
    pub einsum_params: EinsumParams,
    /// the shape of public inputs to the model (in order of appearance)
    pub model_instance_shapes: Vec<Vec<usize>>,
    /// model output scales
    pub model_output_scales: Vec<crate::Scale>,
    /// model input scales
    pub model_input_scales: Vec<crate::Scale>,
    /// the of instance cells used by modules
    pub module_sizes: ModuleSizes,
    /// required_lookups
    pub required_lookups: Vec<LookupOp>,
    /// required range_checks
    pub required_range_checks: Vec<Range>,
    /// check mode
    pub check_mode: CheckMode,
    /// ezkl version used
    pub version: String,
    /// num blinding factors
    pub num_blinding_factors: Option<usize>,
    /// unix time timestamp
    pub timestamp: Option<u128>,
    /// Model inputs types (if any)
    pub input_types: Option<Vec<InputType>>,
    /// Model outputs types (if any)
    pub output_types: Option<Vec<InputType>>,
}

impl Default for GraphSettings {
    fn default() -> Self {
        let mut run_args = RunArgs::default();

        // Canonical top-level probabilistic config defaults.
        let probabilistic_settings = ProbabilisticSettings::default();
        let execution_mode = ExecutionMode::default();
        let prob_k = probabilistic_settings.k_repetitions;
        let prob_ops = ProbOps::default();

        // Mirror top-level config into run_args for backwards compatibility with existing codepaths.
        run_args.execution_mode = execution_mode;
        run_args.prob_k = prob_k as _;
        run_args.prob_seed_mode = probabilistic_settings.seed_mode;
        run_args.prob_ops = prob_ops.clone();

        Self {
            run_args,
            execution_mode,
            prob_k,
            prob_ops,
            probabilistic_settings,
            num_rows: 0,
            total_assignments: 0,
            total_const_size: 0,
            dynamic_lookup_params: DynamicLookupParams::default(),
            shuffle_params: ShuffleParams::default(),
            einsum_params: EinsumParams::default(),
            model_instance_shapes: vec![],
            model_output_scales: vec![],
            model_input_scales: vec![],
            module_sizes: ModuleSizes::default(),
            required_lookups: vec![],
            required_range_checks: vec![],
            check_mode: CheckMode::default(),
            version: String::new(),
            num_blinding_factors: None,
            timestamp: None,
            input_types: None,
            output_types: None,
        }
    }
}

impl Serialize for GraphSettings {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        // Always serialize a RunArgs snapshot that is consistent with the top-level config.
        let mut run_args = self.run_args.clone();
        run_args.execution_mode = self.execution_mode;
        run_args.prob_k = self.prob_k as _;
        run_args.prob_seed_mode = self.probabilistic_settings.seed_mode;
        run_args.prob_ops = self.prob_ops.clone();
        run_args.check_mode = self.check_mode;

        if serializer.is_human_readable() {
            // JSON format - use flattened fields for backwards compatibility
            use serde::ser::SerializeStruct;

            // NOTE: include the "execution contract" fields at the top-level so that
            // the security/soundness contract is easy to audit from settings.json.
            let mut state = serializer.serialize_struct("GraphSettings", 26)?;
            state.serialize_field("run_args", &run_args)?;
            state.serialize_field("execution_mode", &self.execution_mode)?;
            state.serialize_field("prob_k", &self.prob_k)?;
            state.serialize_field("prob_ops", &self.prob_ops)?;
            state.serialize_field("probabilistic_settings", &self.probabilistic_settings)?;
            state.serialize_field("num_rows", &self.num_rows)?;
            state.serialize_field("total_assignments", &self.total_assignments)?;
            state.serialize_field("total_const_size", &self.total_const_size)?;

            // Flatten DynamicLookupParams fields
            state.serialize_field(
                "total_dynamic_col_size",
                &self.dynamic_lookup_params.total_dynamic_col_size,
            )?;
            state.serialize_field(
                "max_dynamic_input_len",
                &self.dynamic_lookup_params.max_dynamic_input_len,
            )?;
            state.serialize_field(
                "num_dynamic_lookups",
                &self.dynamic_lookup_params.num_dynamic_lookups,
            )?;

            // Flatten ShuffleParams fields
            state.serialize_field("num_shuffles", &self.shuffle_params.num_shuffles)?;
            state.serialize_field(
                "total_shuffle_col_size",
                &self.shuffle_params.total_shuffle_col_size,
            )?;

            // Serialize EinsumParams
            state.serialize_field("einsum_params", &self.einsum_params)?;

            state.serialize_field("model_instance_shapes", &self.model_instance_shapes)?;
            state.serialize_field("model_output_scales", &self.model_output_scales)?;
            state.serialize_field("model_input_scales", &self.model_input_scales)?;
            state.serialize_field("module_sizes", &self.module_sizes)?;
            state.serialize_field("required_lookups", &self.required_lookups)?;
            state.serialize_field("required_range_checks", &self.required_range_checks)?;
            state.serialize_field("check_mode", &self.check_mode)?;
            state.serialize_field("version", &self.version)?;
            state.serialize_field("num_blinding_factors", &self.num_blinding_factors)?;
            state.serialize_field("timestamp", &self.timestamp)?;
            state.serialize_field("input_types", &self.input_types)?;
            state.serialize_field("output_types", &self.output_types)?;
            state.end()
        } else {
            // Binary format (bincode) - use nested struct format
            // NOTE: keep the bincode tuple format unchanged for compatibility.
            use serde::ser::SerializeTuple;
            let mut state = serializer.serialize_tuple(19)?;
            state.serialize_element(&run_args)?;
            state.serialize_element(&self.num_rows)?;
            state.serialize_element(&self.total_assignments)?;
            state.serialize_element(&self.total_const_size)?;
            state.serialize_element(&self.dynamic_lookup_params)?;
            state.serialize_element(&self.shuffle_params)?;
            state.serialize_element(&self.einsum_params)?;
            state.serialize_element(&self.model_instance_shapes)?;
            state.serialize_element(&self.model_output_scales)?;
            state.serialize_element(&self.model_input_scales)?;
            state.serialize_element(&self.module_sizes)?;
            state.serialize_element(&self.required_lookups)?;
            state.serialize_element(&self.required_range_checks)?;
            state.serialize_element(&self.check_mode)?;
            state.serialize_element(&self.version)?;
            state.serialize_element(&self.num_blinding_factors)?;
            state.serialize_element(&self.timestamp)?;
            state.serialize_element(&self.input_types)?;
            state.serialize_element(&self.output_types)?;
            state.end()
        }
    }
}

impl<'de> Deserialize<'de> for GraphSettings {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        use serde::de::{self, MapAccess, Visitor};
        use std::fmt;

        #[derive(Deserialize)]
        #[serde(field_identifier, rename_all = "snake_case")]
        enum Field {
            RunArgs,

            // Step 1: top-level config for execution contract.
            ExecutionMode,
            ProbK,
            ProbOps,

            // Nested probabilistic settings (seed mode + future expansion).
            ProbabilisticSettings,

            NumRows,
            TotalAssignments,
            TotalConstSize,
            // Flattened DynamicLookupParams fields
            TotalDynamicColSize,
            MaxDynamicInputLen,
            NumDynamicLookups,
            // Flattened ShuffleParams fields
            NumShuffles,
            TotalShuffleColSize,
            // EinsumParams field
            EinsumParams,
            ModelInstanceShapes,
            ModelOutputScales,
            ModelInputScales,
            ModuleSizes,
            RequiredLookups,
            RequiredRangeChecks,
            CheckMode,
            Version,
            NumBlindingFactors,
            Timestamp,
            InputTypes,
            OutputTypes,
            // Legacy nested struct fields for backwards compatibility
            DynamicLookupParams,
            ShuffleParams,
        }

        struct GraphSettingsVisitor;

        impl<'de> Visitor<'de> for GraphSettingsVisitor {
            type Value = GraphSettings;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("struct GraphSettings")
            }

            fn visit_map<V>(self, mut map: V) -> Result<GraphSettings, V::Error>
            where
                V: MapAccess<'de>,
            {
                let mut run_args: Option<RunArgs> = None;

                // Step 1: new top-level config.
                let mut execution_mode: Option<ExecutionMode> = None;
                let mut prob_k: Option<u32> = None;
                let mut prob_ops: Option<ProbOps> = None;
                let mut probabilistic_settings: Option<ProbabilisticSettings> = None;

                let mut num_rows = None;
                let mut total_assignments = None;
                let mut total_const_size = None;
                let mut total_dynamic_col_size = None;
                let mut max_dynamic_input_len = None;
                let mut num_dynamic_lookups = None;
                let mut num_shuffles = None;
                let mut total_shuffle_col_size = None;
                let mut einsum_params = None;
                let mut model_instance_shapes = None;
                let mut model_output_scales = None;
                let mut model_input_scales = None;
                let mut module_sizes = None;
                let mut required_lookups = None;
                let mut required_range_checks = None;
                let mut check_mode = None;
                let mut version = None;
                let mut num_blinding_factors = None;
                let mut timestamp = None;
                let mut input_types = None;
                let mut output_types = None;

                while let Some(key) = map.next_key()? {
                    match key {
                        Field::RunArgs => {
                            if run_args.is_some() {
                                return Err(de::Error::duplicate_field("run_args"));
                            }
                            run_args = Some(map.next_value()?);
                        }
                        Field::ExecutionMode => {
                            if execution_mode.is_some() {
                                return Err(de::Error::duplicate_field("execution_mode"));
                            }
                            execution_mode = Some(map.next_value()?);
                        }
                        Field::ProbK => {
                            if prob_k.is_some() {
                                return Err(de::Error::duplicate_field("prob_k"));
                            }
                            prob_k = Some(map.next_value()?);
                        }
                        Field::ProbOps => {
                            if prob_ops.is_some() {
                                return Err(de::Error::duplicate_field("prob_ops"));
                            }
                            prob_ops = Some(map.next_value()?);
                        }
                        Field::ProbabilisticSettings => {
                            if probabilistic_settings.is_some() {
                                return Err(de::Error::duplicate_field("probabilistic_settings"));
                            }
                            probabilistic_settings = Some(map.next_value()?);
                        }
                        Field::NumRows => {
                            if num_rows.is_some() {
                                return Err(de::Error::duplicate_field("num_rows"));
                            }
                            num_rows = Some(map.next_value()?);
                        }
                        Field::TotalAssignments => {
                            if total_assignments.is_some() {
                                return Err(de::Error::duplicate_field("total_assignments"));
                            }
                            total_assignments = Some(map.next_value()?);
                        }
                        Field::TotalConstSize => {
                            if total_const_size.is_some() {
                                return Err(de::Error::duplicate_field("total_const_size"));
                            }
                            total_const_size = Some(map.next_value()?);
                        }
                        Field::TotalDynamicColSize => {
                            if total_dynamic_col_size.is_some() {
                                return Err(de::Error::duplicate_field("total_dynamic_col_size"));
                            }
                            total_dynamic_col_size = Some(map.next_value()?);
                        }
                        Field::MaxDynamicInputLen => {
                            if max_dynamic_input_len.is_some() {
                                return Err(de::Error::duplicate_field("max_dynamic_input_len"));
                            }
                            max_dynamic_input_len = Some(map.next_value()?);
                        }
                        Field::NumDynamicLookups => {
                            if num_dynamic_lookups.is_some() {
                                return Err(de::Error::duplicate_field("num_dynamic_lookups"));
                            }
                            num_dynamic_lookups = Some(map.next_value()?);
                        }
                        Field::NumShuffles => {
                            if num_shuffles.is_some() {
                                return Err(de::Error::duplicate_field("num_shuffles"));
                            }
                            num_shuffles = Some(map.next_value()?);
                        }
                        Field::TotalShuffleColSize => {
                            if total_shuffle_col_size.is_some() {
                                return Err(de::Error::duplicate_field("total_shuffle_col_size"));
                            }
                            total_shuffle_col_size = Some(map.next_value()?);
                        }
                        Field::EinsumParams => {
                            if einsum_params.is_some() {
                                return Err(de::Error::duplicate_field("einsum_params"));
                            }
                            einsum_params = Some(map.next_value()?);
                        }
                        Field::ModelInstanceShapes => {
                            if model_instance_shapes.is_some() {
                                return Err(de::Error::duplicate_field("model_instance_shapes"));
                            }
                            model_instance_shapes = Some(map.next_value()?);
                        }
                        Field::ModelOutputScales => {
                            if model_output_scales.is_some() {
                                return Err(de::Error::duplicate_field("model_output_scales"));
                            }
                            model_output_scales = Some(map.next_value()?);
                        }
                        Field::ModelInputScales => {
                            if model_input_scales.is_some() {
                                return Err(de::Error::duplicate_field("model_input_scales"));
                            }
                            model_input_scales = Some(map.next_value()?);
                        }
                        Field::ModuleSizes => {
                            if module_sizes.is_some() {
                                return Err(de::Error::duplicate_field("module_sizes"));
                            }
                            module_sizes = Some(map.next_value()?);
                        }
                        Field::RequiredLookups => {
                            if required_lookups.is_some() {
                                return Err(de::Error::duplicate_field("required_lookups"));
                            }
                            required_lookups = Some(map.next_value()?);
                        }
                        Field::RequiredRangeChecks => {
                            if required_range_checks.is_some() {
                                return Err(de::Error::duplicate_field("required_range_checks"));
                            }
                            required_range_checks = Some(map.next_value()?);
                        }
                        Field::CheckMode => {
                            if check_mode.is_some() {
                                return Err(de::Error::duplicate_field("check_mode"));
                            }
                            check_mode = Some(map.next_value()?);
                        }
                        Field::Version => {
                            if version.is_some() {
                                return Err(de::Error::duplicate_field("version"));
                            }
                            version = Some(map.next_value()?);
                        }
                        Field::NumBlindingFactors => {
                            if num_blinding_factors.is_some() {
                                return Err(de::Error::duplicate_field("num_blinding_factors"));
                            }
                            num_blinding_factors = map.next_value()?;
                        }
                        Field::Timestamp => {
                            if timestamp.is_some() {
                                return Err(de::Error::duplicate_field("timestamp"));
                            }
                            timestamp = Some(map.next_value()?);
                        }
                        Field::InputTypes => {
                            if input_types.is_some() {
                                return Err(de::Error::duplicate_field("input_types"));
                            }
                            input_types = map.next_value()?;
                        }
                        Field::OutputTypes => {
                            if output_types.is_some() {
                                return Err(de::Error::duplicate_field("output_types"));
                            }
                            output_types = map.next_value()?;
                        }
                        // Handle legacy nested struct fields for backwards compatibility
                        Field::DynamicLookupParams => {
                            let legacy_params: DynamicLookupParams = map.next_value()?;
                            if total_dynamic_col_size.is_none() {
                                total_dynamic_col_size = Some(legacy_params.total_dynamic_col_size);
                            }
                            if max_dynamic_input_len.is_none() {
                                max_dynamic_input_len = Some(legacy_params.max_dynamic_input_len);
                            }
                            if num_dynamic_lookups.is_none() {
                                num_dynamic_lookups = Some(legacy_params.num_dynamic_lookups);
                            }
                        }
                        Field::ShuffleParams => {
                            let legacy_params: ShuffleParams = map.next_value()?;
                            if num_shuffles.is_none() {
                                num_shuffles = Some(legacy_params.num_shuffles);
                            }
                            if total_shuffle_col_size.is_none() {
                                total_shuffle_col_size = Some(legacy_params.total_shuffle_col_size);
                            }
                        }
                    }
                }

                let mut run_args = run_args.ok_or_else(|| de::Error::missing_field("run_args"))?;

                // Determine execution mode:
                // * prefer top-level execution_mode
                // * else use run_args.execution_mode (backwards compatible)
                let execution_mode_final = execution_mode.unwrap_or(run_args.execution_mode);
                run_args.execution_mode = execution_mode_final;

                // Determine prob_k / seed mode:
                // Priority:
                // 1) top-level probabilistic_settings (new contract)
                // 2) top-level prob_k (Step 1 top-level mirror)
                // 3) run_args values (if present / non-default)
                // 4) otherwise defaults
                let mut probabilistic_settings_final = ProbabilisticSettings::default();
                if let Some(ps) = probabilistic_settings {
                    probabilistic_settings_final = ps;
                } else if let Some(k) = prob_k {
                    probabilistic_settings_final.k_repetitions = k;
                    probabilistic_settings_final.seed_mode = run_args.prob_seed_mode;
                } else {
                    // If RunArgs carries a non-default value, prefer it; otherwise keep the
                    // new default (k=40, public_seed).
                    if (run_args.prob_k as u32) != probabilistic_settings_final.k_repetitions {
                        probabilistic_settings_final.k_repetitions = run_args.prob_k as u32;
                    }
                    if run_args.prob_seed_mode != probabilistic_settings_final.seed_mode {
                        probabilistic_settings_final.seed_mode = run_args.prob_seed_mode;
                    }
                }

                let prob_k_final = probabilistic_settings_final.k_repetitions;

                // Determine prob_ops:
                // * prefer top-level prob_ops
                // * else use run_args.prob_ops
                // * else default
                let prob_ops_final = prob_ops.unwrap_or_else(|| run_args.prob_ops.clone());

                // Mirror into run_args for existing codepaths.
                run_args.prob_k = prob_k_final as _;
                run_args.prob_seed_mode = probabilistic_settings_final.seed_mode;
                run_args.prob_ops = prob_ops_final.clone();

                let num_rows = num_rows.ok_or_else(|| de::Error::missing_field("num_rows"))?;
                let total_assignments = total_assignments
                    .ok_or_else(|| de::Error::missing_field("total_assignments"))?;
                let total_const_size =
                    total_const_size.ok_or_else(|| de::Error::missing_field("total_const_size"))?;
                let model_instance_shapes = model_instance_shapes
                    .ok_or_else(|| de::Error::missing_field("model_instance_shapes"))?;
                let model_output_scales = model_output_scales
                    .ok_or_else(|| de::Error::missing_field("model_output_scales"))?;
                let model_input_scales = model_input_scales
                    .ok_or_else(|| de::Error::missing_field("model_input_scales"))?;
                let module_sizes =
                    module_sizes.ok_or_else(|| de::Error::missing_field("module_sizes"))?;
                let required_lookups =
                    required_lookups.ok_or_else(|| de::Error::missing_field("required_lookups"))?;
                let required_range_checks = required_range_checks
                    .ok_or_else(|| de::Error::missing_field("required_range_checks"))?;
                let check_mode =
                    check_mode.ok_or_else(|| de::Error::missing_field("check_mode"))?;
                let version = version.ok_or_else(|| de::Error::missing_field("version"))?;

                run_args.check_mode = check_mode;

                // Build the nested structs from flattened fields, with defaults if missing
                let dynamic_lookup_params = DynamicLookupParams {
                    total_dynamic_col_size: total_dynamic_col_size.unwrap_or_default(),
                    max_dynamic_input_len: max_dynamic_input_len.unwrap_or_default(),
                    num_dynamic_lookups: num_dynamic_lookups.unwrap_or_default(),
                };

                let shuffle_params = ShuffleParams {
                    num_shuffles: num_shuffles.unwrap_or_default(),
                    total_shuffle_col_size: total_shuffle_col_size.unwrap_or_default(),
                };

                Ok(GraphSettings {
                    run_args,
                    execution_mode: execution_mode_final,
                    prob_k: prob_k_final,
                    prob_ops: prob_ops_final,
                    probabilistic_settings: probabilistic_settings_final,
                    num_rows,
                    total_assignments,
                    total_const_size,
                    dynamic_lookup_params,
                    shuffle_params,
                    einsum_params: einsum_params.unwrap_or_default(),
                    model_instance_shapes,
                    model_output_scales,
                    model_input_scales,
                    module_sizes,
                    required_lookups,
                    required_range_checks,
                    check_mode,
                    version,
                    num_blinding_factors,
                    timestamp,
                    input_types,
                    output_types,
                })
            }

            fn visit_seq<V>(self, mut seq: V) -> Result<GraphSettings, V::Error>
            where
                V: serde::de::SeqAccess<'de>,
            {
                use serde::de::Error;

                // For bincode compatibility, deserialize in the same order as tuple serialization
                let mut run_args: RunArgs = seq
                    .next_element()?
                    .ok_or_else(|| Error::invalid_length(0, &self))?;
                let num_rows = seq
                    .next_element()?
                    .ok_or_else(|| Error::invalid_length(1, &self))?;
                let total_assignments = seq
                    .next_element()?
                    .ok_or_else(|| Error::invalid_length(2, &self))?;
                let total_const_size = seq
                    .next_element()?
                    .ok_or_else(|| Error::invalid_length(3, &self))?;
                let dynamic_lookup_params = seq
                    .next_element()?
                    .ok_or_else(|| Error::invalid_length(4, &self))?;
                let shuffle_params = seq
                    .next_element()?
                    .ok_or_else(|| Error::invalid_length(5, &self))?;
                let einsum_params = seq
                    .next_element()?
                    .ok_or_else(|| Error::invalid_length(6, &self))?;
                let model_instance_shapes = seq
                    .next_element()?
                    .ok_or_else(|| Error::invalid_length(7, &self))?;
                let model_output_scales = seq
                    .next_element()?
                    .ok_or_else(|| Error::invalid_length(8, &self))?;
                let model_input_scales = seq
                    .next_element()?
                    .ok_or_else(|| Error::invalid_length(9, &self))?;
                let module_sizes = seq
                    .next_element()?
                    .ok_or_else(|| Error::invalid_length(10, &self))?;
                let required_lookups = seq
                    .next_element()?
                    .ok_or_else(|| Error::invalid_length(11, &self))?;
                let required_range_checks = seq
                    .next_element()?
                    .ok_or_else(|| Error::invalid_length(12, &self))?;
                let check_mode = seq
                    .next_element()?
                    .ok_or_else(|| Error::invalid_length(13, &self))?;
                let version = seq
                    .next_element()?
                    .ok_or_else(|| Error::invalid_length(14, &self))?;
                let num_blinding_factors = seq
                    .next_element()?
                    .ok_or_else(|| Error::invalid_length(15, &self))?;
                let timestamp = seq
                    .next_element()?
                    .ok_or_else(|| Error::invalid_length(16, &self))?;
                let input_types = seq
                    .next_element()?
                    .ok_or_else(|| Error::invalid_length(17, &self))?;
                let output_types = seq
                    .next_element()?
                    .ok_or_else(|| Error::invalid_length(18, &self))?;

                // Derive the new top-level config from run_args (keeps bincode format unchanged).
                let execution_mode = run_args.execution_mode;

                // Derive prob_k from run_args.
                let prob_k = run_args.prob_k as u32;

                // Derive prob_ops from run_args.
                let prob_ops = run_args.prob_ops.clone();

                let mut probabilistic_settings = ProbabilisticSettings::default();
                // Prefer run_args if it differs from default.
                if prob_k != probabilistic_settings.k_repetitions {
                    probabilistic_settings.k_repetitions = prob_k;
                }
                if run_args.prob_seed_mode != probabilistic_settings.seed_mode {
                    probabilistic_settings.seed_mode = run_args.prob_seed_mode;
                }

                // Mirror top-level config back into run_args to ensure internal consistency.
                run_args.execution_mode = execution_mode;
                run_args.prob_k = prob_k as _;
                run_args.prob_seed_mode = probabilistic_settings.seed_mode;
                run_args.prob_ops = prob_ops.clone();
                run_args.check_mode = check_mode;

                Ok(GraphSettings {
                    run_args,
                    execution_mode,
                    prob_k,
                    prob_ops,
                    probabilistic_settings,
                    num_rows,
                    total_assignments,
                    total_const_size,
                    dynamic_lookup_params,
                    shuffle_params,
                    einsum_params,
                    model_instance_shapes,
                    model_output_scales,
                    model_input_scales,
                    module_sizes,
                    required_lookups,
                    required_range_checks,
                    check_mode,
                    version,
                    num_blinding_factors,
                    timestamp,
                    input_types,
                    output_types,
                })
            }
        }

        // Universal deserializer that works with both JSON (map) and bincode (tuple)
        if deserializer.is_human_readable() {
            // JSON format - use struct/map deserialization with flattened fields
            const FIELDS: &'static [&'static str] = &[
                "run_args",
                "execution_mode",
                "prob_k",
                "prob_ops",
                "probabilistic_settings",
                "num_rows",
                "total_assignments",
                "total_const_size",
                "total_dynamic_col_size",
                "max_dynamic_input_len",
                "num_dynamic_lookups",
                "num_shuffles",
                "total_shuffle_col_size",
                "einsum_params",
                "model_instance_shapes",
                "model_output_scales",
                "model_input_scales",
                "module_sizes",
                "required_lookups",
                "required_range_checks",
                "check_mode",
                "version",
                "num_blinding_factors",
                "timestamp",
                "input_types",
                "output_types",
                "dynamic_lookup_params",
                "shuffle_params",
            ];
            deserializer.deserialize_struct("GraphSettings", FIELDS, GraphSettingsVisitor)
        } else {
            // Binary format (bincode) - use tuple deserialization
            deserializer.deserialize_tuple(19, GraphSettingsVisitor)
        }
    }
}

impl GraphSettings {
    /// Calc the number of rows required for lookup tables
    pub fn lookup_log_rows(&self) -> u32 {
        ((self.run_args.lookup_range.1 - self.run_args.lookup_range.0) as f32)
            .log2()
            .ceil() as u32
    }

    /// Calc the number of rows required for lookup tables
    pub fn lookup_log_rows_with_blinding(&self) -> u32 {
        ((self.run_args.lookup_range.1 - self.run_args.lookup_range.0) as f32
            + RESERVED_BLINDING_ROWS as f32)
            .log2()
            .ceil() as u32
    }

    /// Calc the number of rows required for the range checks
    pub fn range_check_log_rows_with_blinding(&self) -> u32 {
        let max_range = self
            .required_range_checks
            .iter()
            .map(|x| x.1 - x.0)
            .max()
            .unwrap_or(0);

        (max_range as f32).log2().ceil() as u32
    }

    fn model_constraint_logrows_with_blinding(&self) -> u32 {
        (self.num_rows as f64 + RESERVED_BLINDING_ROWS as f64)
            .log2()
            .ceil() as u32
    }

    fn dynamic_lookup_and_shuffle_logrows(&self) -> u32 {
        (self.dynamic_lookup_params.total_dynamic_col_size as f64
            + self.shuffle_params.total_shuffle_col_size as f64)
            .log2()
            .ceil() as u32
    }

    /// calculate the number of rows required for the dynamic lookup and shuffle
    pub fn dynamic_lookup_and_shuffle_logrows_with_blinding(&self) -> u32 {
        (self.dynamic_lookup_params.total_dynamic_col_size as f64
            + self.shuffle_params.total_shuffle_col_size as f64
            + RESERVED_BLINDING_ROWS as f64)
            .log2()
            .ceil() as u32
    }

    /// calculate the number of rows required for the dynamic lookup and shuffle
    pub fn min_dynamic_lookup_and_shuffle_logrows_with_blinding(&self) -> u32 {
        (self.dynamic_lookup_params.max_dynamic_input_len as f64 + RESERVED_BLINDING_ROWS as f64)
            .log2()
            .ceil() as u32
    }

    pub fn dynamic_lookup_and_shuffle_col_size(&self) -> usize {
        self.dynamic_lookup_params.total_dynamic_col_size
            + self.shuffle_params.total_shuffle_col_size
    }

    /// calculate the number of rows required for the module constraints
    pub fn module_constraint_logrows(&self) -> u32 {
        (self.module_sizes.max_constraints() as f64).log2().ceil() as u32
    }

    /// calculate the number of rows required for the module constraints
    pub fn module_constraint_logrows_with_blinding(&self) -> u32 {
        (self.module_sizes.max_constraints() as f64 + RESERVED_BLINDING_ROWS as f64)
            .log2()
            .ceil() as u32
    }

    fn constants_logrows(&self) -> u32 {
        (self.total_const_size as f64 / self.run_args.num_inner_cols as f64)
            .log2()
            .ceil() as u32
    }

    /// Calculates the logrows for einsum computation area in which there is no column overflow
    pub fn einsum_logrows(&self) -> u32 {
        (self.einsum_params.total_einsum_col_size as f64 / self.run_args.num_inner_cols as f64)
            .log2()
            .ceil() as u32
    }

    /// calculate the total number of instances
    pub fn total_instances(&self) -> Vec<usize> {
        let mut instances: Vec<usize> = self.module_sizes.num_instances();
        instances.extend(
            self.model_instance_shapes
                .iter()
                .map(|x| x.iter().product::<usize>()),
        );

        instances
    }

    /// get the scale data for instances
    pub fn get_model_instance_scales(&self) -> Vec<crate::Scale> {
        let mut scales = vec![];
        if self.run_args.input_visibility.is_public() {
            scales.extend(self.model_input_scales.iter().cloned());
        };
        if self.run_args.output_visibility.is_public() {
            scales.extend(self.model_output_scales.iter().cloned());
        };
        scales
    }

    /// calculate the log2 of the total number of instances
    pub fn log2_total_instances(&self) -> u32 {
        let sum = self.total_instances().iter().sum::<usize>();

        // max between 1 and the log2 of the sums
        std::cmp::max((sum as f64).log2().ceil() as u32, 1)
    }

    /// calculate the log2 of the total number of instances
    pub fn log2_total_instances_with_blinding(&self) -> u32 {
        let sum = self.total_instances().iter().sum::<usize>() + RESERVED_BLINDING_ROWS;

        // max between 1 and the log2 of the sums
        std::cmp::max((sum as f64).log2().ceil() as u32, 1)
    }

    /// save params to file
    pub fn save(&self, path: &std::path::PathBuf) -> Result<(), std::io::Error> {
        // buf writer
        let writer =
            std::io::BufWriter::with_capacity(EZKL_BUF_CAPACITY, std::fs::File::create(path)?);
        serde_json::to_writer(writer, &self).map_err(|e| {
            error!("failed to save settings file at {}", e);
            std::io::Error::other(e)
        })
    }

    /// load params from file
    pub fn load(path: &std::path::PathBuf) -> Result<Self, std::io::Error> {
        // buf reader
        let reader =
            std::io::BufReader::with_capacity(EZKL_BUF_CAPACITY, std::fs::File::open(path)?);
        let settings: GraphSettings = serde_json::from_reader(reader).map_err(|e| {
            error!("failed to load settings file at {}", e);
            std::io::Error::other(e)
        })?;

        crate::check_version_string_matches(&settings.version);
        // Keep the thread-local GLOBAL_SETTINGS in sync so circuit configuration can access it.
        GLOBAL_SETTINGS.with(|gs| {
            *gs.borrow_mut() = Some(settings.clone());
        });

        Ok(settings)
    }

    /// Export the ezkl configuration as json
    pub fn as_json(&self) -> Result<String, GraphError> {
        let serialized = match serde_json::to_string(&self) {
            Ok(s) => s,
            Err(e) => return Err(e.into()),
        };
        Ok(serialized)
    }

    /// Parse an ezkl configuration from a json
    pub fn from_json(arg_json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(arg_json)
    }

    fn set_num_blinding_factors(&mut self, num_blinding_factors: usize) {
        self.num_blinding_factors = Some(num_blinding_factors);
    }

    ///
    pub fn available_col_size(&self) -> usize {
        let base = 2u32;
        if let Some(num_blinding_factors) = self.num_blinding_factors {
            base.pow(self.run_args.logrows) as usize - num_blinding_factors - 1
        } else {
            log::error!("num_blinding_factors not set");
            warn!("using default available_col_size");
            base.pow(self.run_args.logrows) as usize - ASSUMED_BLINDING_FACTORS - 1
        }
    }

    /// if any visibility is encrypted or hashed
    pub fn module_requires_fixed(&self) -> bool {
        self.run_args.input_visibility.is_hashed()
            || self.run_args.output_visibility.is_hashed()
            || self.run_args.param_visibility.is_hashed()
    }

    /// requires dynamic lookup
    pub fn requires_dynamic_lookup(&self) -> bool {
        self.dynamic_lookup_params.num_dynamic_lookups > 0
    }

    /// requires dynamic shuffle
    pub fn requires_shuffle(&self) -> bool {
        self.shuffle_params.num_shuffles > 0
    }

    /// any kzg visibility
    pub fn module_requires_polycommit(&self) -> bool {
        self.run_args.input_visibility.is_polycommit()
            || self.run_args.output_visibility.is_polycommit()
            || self.run_args.param_visibility.is_polycommit()
    }
}

#[cfg(test)]
mod tests;
