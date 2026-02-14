// Re-export the "execution contract" types from the settings layer so the CLI and settings.json
// share a single set of enums/structs.
pub use crate::graph::{ExecutionMode, ProbOp, ProbOps, ProbSeedMode, ProbabilisticSettings};

pub const DEFAULT_EXECUTION_MODE: ExecutionMode = ExecutionMode::Exact;
pub const DEFAULT_PROB_K_U32: u32 = 40;
pub const DEFAULT_PROB_OPS: &str = "MatMul,Gemm,Conv";
pub const DEFAULT_PROB_SEED_MODE: ProbSeedMode = ProbSeedMode::PublicSeed;

#[allow(missing_docs)]
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Args)]
#[serde(default)]
pub struct RunArgs {
    /// int: The denominator in the fixed point representation used when quantizing inputs
    #[arg(long, default_value_t = 0)]
    pub input_scale: Scale,

    /// int:  The denominator in the fixed point representation used when quantizing parameters
    #[arg(long, default_value_t = 0)]
    pub param_scale: Scale,

    /// int: The scale to rebase to (optional). If None, we rebase to the max of input_scale and param_scale
    /// This is an advanced parameter that should be used with caution
    #[arg(long)]
    pub rebase_scale: Option<Scale>,

    /// int: If the scale is ever > scale_rebase_multiplier * input_scale then the scale is rebased to input_scale (this a more advanced parameter, use with caution)
    #[arg(long, default_value_t = 10)]
    pub scale_rebase_multiplier: u32,

    /// list[int]: The min and max elements in the lookup table input column
    #[arg(long, value_parser = parse_range, default_value = "0,0")]
    pub lookup_range: Range,

    /// int: The log_2 number of rows
    #[arg(long, default_value_t = 6)]
    pub logrows: u32,

    /// int: The number of inner columns used for the lookup table
    #[arg(long, default_value_t = 2)]
    pub num_inner_cols: usize,

    /// string: accepts `public`, `private`, `fixed`, `hashed/public`, `hashed/private/<outlets>`, `polycommit`
    #[arg(long, default_value_t = Visibility::Private)]
    pub input_visibility: Visibility,

    /// string: accepts `public`, `private`, `fixed`, `hashed/public`, `hashed/private/<outlets>`, `polycommit`
    #[arg(long, default_value_t = Visibility::Public)]
    pub output_visibility: Visibility,

    /// string: accepts `public`, `private`, `fixed`, `hashed/public`, `hashed/private/<outlets>`, `polycommit`
    #[arg(long, default_value_t = Visibility::Private)]
    pub param_visibility: Visibility,

    /// list[tuple[str, int]]: Hand-written parser for graph variables, eg. batch_size=1
    #[arg(
        long,
        value_delimiter = ',',
        default_value = "batch_size->1",
        value_parser = parse_usize_kv
    )]
    pub variables: Vec<(String, usize)>,

    /// bool: Should constants with 0.0 fraction be rebased to scale 0
    #[arg(long, default_value_t = false)]
    pub rebase_frac_zero_constants: bool,

    /// str: check mode, accepts `safe`, `unsafe`
    #[arg(long, default_value = DEFAULT_CHECKMODE)]
    pub check_mode: CheckMode,

    /// int: The base used for decomposition
    #[arg(long, default_value_t = 128)]
    pub decomp_base: usize,

    /// int: The number of legs used for decomposition
    ///
    /// IMPORTANT:
    /// A too-small value can make `calibrate-settings` fail with:
    ///   "[tensor] decomposition error: integer X is too large to be represented by base 128 and n <legs>"
    ///
    /// We default to 5 (instead of 4) so typical NN activations fit during calibration/witnessing.
    #[arg(long, default_value_t = 5)]
    pub decomp_legs: usize,

    /// bool: Should the circuit use unbounded lookups for log
    #[arg(long, default_value_t = false)]
    pub bounded_log_lookup: bool,

    /// bool: Should the circuit use range checks for inputs and outputs (set to false if the input is a felt)
    #[arg(long, default_value_t = false)]
    pub ignore_range_check_inputs_outputs: bool,

    /// float: epsilon used for arguments that use division
    #[arg(long)]
    pub epsilon: Option<f64>,

    /// bool: Whether to disable using Freivalds' argument in einsum operations
    #[arg(long, default_value_t = false)]
    pub disable_freivalds: bool,

    // --- probabilistic execution settings ---

    /// str: execution mode, accepts `exact` or `probabilistic`
    #[arg(long = "execution-mode", default_value_t = DEFAULT_EXECUTION_MODE)]
    pub execution_mode: ExecutionMode,

    /// str: ops to apply probabilistic checks to (comma-separated). Example: MatMul,Gemm,Conv
    #[arg(long = "prob-ops", default_value = DEFAULT_PROB_OPS)]
    pub prob_ops: ProbOps,

    /// int: number of Freivalds repetitions in probabilistic execution mode
    #[arg(long = "prob-k", default_value_t = DEFAULT_PROB_K_U32)]
    pub prob_k: u32,

    /// str: probabilistic seed mode, accepts `public_seed` or `fiat_shamir`
    #[arg(long = "prob-seed-mode", default_value_t = DEFAULT_PROB_SEED_MODE)]
    pub prob_seed_mode: ProbSeedMode,
}

impl Default for RunArgs {
    fn default() -> Self {
        Self {
            input_scale: 0,
            param_scale: 0,
            rebase_scale: None,
            scale_rebase_multiplier: 10,
            lookup_range: (0, 0),
            logrows: 6,
            num_inner_cols: 2,
            input_visibility: Visibility::Private,
            output_visibility: Visibility::Public,
            param_visibility: Visibility::Private,
            variables: vec![("batch_size".to_string(), 1)],
            rebase_frac_zero_constants: false,
            check_mode: CheckMode::SAFE,
            decomp_base: 128,
            decomp_legs: 5,
            bounded_log_lookup: false,
            ignore_range_check_inputs_outputs: false,
            epsilon: None,
            disable_freivalds: false,
            execution_mode: DEFAULT_EXECUTION_MODE,
            prob_ops: ProbOps::default(),
            prob_k: DEFAULT_PROB_K_U32,
            prob_seed_mode: DEFAULT_PROB_SEED_MODE,
        }
    }
}

impl RunArgs {
    /// Returns epsilon used for division-ish ops (if not set, returns 0.0).
    pub fn get_epsilon(&self) -> f64 {
        self.epsilon.unwrap_or(0.0)
    }

    /// Validate run args (lightweight; meant to prevent obvious config mistakes).
    pub fn validate(&self) -> Result<(), RunArgsError> {
        if self.logrows == 0 {
            return Err("logrows must be > 0".to_string());
        }
        if self.num_inner_cols == 0 {
            return Err("num_inner_cols must be > 0".to_string());
        }
        if self.lookup_range.0 > self.lookup_range.1 {
            return Err("lookup_range must satisfy min <= max".to_string());
        }
        if self.execution_mode == ExecutionMode::Probabilistic && self.prob_k == 0 {
            return Err("prob_k must be > 0 in probabilistic mode".to_string());
        }
        Ok(())
    }
}
