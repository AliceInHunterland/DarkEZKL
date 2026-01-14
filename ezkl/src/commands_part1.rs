use clap::{Args, Subcommand, ValueEnum};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::str::FromStr;

#[cfg(all(feature = "ezkl", not(target_arch = "wasm32")))]
use tosubcommand::ToSubcommand;
#[cfg(all(feature = "ezkl", not(target_arch = "wasm32")))]
use tosubcommand::ToFlags;

use crate::{
    circuit::{table::Range, CheckMode},
    fieldutils::IntegerRep,
    graph::Visibility,
    CalibrationTarget, ContractType, H160Flag, Scale, DEFAULT_CALIBRATION_FILE,
    DEFAULT_CALIBRATION_TARGET, DEFAULT_CALLDATA, DEFAULT_CHECKMODE, DEFAULT_COMPILED_CIRCUIT,
    DEFAULT_CONTRACT_ADDRESS, DEFAULT_CONTRACT_DEPLOYMENT_TYPE, DEFAULT_DATA,
    DEFAULT_DECIMALS, DEFAULT_DISABLE_SELECTOR_COMPRESSION, DEFAULT_LOOKUP_SAFETY_MARGIN,
    DEFAULT_MODEL, DEFAULT_OPTIMIZER_RUNS, DEFAULT_PK, DEFAULT_PROOF, DEFAULT_RENDER_REUSABLE,
    DEFAULT_SCALE_REBASE_MULTIPLIERS, DEFAULT_SEED, DEFAULT_SETTINGS, DEFAULT_SOL_CODE,
    DEFAULT_USE_REDUCED_SRS_FOR_VERIFICATION, DEFAULT_VERIFIER_ABI, DEFAULT_VK, DEFAULT_VKA,
    DEFAULT_VKA_DIGEST, DEFAULT_WITNESS,
};
#[cfg(feature = "python-bindings")]
use pyo3::{conversion::FromPyObject, exceptions::PyValueError, prelude::*, IntoPyObject};

/// Custom parser for data field that handles both direct JSON strings and file paths with '@' prefix
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, PartialOrd)]
pub struct DataField(pub String);

impl FromStr for DataField {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        // Check if the input starts with '@'
        if let Some(file_path) = s.strip_prefix('@') {
            // Extract the file path (remove the '@' prefix)
            // Read the file content
            let content = std::fs::read_to_string(file_path)
                .map_err(|e| format!("Failed to read data file '{}': {}", file_path, e))?;

            // Return the file content as the data field value
            Ok(DataField(content))
        } else {
            // Use the input string directly
            Ok(DataField(s.to_string()))
        }
    }
}

#[cfg(all(feature = "ezkl", not(target_arch = "wasm32")))]
impl ToFlags for DataField {
    fn to_flags(&self) -> Vec<String> {
        vec![self.0.clone()]
    }
}

impl std::fmt::Display for DataField {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[allow(missing_docs)]
#[derive(Debug, Subcommand, Clone, Deserialize, Serialize, PartialEq, PartialOrd, ToSubcommand)]
pub enum Commands {
    #[cfg(feature = "empty-cmd")]
    /// Creates an empty buffer
    Empty,
    /// Loads model and prints model table
    Table {
        /// The path to the .onnx model file
        #[arg(
            short = 'M',
            long,
            default_value = DEFAULT_MODEL,
            value_hint = clap::ValueHint::FilePath
        )]
        model: Option<PathBuf>,
        /// proving arguments
        #[clap(flatten)]
        args: RunArgs,
    },

    /// Generates the witness from an input file.
    GenWitness {
        /// The path to the .json data file (with @ prefix) or a raw data string of the form '{"input_data": [[1, 2, 3]]}'
        #[arg(
            short = 'D',
            long,
            default_value = DEFAULT_DATA,
            value_parser = DataField::from_str
        )]
        data: Option<DataField>,
        /// The path to the compiled model file (generated using the compile-circuit command)
        #[arg(
            short = 'M',
            long,
            default_value = DEFAULT_COMPILED_CIRCUIT,
            value_hint = clap::ValueHint::FilePath
        )]
        compiled_circuit: Option<PathBuf>,
        /// Path to output the witness .json file
        #[arg(
            short = 'O',
            long,
            default_value = DEFAULT_WITNESS,
            value_hint = clap::ValueHint::FilePath
        )]
        output: Option<PathBuf>,
        /// Path to the verification key file (optional - solely used to generate kzg commits)
        #[arg(short = 'V', long, value_hint = clap::ValueHint::FilePath)]
        vk_path: Option<PathBuf>,
        /// Path to the srs file (optional - solely used to generate kzg commits)
        #[arg(short = 'P', long, value_hint = clap::ValueHint::FilePath)]
        srs_path: Option<PathBuf>,
    },

    /// Produces the proving hyperparameters, from run-args
    GenSettings {
        /// The path to the .onnx model file
        #[arg(
            short = 'M',
            long,
            default_value = DEFAULT_MODEL,
            value_hint = clap::ValueHint::FilePath
        )]
        model: Option<PathBuf>,
        /// The path to generate the circuit settings .json file to
        #[arg(
            short = 'O',
            long,
            default_value = DEFAULT_SETTINGS,
            value_hint = clap::ValueHint::FilePath
        )]
        settings_path: Option<PathBuf>,
        /// proving arguments
        #[clap(flatten)]
        args: RunArgs,
    },
    /// Generate random data for a model
    GenRandomData {
        /// The path to the .onnx model file
        #[arg(
            short = 'M',
            long,
            default_value = DEFAULT_MODEL,
            value_hint = clap::ValueHint::FilePath
        )]
        model: Option<PathBuf>,
        /// The path to the .json data file
        #[arg(short = 'D', long, default_value = DEFAULT_DATA, value_hint = clap::ValueHint::FilePath)]
        data: Option<PathBuf>,
        /// Hand-written parser for graph variables, eg. batch_size=1
        #[cfg_attr(
            all(feature = "ezkl", not(target_arch = "wasm32")),
            arg(
                short = 'V',
                long,
                value_parser = crate::parse_key_val::<String, usize>,
                default_value = "batch_size->1",
                value_delimiter = ',',
                value_hint = clap::ValueHint::Other
            )
        )]
        variables: Vec<(String, usize)>,
        /// random seed for reproducibility (optional)
        #[arg(long, value_hint = clap::ValueHint::Other, default_value = DEFAULT_SEED)]
        seed: u64,
        /// min value for random data
        #[arg(long, value_hint = clap::ValueHint::Other)]
        min: Option<f32>,
        /// max value for random data
        #[arg(long, value_hint = clap::ValueHint::Other)]
        max: Option<f32>,
    },
    /// Calibrates the proving scale, lookup bits and logrows from a circuit settings file.
    CalibrateSettings {
        /// The path to the .json calibration data file.
        #[arg(
            short = 'D',
            long,
            default_value = DEFAULT_CALIBRATION_FILE,
            value_hint = clap::ValueHint::FilePath
        )]
        data: Option<String>,
        /// The path to the .onnx model file
        #[arg(
            short = 'M',
            long,
            default_value = DEFAULT_MODEL,
            value_hint = clap::ValueHint::FilePath
        )]
        model: Option<PathBuf>,
        /// The path to load circuit settings .json file AND overwrite (generated using the gen-settings command).
        #[arg(
            short = 'O',
            long,
            default_value = DEFAULT_SETTINGS,
            value_hint = clap::ValueHint::FilePath
        )]
        settings_path: Option<PathBuf>,
        #[arg(long = "target", default_value = DEFAULT_CALIBRATION_TARGET, value_hint = clap::ValueHint::Other)]
        /// Target for calibration. Set to "resources" to optimize for computational resource. Otherwise, set to "accuracy" to optimize for accuracy.
        target: CalibrationTarget,
        /// the lookup safety margin to use for calibration. if the max lookup is 2^k, then the max lookup will be ceil(2^k * lookup_safety_margin). larger = safer but slower
        #[arg(long, default_value = DEFAULT_LOOKUP_SAFETY_MARGIN, value_hint = clap::ValueHint::Other)]
        lookup_safety_margin: f64,
        /// Optional scales to specifically try for calibration. Example, --scales 0,4
        #[arg(long, value_delimiter = ',', allow_hyphen_values = true, value_hint = clap::ValueHint::Other)]
        scales: Option<Vec<crate::Scale>>,
        /// Optional scale rebase multipliers to specifically try for calibration. This is the multiplier at which we divide to return to the input scale. Example, --scale-rebase-multipliers 0,4
        #[arg(
            long,
            value_delimiter = ',',
            allow_hyphen_values = true,
            default_value = DEFAULT_SCALE_REBASE_MULTIPLIERS,
            value_hint = clap::ValueHint::Other
        )]
        scale_rebase_multiplier: Vec<u32>,
        /// max logrows to use for calibration, 26 is the max public SRS size
        #[arg(long, value_hint = clap::ValueHint::Other)]
        max_logrows: Option<u32>,
    },

    /// Generates a dummy SRS
    #[command(name = "gen-srs", arg_required_else_help = true)]
    GenSrs {
        /// The path to output the generated SRS
        #[arg(long, value_hint = clap::ValueHint::FilePath)]
        srs_path: PathBuf,
        /// number of logrows to use for srs
        #[arg(long, value_hint = clap::ValueHint::Other)]
        logrows: usize,
    },

    /// Gets an SRS from a circuit settings file.
    #[command(name = "get-srs")]
    GetSrs {
        /// The path to output the desired srs file, if set to None will save to ~/.ezkl/srs
        #[arg(long, default_value = None, value_hint = clap::ValueHint::FilePath)]
        srs_path: Option<PathBuf>,
        /// Path to the circuit settings .json file to read in logrows from. Overriden by logrows if specified.
        #[arg(
            short = 'S',
            long,
            default_value = DEFAULT_SETTINGS,
            value_hint = clap::ValueHint::FilePath
        )]
        settings_path: Option<PathBuf>,
        /// Number of logrows to use for srs. Overrides settings_path if specified.
        #[arg(long, default_value = None, value_hint = clap::ValueHint::Other)]
        logrows: Option<u32>,
    },
    /// Loads model and input and runs mock prover (for testing)
    Mock {
        /// The path to the .json witness file (generated using the gen-witness command)
        #[arg(
            short = 'W',
            long,
            default_value = DEFAULT_WITNESS,
            value_hint = clap::ValueHint::FilePath
        )]
        witness: Option<PathBuf>,
        /// The path to the compiled model file (generated using the compile-circuit command)
        #[arg(
            short = 'M',
            long,
            default_value = DEFAULT_COMPILED_CIRCUIT,
            value_hint = clap::ValueHint::FilePath
        )]
        model: Option<PathBuf>,
    },

    /// Compiles a circuit from onnx to a simplified graph (einsum + other ops) and parameters as sets of field elements
    CompileCircuit {
        /// The path to the .onnx model file
        #[arg(
            short = 'M',
            long,
            default_value = DEFAULT_MODEL,
            value_hint = clap::ValueHint::FilePath
        )]
        model: Option<PathBuf>,
        /// The path to the compiled model file (generated using the compile-circuit command)
        #[arg(
            long,
            default_value = DEFAULT_COMPILED_CIRCUIT,
            value_hint = clap::ValueHint::FilePath
        )]
        compiled_circuit: Option<PathBuf>,
        /// The path to load circuit settings .json file from (generated using the gen-settings command)
        #[arg(
            short = 'S',
            long,
            default_value = DEFAULT_SETTINGS,
            value_hint = clap::ValueHint::FilePath
        )]
        settings_path: Option<PathBuf>,
    },
    /// Creates pk and vk
    Setup {
        /// The path to the compiled model file (generated using the compile-circuit command)
        #[arg(
            short = 'M',
            long,
            default_value = DEFAULT_COMPILED_CIRCUIT,
            value_hint = clap::ValueHint::FilePath
        )]
        compiled_circuit: Option<PathBuf>,
        /// The path to SRS, if None will use ~/.ezkl/srs/kzg{logrows}.srs
        #[arg(long, value_hint = clap::ValueHint::FilePath)]
        srs_path: Option<PathBuf>,
        /// The path to output the verification key file to
        #[arg(long, default_value = DEFAULT_VK, value_hint = clap::ValueHint::FilePath)]
        vk_path: Option<PathBuf>,
        /// The path to output the proving key file to
        #[arg(long, default_value = DEFAULT_PK, value_hint = clap::ValueHint::FilePath)]
        pk_path: Option<PathBuf>,
        /// The graph witness (optional - used to override fixed values in the circuit)
        #[arg(short = 'W', long, value_hint = clap::ValueHint::FilePath)]
        witness: Option<PathBuf>,
        /// compress selectors
        #[arg(
            long,
            default_value = DEFAULT_DISABLE_SELECTOR_COMPRESSION,
            action = clap::ArgAction::SetTrue
        )]
        disable_selector_compression: Option<bool>,
    },
    /// Swaps the positions in the transcript that correspond to commitments
    SwapProofCommitments {
        /// The path to the proof file
        #[arg(short = 'P', long, default_value = DEFAULT_PROOF, value_hint = clap::ValueHint::FilePath)]
        proof_path: Option<PathBuf>,
        /// The path to the witness file
        #[arg(short = 'W', long, default_value = DEFAULT_WITNESS, value_hint = clap::ValueHint::FilePath)]
        witness_path: Option<PathBuf>,
    },

    /// Loads model, data, and creates proof
    Prove {
        /// The path to the .json witness file (generated using the gen-witness command)
        #[arg(
            short = 'W',
            long,
            default_value = DEFAULT_WITNESS,
            value_hint = clap::ValueHint::FilePath
        )]
        witness: Option<PathBuf>,
        /// The path to the compiled model file (generated using the compile-circuit command)
        #[arg(
            short = 'M',
            long,
            default_value = DEFAULT_COMPILED_CIRCUIT,
            value_hint = clap::ValueHint::FilePath
        )]
        compiled_circuit: Option<PathBuf>,
        /// The path to load the desired proving key file (generated using the setup command)
        #[arg(long, default_value = DEFAULT_PK, value_hint = clap::ValueHint::FilePath)]
        pk_path: Option<PathBuf>,
        /// The path to output the proof file to
        #[arg(long, default_value = DEFAULT_PROOF, value_hint = clap::ValueHint::FilePath)]
        proof_path: Option<PathBuf>,
        /// The path to SRS, if None will use ~/.ezkl/srs/kzg{logrows}.srs
        #[arg(long, value_hint = clap::ValueHint::FilePath)]
        srs_path: Option<PathBuf>,
        /// run sanity checks during calculations (safe or unsafe)
        #[arg(long, default_value = DEFAULT_CHECKMODE, value_hint = clap::ValueHint::Other)]
        check_mode: Option<CheckMode>,
    },
    /// Encodes a proof into evm calldata
    #[command(name = "encode-evm-calldata")]
    #[cfg(all(feature = "eth", not(target_arch = "wasm32")))]
    EncodeEvmCalldata {
        /// The path to the proof file (generated using the prove command)
        #[arg(long, default_value = DEFAULT_PROOF, value_hint = clap::ValueHint::FilePath)]
        proof_path: Option<PathBuf>,
        /// The path to save the calldata to
        #[arg(long, default_value = DEFAULT_CALLDATA, value_hint = clap::ValueHint::FilePath)]
        calldata_path: Option<PathBuf>,
        /// The path to the serialized VKA file
        #[cfg_attr(
            all(feature = "reusable-verifier", not(target_arch = "wasm32")),
            arg(long, value_hint = clap::ValueHint::Other)
        )]
        vka_path: Option<PathBuf>,
    },
    /// Creates an Evm verifier for a single proof
    #[command(name = "create-evm-verifier")]
    #[cfg(all(feature = "eth", not(target_arch = "wasm32")))]
    CreateEvmVerifier {
        /// The path to SRS, if None will use ~/.ezkl/srs/kzg{logrows}.srs
        #[arg(long, value_hint = clap::ValueHint::FilePath)]
        srs_path: Option<PathBuf>,
        /// The path to load circuit settings .json file from (generated using the gen-settings command)
        #[arg(
            short = 'S',
            long,
            default_value = DEFAULT_SETTINGS,
            value_hint = clap::ValueHint::FilePath
        )]
        settings_path: Option<PathBuf>,
        /// The path to load the desired verification key file
        #[arg(long, default_value = DEFAULT_VK, value_hint = clap::ValueHint::FilePath)]
        vk_path: Option<PathBuf>,
        /// The path to output the Solidity code
        #[arg(long, default_value = DEFAULT_SOL_CODE, value_hint = clap::ValueHint::FilePath)]
        sol_code_path: Option<PathBuf>,
        /// The path to output the Solidity verifier ABI
        #[arg(long, default_value = DEFAULT_VERIFIER_ABI, value_hint = clap::ValueHint::FilePath)]
        abi_path: Option<PathBuf>,
        /// Whether to render the verifier as reusable or not. If true, you will need to deploy a VK artifact, passing it as part of the calldata to the verifier.
        #[cfg_attr(
            all(feature = "reusable-verifier", not(target_arch = "wasm32")),
            arg(
                short = 'R',
                long,
                default_value = DEFAULT_RENDER_REUSABLE,
                action = clap::ArgAction::SetTrue
            )
        )]
        reusable: Option<bool>,
    },
    /// Creates an evm verifier artifact to be used by the reusable verifier
    #[command(name = "create-evm-vka")]
    #[cfg(all(
        feature = "eth",
        feature = "reusable-verifier",
        not(target_arch = "wasm32")
    ))]
    CreateEvmVka {
        /// The path to SRS, if None will use ~/.ezkl/srs/kzg{logrows}.srs
        #[arg(long, value_hint = clap::ValueHint::FilePath)]
        srs_path: Option<PathBuf>,
        /// The path to load circuit settings .json file from (generated using the gen-settings command)
        #[arg(
            short = 'S',
            long,
            default_value = DEFAULT_SETTINGS,
            value_hint = clap::ValueHint::FilePath
        )]
        settings_path: Option<PathBuf>,
        /// The path to load the desired verification key file
        #[arg(long, default_value = DEFAULT_VK, value_hint = clap::ValueHint::FilePath)]
        vk_path: Option<PathBuf>,
        /// The path to output the vka calldata
        #[arg(long, default_value = DEFAULT_VKA, value_hint = clap::ValueHint::FilePath)]
        vka_path: Option<PathBuf>,
        /// The number of decimals we want to use for the rescaling of the instances into on-chain floats
        /// Default is 18, which is the number of decimals used by most ERC20 tokens
        #[arg(long, default_value = DEFAULT_DECIMALS, value_hint = clap::ValueHint::Other)]
        decimals: Option<usize>,
    },

    /// Verifies a proof, returning accept or reject
    Verify {
        /// The path to load circuit settings .json file from (generated using the gen-settings command)
        #[arg(
            short = 'S',
            long,
            default_value = DEFAULT_SETTINGS,
            value_hint = clap::ValueHint::FilePath
        )]
        settings_path: Option<PathBuf>,
        /// The path to the proof file (generated using the prove command)
        #[arg(long, default_value = DEFAULT_PROOF, value_hint = clap::ValueHint::FilePath)]
        proof_path: Option<PathBuf>,
        /// The path to the verification key file (generated using the setup command)
        #[arg(long, default_value = DEFAULT_VK, value_hint = clap::ValueHint::FilePath)]
        vk_path: Option<PathBuf>,
        /// The path to SRS, if None will use ~/.ezkl/srs/kzg{logrows}.srs
        #[arg(long, value_hint = clap::ValueHint::FilePath)]
        srs_path: Option<PathBuf>,
        /// Reduce SRS logrows to the number of instances rather than the number of logrows used for proofs (only works if the srs were generated in the same ceremony)
        #[arg(
            long,
            default_value = DEFAULT_USE_REDUCED_SRS_FOR_VERIFICATION,
            action = clap::ArgAction::SetTrue
        )]
        reduced_srs: Option<bool>,
    },

    /// Deploys an evm contract (verifier, reusable verifier, or vk artifact) that is generated by ezkl
    #[cfg(all(feature = "eth", not(target_arch = "wasm32")))]
    DeployEvm {
        /// The path to the Solidity code (generated using the create-evm-verifier command)
        #[arg(long, default_value = DEFAULT_SOL_CODE, value_hint = clap::ValueHint::FilePath)]
        sol_code_path: Option<PathBuf>,
        /// RPC URL for an Ethereum node
        #[arg(
            short = 'U',
            long,
            default_value = DEFAULT_CONTRACT_ADDRESS,
            value_hint = clap::ValueHint::Url
        )]
        rpc_url: String,
        #[arg(long, default_value = DEFAULT_CONTRACT_ADDRESS, value_hint = clap::ValueHint::Other)]
        /// The path to output the contract address
        addr_path: Option<PathBuf>,
        /// The optimizer runs to set on the verifier. Lower values optimize for deployment cost, while higher values optimize for gas cost.
        #[arg(long, default_value = DEFAULT_OPTIMIZER_RUNS, value_hint = clap::ValueHint::Other)]
        optimizer_runs: usize,
        /// Private secp256K1 key in hex format, 64 chars, no 0x prefix, of the account signing transactions. If None the private key will be generated by Anvil
        #[arg(short = 'P', long, value_hint = clap::ValueHint::Other)]
        private_key: Option<String>,
        /// Contract type to be deployed
        #[cfg(all(feature = "reusable-verifier", not(target_arch = "wasm32")))]
        #[arg(
            long = "contract-type",
            short = 'C',
            default_value = DEFAULT_CONTRACT_DEPLOYMENT_TYPE,
            value_hint = clap::ValueHint::Other
        )]
        contract: ContractType,
    },
    /// Verifies a proof using a local Evm executor, returning accept or reject
    #[command(name = "verify-evm")]
    #[cfg(all(feature = "eth", not(target_arch = "wasm32")))]
    VerifyEvm {
        /// The path to the proof file (generated using the prove command)
        #[arg(long, default_value = DEFAULT_PROOF, value_hint = clap::ValueHint::FilePath)]
        proof_path: Option<PathBuf>,
        /// The path to verifier contract's address
        #[arg(
            long,
            default_value = DEFAULT_CONTRACT_ADDRESS,
            value_hint = clap::ValueHint::Other
        )]
        addr_verifier: H160Flag,
        /// RPC URL for an Ethereum node
        #[arg(short = 'U', long, value_hint = clap::ValueHint::Url)]
        rpc_url: String,
        /// The path to the serialized vka file
        #[cfg_attr(
            all(feature = "reusable-verifier", not(target_arch = "wasm32")),
            arg(long, value_hint = clap::ValueHint::FilePath)
        )]
        vka_path: Option<PathBuf>,
        /// The path to the serialized encoded calldata file generated via the encode_calldata command
        #[arg(long, value_hint = clap::ValueHint::FilePath)]
        encoded_calldata: Option<PathBuf>,
    },
    /// Registers a VKA, returning the its digest used to identify it on-chain.
    #[command(name = "register-vka")]
    #[cfg(feature = "reusable-verifier")]
    RegisterVka {
        /// RPC URL for an Ethereum node, if None will use Anvil but WON'T persist state
        #[arg(short = 'U', long, value_hint = clap::ValueHint::Url)]
        rpc_url: String,
        /// The path to the reusable verifier contract's address
        #[arg(
            long,
            default_value = DEFAULT_CONTRACT_ADDRESS,
            value_hint = clap::ValueHint::Other
        )]
        addr_verifier: H160Flag,
        /// The path to the serialized VKA file
        #[arg(long, default_value = DEFAULT_VKA, value_hint = clap::ValueHint::FilePath)]
        vka_path: Option<PathBuf>,
        /// The path to output the VKA digest to
        #[arg(
            long,
            default_value = DEFAULT_VKA_DIGEST,
            value_hint = clap::ValueHint::FilePath
        )]
        vka_digest_path: Option<PathBuf>,
        /// Private secp256K1 key in hex format, 64 chars, no 0x prefix, of the account signing transactions. If None the private key will be generated by Anvil
        #[arg(short = 'P', long, value_hint = clap::ValueHint::Other)]
        private_key: Option<String>,
    },
    #[cfg(not(feature = "no-update"))]
    /// Updates ezkl binary to version specified (or latest if not specified)
    Update {
        /// The version to update to
        #[arg(value_hint = clap::ValueHint::Other, short = 'v', long)]
        version: Option<String>,
    },
}

impl Commands {
    /// Converts the commands to a json string
    pub fn as_json(&self) -> String {
        serde_json::to_string(self).unwrap()
    }

    /// Converts a json string to a Commands struct
    pub fn from_json(json: &str) -> Self {
        serde_json::from_str(json).unwrap()
    }
}

/// Error type returned by `RunArgs::validate`.
pub type RunArgsError = String;

/// Parse a `Range` from a `min,max` string.
fn parse_range(s: &str) -> Result<Range, String> {
    let parts: Vec<&str> = s.split(',').map(|p| p.trim()).collect();
    if parts.len() != 2 {
        return Err(format!(
            "invalid range '{}': expected 'min,max' (e.g. '-128,127')",
            s
        ));
    }
    let min: IntegerRep = parts[0]
        .parse()
        .map_err(|e| format!("invalid range min '{}': {}", parts[0], e))?;
    let max: IntegerRep = parts[1]
        .parse()
        .map_err(|e| format!("invalid range max '{}': {}", parts[1], e))?;
    Ok((min, max))
}

/// Parse `key->value` (or `key=value`) pairs for `--variables`.
fn parse_usize_kv(s: &str) -> Result<(String, usize), String> {
    let (k, v) = if let Some((k, v)) = s.split_once("->") {
        (k.trim(), v.trim())
    } else if let Some((k, v)) = s.split_once('=') {
        (k.trim(), v.trim())
    } else {
        return Err(format!(
            "invalid variables entry '{}': expected 'key->value' (e.g. 'batch_size->1')",
            s
        ));
    };

    if k.is_empty() {
        return Err(format!("invalid variables entry '{}': key is empty", s));
    }

    let v: usize = v
        .parse()
        .map_err(|e| format!("invalid variables entry '{}': value parse error: {}", s, e))?;

    Ok((k.to_string(), v))
}
