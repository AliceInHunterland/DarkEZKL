#![allow(clippy::needless_return)]

//! Dark‑EZKL library + bindings.
//!
//! This crate supports two build profiles:
//! - **Full / CLI build** (`--features ezkl`): compiles the proving/verification engine and enables the real CLI.
//! - **Minimal Python wheel build** (`--no-default-features --features python-bindings`): builds a small CPython
//!   extension with a limited API surface (currently mostly settings helpers).

/// Returns the crate version from Cargo metadata.
pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

/// Python bindings live under `crate::bindings` (compiled only when `feature="python-bindings"`).
pub mod bindings;

#[cfg(feature = "ezkl")]
use serde::{Deserialize, Serialize};

#[cfg(feature = "ezkl")]
use std::fmt;

#[cfg(feature = "ezkl")]
use std::str::FromStr;

#[cfg(feature = "ezkl")]
/// Fixed-point scale used throughout the crate (log2 multiplier).
pub type Scale = i32;

#[cfg(feature = "ezkl")]
/// Default buffer capacity used for IO (1 MiB).
pub const EZKL_BUF_CAPACITY: usize = 1024 * 1024;

#[cfg(feature = "ezkl")]
/// How halo2 keys/params are serialized on disk.
pub const EZKL_KEY_FORMAT: &str = "processed";

#[cfg(feature = "ezkl")]
// --- CLI defaults (match README examples) ---
pub const DEFAULT_MODEL: &str = "network.onnx";
#[cfg(feature = "ezkl")]
pub const DEFAULT_DATA: &str = "input.json";
#[cfg(feature = "ezkl")]
pub const DEFAULT_SETTINGS: &str = "settings.json";
#[cfg(feature = "ezkl")]
pub const DEFAULT_COMPILED_CIRCUIT: &str = "network.ezkl";
#[cfg(feature = "ezkl")]
pub const DEFAULT_WITNESS: &str = "witness.json";
#[cfg(feature = "ezkl")]
pub const DEFAULT_PROOF: &str = "proof.json";
#[cfg(feature = "ezkl")]
pub const DEFAULT_VK: &str = "vk.key";
#[cfg(feature = "ezkl")]
pub const DEFAULT_PK: &str = "pk.key";

#[cfg(feature = "ezkl")]
pub const DEFAULT_CALLDATA: &str = "calldata.bin";
#[cfg(feature = "ezkl")]
pub const DEFAULT_SOL_CODE: &str = "verifier.sol";
#[cfg(feature = "ezkl")]
pub const DEFAULT_VERIFIER_ABI: &str = "verifier.abi.json";
#[cfg(feature = "ezkl")]
pub const DEFAULT_VKA: &str = "vka.bin";
#[cfg(feature = "ezkl")]
pub const DEFAULT_VKA_DIGEST: &str = "vka_digest.txt";

#[cfg(feature = "ezkl")]
pub const DEFAULT_SEED: &str = "0";
#[cfg(feature = "ezkl")]
pub const DEFAULT_DECIMALS: &str = "18";
#[cfg(feature = "ezkl")]
pub const DEFAULT_OPTIMIZER_RUNS: &str = "200";

#[cfg(feature = "ezkl")]
pub const DEFAULT_DISABLE_SELECTOR_COMPRESSION: &str = "false";
#[cfg(feature = "ezkl")]
pub const DEFAULT_RENDER_REUSABLE: &str = "false";
#[cfg(feature = "ezkl")]
pub const DEFAULT_USE_REDUCED_SRS_FOR_VERIFICATION: &str = "false";

#[cfg(feature = "ezkl")]
pub const DEFAULT_LOOKUP_SAFETY_MARGIN: &str = "1.1";
#[cfg(feature = "ezkl")]
pub const DEFAULT_SCALE_REBASE_MULTIPLIERS: &str = "10";

#[cfg(feature = "ezkl")]
pub const DEFAULT_CALIBRATION_FILE: &str = "calibration.json";
#[cfg(feature = "ezkl")]
pub const DEFAULT_CALIBRATION_TARGET: &str = "resources";

#[cfg(feature = "ezkl")]
// NOTE: This constant name comes from upstream CLI plumbing. It is used as a default
// for some EVM-related flags; keep it parseable as a URL.
pub const DEFAULT_CONTRACT_ADDRESS: &str = "http://localhost:8545";
#[cfg(feature = "ezkl")]
pub const DEFAULT_CONTRACT_DEPLOYMENT_TYPE: &str = "verifier";

#[cfg(feature = "ezkl")]
pub const DEFAULT_CHECKMODE: &str = "safe";

#[cfg(feature = "ezkl")]
/// Top-level error type used by the CLI/execution layer.
///
/// Important: do NOT implement `From<&str>`/`From<String>` alongside a blanket
/// `From<E: Error>` impl — it triggers Rust coherence conflicts (because std could
/// theoretically implement `Error` for `&str`/`String` in the future).
#[derive(Debug, Clone)]
pub struct EZKLError(pub String);

#[cfg(feature = "ezkl")]
impl EZKLError {
    /// Construct a message-only error (replacement for the removed `From<&str>`/`From<String>`).
    pub fn msg<T: Into<String>>(s: T) -> Self {
        Self(s.into())
    }
}

#[cfg(feature = "ezkl")]
impl fmt::Display for EZKLError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

#[cfg(feature = "ezkl")]
impl<E> From<E> for EZKLError
where
    E: std::error::Error,
{
    fn from(e: E) -> Self {
        Self(e.to_string())
    }
}

#[cfg(feature = "ezkl")]
/// Parse `key->value` or `key=value`.
///
/// Used by some CLI arguments (e.g. `--variables`).
pub fn parse_key_val<K, V>(s: &str) -> Result<(K, V), String>
where
    K: FromStr,
    V: FromStr,
    K::Err: fmt::Display,
    V::Err: fmt::Display,
{
    let (k, v) = if let Some((k, v)) = s.split_once("->") {
        (k.trim(), v.trim())
    } else if let Some((k, v)) = s.split_once('=') {
        (k.trim(), v.trim())
    } else {
        return Err(format!(
            "invalid key/value '{}': expected 'key->value' or 'key=value'",
            s
        ));
    };

    let k_parsed = k
        .parse::<K>()
        .map_err(|e| format!("invalid key '{}' in '{}': {}", k, s, e))?;
    let v_parsed = v
        .parse::<V>()
        .map_err(|e| format!("invalid value '{}' in '{}': {}", v, s, e))?;

    Ok((k_parsed, v_parsed))
}

#[cfg(feature = "ezkl")]
/// Warn if a settings/proof/witness file version doesn't match the current crate version.
///
/// (Non-fatal by design: we want to be able to load artifacts from nearby versions in research workflows.)
pub fn check_version_string_matches(version_in_file: &str) {
    if version_in_file.trim().is_empty() {
        return;
    }

    let current = version();
    if version_in_file != current {
        log::warn!(
            "artifact version mismatch: file has ezkl version {}, but this build is {}",
            version_in_file,
            current
        );
    }
}

#[cfg(feature = "ezkl")]
/// Calibration target for `calibrate-settings`.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum CalibrationTarget {
    /// Optimize primarily for proving resources (logrows, memory).
    Resources {
        /// If true, also shrink `logrows` to avoid column overflow when possible.
        col_overflow: bool,
    },
    /// Optimize primarily for accuracy (larger scales, then smallest logrows).
    Accuracy,
}

#[cfg(feature = "ezkl")]
impl Default for CalibrationTarget {
    fn default() -> Self {
        CalibrationTarget::Resources {
            col_overflow: false,
        }
    }
}

#[cfg(feature = "ezkl")]
impl fmt::Display for CalibrationTarget {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CalibrationTarget::Resources {
                col_overflow: false,
            } => write!(f, "resources"),
            CalibrationTarget::Resources { col_overflow: true } => {
                write!(f, "resources_col_overflow")
            }
            CalibrationTarget::Accuracy => write!(f, "accuracy"),
        }
    }
}

#[cfg(feature = "ezkl")]
impl FromStr for CalibrationTarget {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let v = s.trim().to_lowercase();
        match v.as_str() {
            "resources" => Ok(CalibrationTarget::Resources { col_overflow: false }),
            "resources_col_overflow" | "resources-col-overflow" | "resources/col_overflow" => {
                Ok(CalibrationTarget::Resources { col_overflow: true })
            }
            "accuracy" => Ok(CalibrationTarget::Accuracy),
            _ => Err(format!(
                "invalid calibration target '{}'; expected 'resources', 'resources_col_overflow', or 'accuracy'",
                s
            )),
        }
    }
}

#[cfg(feature = "ezkl")]
/// EVM contract type for `deploy-evm` (minimal representation used by this snapshot).
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ContractType {
    /// Deploy a verifier (optionally reusable).
    Verifier { reusable: bool },
}

#[cfg(feature = "ezkl")]
impl Default for ContractType {
    fn default() -> Self {
        ContractType::Verifier { reusable: false }
    }
}

#[cfg(feature = "ezkl")]
impl fmt::Display for ContractType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ContractType::Verifier { reusable: false } => write!(f, "verifier"),
            ContractType::Verifier { reusable: true } => write!(f, "verifier_reusable"),
        }
    }
}

#[cfg(feature = "ezkl")]
impl FromStr for ContractType {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let v = s.trim().to_lowercase();
        match v.as_str() {
            "verifier" => Ok(ContractType::Verifier { reusable: false }),
            "verifier_reusable" | "verifier-reusable" | "reusable_verifier"
            | "reusable-verifier" => Ok(ContractType::Verifier { reusable: true }),
            _ => Err(format!(
                "invalid contract type '{}'; expected 'verifier' or 'verifier_reusable'",
                s
            )),
        }
    }
}

#[cfg(feature = "ezkl")]
/// CLI-friendly 20-byte address type (0x-prefixed hex).
///
/// This avoids pulling EVM/eth dependencies into the core CLI when `feature="eth"` is disabled.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct H160Flag(pub [u8; 20]);

#[cfg(feature = "ezkl")]
impl fmt::Display for H160Flag {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "0x")?;
        for b in self.0 {
            write!(f, "{:02x}", b)?;
        }
        Ok(())
    }
}

#[cfg(feature = "ezkl")]
impl FromStr for H160Flag {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let mut v = s.trim();
        if let Some(rest) = v.strip_prefix("0x") {
            v = rest;
        }
        if v.len() != 40 {
            return Err(format!(
                "invalid address '{}': expected 40 hex chars (optionally prefixed with 0x)",
                s
            ));
        }

        let mut out = [0u8; 20];
        for i in 0..20 {
            let byte_str = &v[i * 2..i * 2 + 2];
            out[i] = u8::from_str_radix(byte_str, 16).map_err(|e| {
                format!(
                    "invalid address '{}': bad hex at byte {} ('{}'): {}",
                    s, i, byte_str, e
                )
            })?;
        }
        Ok(H160Flag(out))
    }
}

// -------------------------------------------------------------------------------------------------
// Full engine modules (only compiled when feature="ezkl")
// -------------------------------------------------------------------------------------------------

#[cfg(feature = "ezkl")]
pub mod circuit;
#[cfg(feature = "ezkl")]
pub mod commands;
#[cfg(feature = "ezkl")]
pub mod execute;
#[cfg(feature = "ezkl")]
pub mod fieldutils;
#[cfg(feature = "ezkl")]
pub mod graph;
#[cfg(feature = "ezkl")]
pub mod logger;
#[cfg(feature = "ezkl")]
pub mod pfsys;
#[cfg(feature = "ezkl")]
pub mod srs_sha;
#[cfg(feature = "ezkl")]
pub mod tensor;

// Optional EVM support (compiled only when enabled).
#[cfg(all(feature = "ezkl", feature = "eth"))]
pub mod eth;

// Re-exports expected throughout the codebase (crate::RunArgs, crate::ExecutionMode, etc.)
#[cfg(feature = "ezkl")]
pub use circuit::CheckMode;
#[cfg(feature = "ezkl")]
pub use commands::{Commands, RunArgs};
#[cfg(feature = "ezkl")]
pub use graph::{ExecutionMode, ProbOp, ProbOps, ProbSeedMode, ProbabilisticSettings};

/// Returns the crate version (Python-facing alias).
#[cfg(all(feature = "python-bindings", not(target_arch = "wasm32")))]
use pyo3::prelude::*;

#[cfg(all(feature = "python-bindings", not(target_arch = "wasm32")))]
use pyo3::types::PyModuleMethods;

#[cfg(all(feature = "python-bindings", not(target_arch = "wasm32")))]
#[pyfunction]
#[pyo3(name = "version")]
fn version_py() -> &'static str {
    version()
}

#[cfg(all(feature = "python-bindings", not(target_arch = "wasm32")))]
#[pymodule]
#[pyo3(name = "ezkl")]
pub fn ezkl(module: &Bound<'_, PyModule>) -> PyResult<()> {
    // Conventional Python version attribute.
    module.add("__version__", version())?;
    // Provide a callable `version()` too.
    module.add_function(wrap_pyfunction!(version_py, module)?)?;

    // Register Python helpers (gen_settings / calibrate_settings stubs).
    bindings::python::register(module.py(), module)?;

    Ok(())
}
