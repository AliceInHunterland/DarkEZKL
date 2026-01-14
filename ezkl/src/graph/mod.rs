use serde::{Deserialize, Serialize};
use std::str::FromStr;

#[cfg(all(feature = "ezkl", not(target_arch = "wasm32")))]
use tosubcommand::ToFlags;

#[cfg(feature = "python-bindings")]
use pyo3::{conversion::FromPyObject, exceptions::PyValueError, prelude::*, IntoPyObject};

/// Representations of a computational graph's inputs.
pub mod input;
/// Crate for defining a computational graph and building a ZK-circuit from it.
pub mod model;
/// Representations of a computational graph's modules.
pub mod modules;
/// Inner elements of a computational graph that represent a single operation / constraints.
pub mod node;
/// Helper functions
pub mod utilities;
/// Representations of a computational graph's variables.
pub mod vars;

/// errors for the graph
pub mod errors;

/// Graph settings (settings.json), circuit sizing metadata, and shared constants.
pub mod config;
/// Graph witness (witness.json) and related helpers.
pub mod witness;

pub use config::*;
pub use witness::*;

pub use input::DataSource;
pub use model::*;
pub use node::*;
pub use utilities::*;
pub use vars::*;

impl std::fmt::Display for ExecutionMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ExecutionMode::Exact => write!(f, "exact"),
            ExecutionMode::Probabilistic => write!(f, "probabilistic"),
        }
    }
}

impl FromStr for ExecutionMode {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "exact" => Ok(ExecutionMode::Exact),
            "probabilistic" => Ok(ExecutionMode::Probabilistic),
            _ => Err("Invalid value for ExecutionMode".to_string()),
        }
    }
}

#[cfg(feature = "python-bindings")]
impl<'py> IntoPyObject<'py> for ExecutionMode {
    type Target = pyo3::PyAny;
    type Output = pyo3::Bound<'py, Self::Target>;
    type Error = pyo3::PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        let result = match self {
            ExecutionMode::Exact => "exact",
            ExecutionMode::Probabilistic => "probabilistic",
        };
        Ok(result.into_pyobject(py)?.into_any())
    }
}

#[cfg(feature = "python-bindings")]
impl<'source> FromPyObject<'source> for ExecutionMode {
    fn extract_bound(ob: &pyo3::Bound<'source, pyo3::PyAny>) -> PyResult<Self> {
        let trystr = String::extract_bound(ob)?;
        match trystr.to_lowercase().as_str() {
            "exact" => Ok(ExecutionMode::Exact),
            "probabilistic" => Ok(ExecutionMode::Probabilistic),
            _ => Err(PyValueError::new_err("Invalid value for ExecutionMode")),
        }
    }
}

#[cfg(all(feature = "ezkl", not(target_arch = "wasm32")))]
impl ToFlags for ExecutionMode {
    fn to_flags(&self) -> Vec<String> {
        vec![format!("{}", self)]
    }
}

impl std::fmt::Display for ProbOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ProbOp::MatMul => write!(f, "matmul"),
            ProbOp::Gemm => write!(f, "gemm"),
            ProbOp::Conv => write!(f, "conv"),
        }
    }
}

impl FromStr for ProbOp {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let v = s.trim().to_lowercase();
        match v.as_str() {
            "matmul" | "mat_mul" | "mat-mul" => Ok(ProbOp::MatMul),
            "gemm" => Ok(ProbOp::Gemm),
            "conv" | "convolution" => Ok(ProbOp::Conv),
            _ => Err(format!(
                "Invalid value for ProbOp '{}'. Expected one of: MatMul,Gemm,Conv",
                s
            )),
        }
    }
}

impl std::fmt::Display for ProbOps {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let parts: Vec<String> = self.0.iter().map(|op| op.to_string()).collect();
        write!(f, "{}", parts.join(","))
    }
}

impl FromStr for ProbOps {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let trimmed = s.trim();
        if trimmed.is_empty() {
            return Ok(ProbOps(vec![]));
        }

        let mut ops = Vec::new();
        for part in trimmed.split(',') {
            let p = part.trim();
            if p.is_empty() {
                continue;
            }
            ops.push(ProbOp::from_str(p)?);
        }
        Ok(ProbOps(ops))
    }
}

#[cfg(all(feature = "ezkl", not(target_arch = "wasm32")))]
impl ToFlags for ProbOps {
    fn to_flags(&self) -> Vec<String> {
        vec![format!("{}", self)]
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, PartialOrd, Default)]
#[serde(rename_all = "snake_case")]
pub enum ProbSeedMode {
    #[default]
    #[serde(alias = "PublicSeed", alias = "PUBLICSEED")]
    PublicSeed,
    #[serde(alias = "FiatShamir", alias = "FIATSHAMIR", alias = "fiat_shamir")]
    FiatShamir,
}

impl std::fmt::Display for ProbSeedMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ProbSeedMode::PublicSeed => write!(f, "public_seed"),
            ProbSeedMode::FiatShamir => write!(f, "fiat_shamir"),
        }
    }
}

impl FromStr for ProbSeedMode {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "public_seed" => Ok(ProbSeedMode::PublicSeed),
            "fiat_shamir" => Ok(ProbSeedMode::FiatShamir),
            _ => Err("Invalid value for ProbSeedMode".to_string()),
        }
    }
}

#[cfg(feature = "python-bindings")]
impl<'py> IntoPyObject<'py> for ProbSeedMode {
    type Target = pyo3::PyAny;
    type Output = pyo3::Bound<'py, Self::Target>;
    type Error = pyo3::PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        let result = match self {
            ProbSeedMode::PublicSeed => "public_seed",
            ProbSeedMode::FiatShamir => "fiat_shamir",
        };
        Ok(result.into_pyobject(py)?.into_any())
    }
}

#[cfg(feature = "python-bindings")]
impl<'source> FromPyObject<'source> for ProbSeedMode {
    fn extract_bound(ob: &pyo3::Bound<'source, pyo3::PyAny>) -> PyResult<Self> {
        let trystr = String::extract_bound(ob)?;
        match trystr.to_lowercase().as_str() {
            "public_seed" => Ok(ProbSeedMode::PublicSeed),
            "fiat_shamir" => Ok(ProbSeedMode::FiatShamir),
            _ => Err(PyValueError::new_err("Invalid value for ProbSeedMode")),
        }
    }
}

#[cfg(all(feature = "ezkl", not(target_arch = "wasm32")))]
impl ToFlags for ProbSeedMode {
    fn to_flags(&self) -> Vec<String> {
        vec![format!("{}", self)]
    }
}

/// Settings for probabilistic execution.
///
/// This is intentionally small and auditable since it modifies the statement being proven:
/// * `k_repetitions`: soundness error per-check is ≤ 2^-k (for classic 0/1 Freivalds vectors).
/// * `seed_mode`: how challenges are derived (public seed recommended for deployments).
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, PartialOrd)]
#[serde(rename_all = "snake_case")]
pub struct ProbabilisticSettings {
    /// Number of Freivalds repetitions.
    #[serde(alias = "prob_k", alias = "k")]
    pub k_repetitions: u32,
    /// Randomness / challenge derivation mode.
    #[serde(alias = "prob_seed_mode")]
    pub seed_mode: ProbSeedMode,
}

impl Default for ProbabilisticSettings {
    fn default() -> Self {
        Self {
            k_repetitions: 40,
            seed_mode: ProbSeedMode::PublicSeed,
        }
    }
}
