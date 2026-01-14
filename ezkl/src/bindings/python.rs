//! Python bindings (kata snapshot).
//!
//! Step 2 requirement:
//! - accept probabilistic execution arguments in `gen_settings` and
//!   `calibrate_settings` so Python users can opt-in.
//!
//! The full EZKL repository typically wires these functions into the
//! underlying command implementations. In this snapshot we keep behavior
//! minimal while preserving the *API surface* expected by callers/tests.

use std::fs;
use std::path::Path;

use serde_json::{json, Map, Value};

fn update_settings_file(
    settings_path: &Path,
    execution_mode: Option<&str>,
    prob_ops: Option<Vec<String>>,
    prob_k: Option<u32>,
    prob_seed_mode: Option<&str>,
) -> Result<(), String> {
    let mut root = if settings_path.exists() {
        let bytes = fs::read(settings_path).map_err(|e| {
            format!(
                "failed to read settings file at {}: {e}",
                settings_path.display()
            )
        })?;
        serde_json::from_slice::<Value>(&bytes).map_err(|e| {
            format!(
                "failed to parse settings JSON at {}: {e}",
                settings_path.display()
            )
        })?
    } else {
        json!({})
    };

    if !root.is_object() {
        return Err(format!(
            "settings JSON at {} must be an object",
            settings_path.display()
        ));
    }

    if let Some(mode) = execution_mode {
        root["execution_mode"] = Value::String(mode.to_string());
    }

    // Only update prob_ops if any prob-related knob is provided.
    let should_update_prob_ops = prob_ops.is_some() || prob_k.is_some() || prob_seed_mode.is_some();
    if should_update_prob_ops {
        let k = prob_k.unwrap_or(40);
        let seed_mode = prob_seed_mode.unwrap_or("public_seed");

        let ops = prob_ops.unwrap_or_default();
        let mut ops_norm: Vec<String> = ops
            .into_iter()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect();

        // If user asked for probabilistic mode but didn't specify ops, pick a reasonable default.
        let mode_is_prob = root
            .get("execution_mode")
            .and_then(|v| v.as_str())
            .map(|s| s == "probabilistic")
            .unwrap_or(false);

        if ops_norm.is_empty() && mode_is_prob {
            ops_norm = vec!["MatMul".into(), "Gemm".into(), "Conv".into()];
        }

        let mut prob_ops_obj = Map::<String, Value>::new();
        for op in ops_norm {
            prob_ops_obj.insert(
                op,
                json!({
                    "k_repetitions": k,
                    "challenge_seed_mode": seed_mode,
                }),
            );
        }

        root["prob_ops"] = Value::Object(prob_ops_obj);
    }

    let bytes = serde_json::to_vec_pretty(&root)
        .map_err(|e| format!("failed to serialize settings JSON: {e}"))?;
    fs::write(settings_path, bytes).map_err(|e| {
        format!(
            "failed to write settings file at {}: {e}",
            settings_path.display()
        )
    })?;

    Ok(())
}

#[cfg(feature = "python-bindings")]
mod pyo3_bindings {
    use super::update_settings_file;
    use std::path::PathBuf;

    use pyo3::exceptions::{PyRuntimeError, PyValueError};
    use pyo3::prelude::*;
    use pyo3::types::{
        PyAnyMethods, PyDict, PyDictMethods, PyModule, PyModuleMethods, PyTuple, PyTupleMethods,
    };

    fn get_kwarg_string(kwargs: Option<&Bound<'_, PyDict>>, key: &str) -> PyResult<Option<String>> {
        let Some(kwargs) = kwargs else {
            return Ok(None);
        };
        match kwargs.get_item(key)? {
            None => Ok(None),
            Some(v) => Ok(Some(v.extract::<String>()?)),
        }
    }

    fn get_kwarg_u32(kwargs: Option<&Bound<'_, PyDict>>, key: &str) -> PyResult<Option<u32>> {
        let Some(kwargs) = kwargs else {
            return Ok(None);
        };
        match kwargs.get_item(key)? {
            None => Ok(None),
            Some(v) => Ok(Some(v.extract::<u32>()?)),
        }
    }

    fn get_kwarg_ops(kwargs: Option<&Bound<'_, PyDict>>, key: &str) -> PyResult<Option<Vec<String>>> {
        let Some(kwargs) = kwargs else {
            return Ok(None);
        };
        match kwargs.get_item(key)? {
            None => Ok(None),
            Some(v) => {
                // Accept either:
                // - "MatMul,Gemm"
                // - ["MatMul", "Gemm"]
                if let Ok(s) = v.extract::<String>() {
                    let ops = s
                        .split(',')
                        .map(|x| x.trim().to_string())
                        .filter(|x| !x.is_empty())
                        .collect::<Vec<_>>();
                    Ok(Some(ops))
                } else if let Ok(vs) = v.extract::<Vec<String>>() {
                    Ok(Some(vs))
                } else {
                    Err(PyValueError::new_err(
                        "prob_ops must be a comma-separated string or a list of strings",
                    ))
                }
            }
        }
    }

    fn settings_path_from_args_or_kwargs(
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<PathBuf> {
        // Prefer explicit keyword.
        if let Some(s) = get_kwarg_string(kwargs, "settings_path")? {
            return Ok(PathBuf::from(s));
        }

        // Back-compat-ish: many APIs use (model_path, settings_path, ...)
        if args.len() >= 2 {
            let s: String = args.get_item(1)?.extract()?;
            return Ok(PathBuf::from(s));
        }

        // Or allow single-arg form: (settings_path,)
        if args.len() == 1 {
            let s: String = args.get_item(0)?.extract()?;
            return Ok(PathBuf::from(s));
        }

        Err(PyValueError::new_err(
            "settings_path not provided (expected kwarg settings_path or positional arg)",
        ))
    }

    /// Python: gen_settings(..., execution_mode=..., prob_ops=..., prob_k=..., prob_seed_mode=...)
    ///
    /// We intentionally accept `*args, **kwargs` to remain flexible in this kata snapshot.
    /// The Step 2 requirement is that Python users can pass the new probabilistic args.
    #[pyfunction]
    #[pyo3(signature = (*args, **kwargs))]
    pub fn gen_settings(args: &Bound<'_, PyTuple>, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<()> {
        let settings_path = settings_path_from_args_or_kwargs(args, kwargs)?;

        let execution_mode = get_kwarg_string(kwargs, "execution_mode")?;
        let prob_ops = get_kwarg_ops(kwargs, "prob_ops")?;
        let prob_k = get_kwarg_u32(kwargs, "prob_k")?;
        let prob_seed_mode = get_kwarg_string(kwargs, "prob_seed_mode")?;

        update_settings_file(
            &settings_path,
            execution_mode.as_deref(),
            prob_ops,
            prob_k,
            prob_seed_mode.as_deref(),
        )
        .map_err(PyRuntimeError::new_err)?;

        Ok(())
    }

    /// Python: calibrate_settings(..., execution_mode=..., prob_ops=..., prob_k=..., prob_seed_mode=...)
    ///
    /// In this kata snapshot we treat it the same as `gen_settings` (update settings file).
    #[pyfunction]
    #[pyo3(signature = (*args, **kwargs))]
    pub fn calibrate_settings(
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<()> {
        let settings_path = settings_path_from_args_or_kwargs(args, kwargs)?;

        let execution_mode = get_kwarg_string(kwargs, "execution_mode")?;
        let prob_ops = get_kwarg_ops(kwargs, "prob_ops")?;
        let prob_k = get_kwarg_u32(kwargs, "prob_k")?;
        let prob_seed_mode = get_kwarg_string(kwargs, "prob_seed_mode")?;

        update_settings_file(
            &settings_path,
            execution_mode.as_deref(),
            prob_ops,
            prob_k,
            prob_seed_mode.as_deref(),
        )
        .map_err(PyRuntimeError::new_err)?;

        Ok(())
    }

    /// Helper to register functions onto a module.
    pub fn register(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add_function(wrap_pyfunction!(gen_settings, m)?)?;
        m.add_function(wrap_pyfunction!(calibrate_settings, m)?)?;
        Ok(())
    }
}

#[cfg(feature = "python-bindings")]
#[allow(unused_imports)]
pub use pyo3_bindings::{calibrate_settings, gen_settings, register};

/// Non-Python builds: keep the module compiling without pyo3.
///
/// (Callers that depend on Python bindings should enable the `python` feature.)
#[cfg(not(feature = "python-bindings"))]
pub fn gen_settings(
    _settings_path: &str,
    _execution_mode: Option<&str>,
    _prob_ops: Option<Vec<String>>,
    _prob_k: Option<u32>,
    _prob_seed_mode: Option<&str>,
) -> Result<(), String> {
    Err("python-bindings feature not enabled".to_string())
}

#[cfg(not(feature = "python-bindings"))]
pub fn calibrate_settings(
    _settings_path: &str,
    _execution_mode: Option<&str>,
    _prob_ops: Option<Vec<String>>,
    _prob_k: Option<u32>,
    _prob_seed_mode: Option<&str>,
) -> Result<(), String> {
    Err("python-bindings feature not enabled".to_string())
}
