//! Python bindings (kata snapshot).
//!
//! Step 2 requirement:
//! - accept probabilistic execution arguments in `gen_settings` and
//!   `calibrate_settings` so Python users can opt-in.
//!
//! In this snapshot we *patch* an existing settings.json. Importantly, we write
//! settings in a shape compatible with `GraphSettings` (top-level mirrors +
//! `run_args` mirrors), so a subsequent CLI run can deserialize it.

use std::fs;
use std::path::Path;

use serde_json::{json, Value};

fn ensure_object<'a>(v: &'a mut Value, key: &str) -> &'a mut serde_json::Map<String, Value> {
    if !v.get(key).is_some_and(|x| x.is_object()) {
        v[key] = json!({});
    }
    v[key].as_object_mut().expect("must be object")
}

fn parse_ops_value(v: &Value) -> Option<Vec<String>> {
    if let Some(arr) = v.as_array() {
        let ops = arr
            .iter()
            .filter_map(|x| x.as_str())
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect::<Vec<_>>();
        return Some(ops);
    }

    if let Some(s) = v.as_str() {
        let ops = s
            .split(',')
            .map(|x| x.trim().to_string())
            .filter(|x| !x.is_empty())
            .collect::<Vec<_>>();
        return Some(ops);
    }

    None
}

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

    // IMPORTANT (borrow-checker):
    // We must not keep `&str` references into `root` alive across mutations of `root`.
    // So we always copy string values out of `root` into owned `String`s here.
    let existing_mode: String = root
        .get("execution_mode")
        .and_then(Value::as_str)
        .or_else(|| {
            root.get("run_args")
                .and_then(|ra| ra.get("execution_mode"))
                .and_then(Value::as_str)
        })
        .unwrap_or("exact")
        .to_string();

    // (A) execution_mode (top-level + run_args mirror)
    let final_mode: String = execution_mode
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or(existing_mode);

    let mode_is_prob = final_mode.eq_ignore_ascii_case("probabilistic");

    // Determine whether we should update probabilistic knobs.
    let should_update_prob =
        prob_ops.is_some() || prob_k.is_some() || prob_seed_mode.is_some() || mode_is_prob;

    // Pull existing probabilistic settings as fallbacks (owned) so we don't keep borrows into `root`.
    let existing_k: Option<u32> = root
        .get("prob_k")
        .and_then(Value::as_u64)
        .and_then(|x| u32::try_from(x).ok())
        .or_else(|| {
            root.get("probabilistic_settings")
                .and_then(|ps| ps.get("k_repetitions"))
                .and_then(Value::as_u64)
                .and_then(|x| u32::try_from(x).ok())
        })
        .or_else(|| {
            root.get("run_args")
                .and_then(|ra| ra.get("prob_k"))
                .and_then(Value::as_u64)
                .and_then(|x| u32::try_from(x).ok())
        });

    let existing_seed_mode: Option<String> = root
        .get("probabilistic_settings")
        .and_then(|ps| ps.get("seed_mode"))
        .and_then(Value::as_str)
        .map(|s| s.to_string())
        .or_else(|| {
            root.get("run_args")
                .and_then(|ra| ra.get("prob_seed_mode"))
                .and_then(Value::as_str)
                .map(|s| s.to_string())
        });

    let existing_ops: Option<Vec<String>> = root
        .get("prob_ops")
        .and_then(parse_ops_value)
        .or_else(|| {
            root.get("run_args")
                .and_then(|ra| ra.get("prob_ops"))
                .and_then(parse_ops_value)
        });

    // Update top-level execution_mode mirror.
    root["execution_mode"] = Value::String(final_mode.clone());

    // (B) probabilistic knobs (top-level + run_args mirrors)
    // Keep the previous behavior: if the caller asks for *any* prob settings (or sets
    // probabilistic mode), make sure the prob fields exist (with sensible defaults).
    let (k_final, seed_mode_final, ops_final) = if should_update_prob {
        let k = prob_k.or(existing_k).unwrap_or(40);

        let seed_mode = prob_seed_mode
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .or(existing_seed_mode)
            .unwrap_or_else(|| "public_seed".to_string());

        let mut ops = prob_ops.or(existing_ops).unwrap_or_default();
        ops = ops
            .into_iter()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect::<Vec<_>>();

        // Reasonable defaults if the caller enabled probabilistic mode but didn't specify ops.
        if ops.is_empty() && mode_is_prob {
            ops = vec!["MatMul".into(), "Gemm".into(), "Conv".into()];
        }

        // Top-level mirrors (auditable).
        root["prob_k"] = json!(k);
        root["prob_ops"] = json!(ops.clone());
        root["probabilistic_settings"] = json!({
            "k_repetitions": k,
            "seed_mode": seed_mode.clone(),
        });

        (Some(k), Some(seed_mode), Some(ops))
    } else {
        (None, None, None)
    };

    // Update run_args mirror in a tight scope so we don't keep a long-lived &mut borrow of `root`.
    {
        let run_args = ensure_object(&mut root, "run_args");

        run_args.insert("execution_mode".into(), Value::String(final_mode.clone()));

        if should_update_prob {
            let k = k_final.expect("k_final must be Some when should_update_prob");
            let seed_mode =
                seed_mode_final.expect("seed_mode_final must be Some when should_update_prob");
            let ops = ops_final.expect("ops_final must be Some when should_update_prob");

            run_args.insert("prob_k".into(), json!(k));
            run_args.insert("prob_seed_mode".into(), json!(seed_mode));
            run_args.insert("prob_ops".into(), json!(ops));
        }
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
    use std::process::Command;

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

    fn get_kwarg_ops(
        kwargs: Option<&Bound<'_, PyDict>>,
        key: &str,
    ) -> PyResult<Option<Vec<String>>> {
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
    #[pyfunction]
    #[pyo3(signature = (*args, **kwargs))]
    pub fn gen_settings(
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

    fn maybe_positional_string(args: &Bound<'_, PyTuple>, idx: usize) -> PyResult<Option<String>> {
        if args.len() <= idx {
            return Ok(None);
        }
        let s: String = args.get_item(idx)?.extract()?;
        let s = s.trim().to_string();
        if s.is_empty() {
            Ok(None)
        } else {
            Ok(Some(s))
        }
    }

    /// Python: get_srs(..., settings_path=..., srs_path=..., logrows=...)
    ///
    /// IMPORTANT:
    /// The "minimal" Dark-EZKL Python wheel (built with `--features python-bindings` only)
    /// does not embed the full proving system. To keep the wheel small (and avoid feature
    /// conflicts), we implement `get_srs()` by **shelling out to the `ezkl` CLI**.
    ///
    /// This fixes:
    ///   AttributeError: module 'ezkl' has no attribute 'get_srs'
    ///
    /// Expected kwargs:
    /// - settings_path: path to settings.json (optional if logrows is provided)
    /// - srs_path: output path (optional; if omitted, ezkl uses ~/.ezkl/srs/kzg{k}.srs)
    /// - logrows: u32 (optional; overrides settings_path if provided)
    #[pyfunction]
    #[pyo3(signature = (*args, **kwargs))]
    pub fn get_srs(args: &Bound<'_, PyTuple>, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<bool> {
        let settings_path = get_kwarg_string(kwargs, "settings_path")?
            .or_else(|| maybe_positional_string(args, 0).ok().flatten())
            .map(PathBuf::from);

        let srs_path = get_kwarg_string(kwargs, "srs_path")?
            .or_else(|| maybe_positional_string(args, 1).ok().flatten())
            .map(PathBuf::from);

        let logrows = get_kwarg_u32(kwargs, "logrows")?;

        if settings_path.is_none() && logrows.is_none() {
            return Err(PyValueError::new_err(
                "get_srs requires either settings_path=<path> or logrows=<u32>",
            ));
        }

        let mut cmd = Command::new("ezkl");
        cmd.arg("get-srs");

        if let Some(sp) = &settings_path {
            cmd.arg("--settings-path").arg(sp);
        }
        if let Some(op) = &srs_path {
            cmd.arg("--srs-path").arg(op);
        }
        if let Some(k) = logrows {
            cmd.arg("--logrows").arg(k.to_string());
        }

        let out = cmd.output().map_err(|e| {
            PyRuntimeError::new_err(format!(
                "failed to execute `ezkl get-srs` via subprocess: {e}\n\
                 Hint: ensure the ezkl CLI is installed and on PATH."
            ))
        })?;

        if out.status.success() {
            return Ok(true);
        }

        // Include both streams for easier debugging in notebook/runner environments.
        let stdout = String::from_utf8_lossy(&out.stdout);
        let stderr = String::from_utf8_lossy(&out.stderr);
        Err(PyRuntimeError::new_err(format!(
            "`ezkl get-srs` failed (exit={:?}).\n\nstdout:\n{stdout}\n\nstderr:\n{stderr}",
            out.status.code()
        )))
    }

    /// Helper to register functions onto a module.
    pub fn register(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add_function(wrap_pyfunction!(gen_settings, m)?)?;
        m.add_function(wrap_pyfunction!(calibrate_settings, m)?)?;
        m.add_function(wrap_pyfunction!(get_srs, m)?)?;
        Ok(())
    }
}

#[cfg(feature = "python-bindings")]
#[allow(unused_imports)]
pub use pyo3_bindings::{calibrate_settings, gen_settings, get_srs, register};

/// Non-Python builds: keep the module compiling without pyo3.
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
