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

    let existing_ops: Option<Vec<String>> =
        root.get("prob_ops").and_then(parse_ops_value).or_else(|| {
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

    fn get_kwarg_f64(kwargs: Option<&Bound<'_, PyDict>>, key: &str) -> PyResult<Option<f64>> {
        let Some(kwargs) = kwargs else {
            return Ok(None);
        };
        match kwargs.get_item(key)? {
            None => Ok(None),
            Some(v) => Ok(Some(v.extract::<f64>()?)),
        }
    }

    fn parse_i32_csv(raw: &str) -> PyResult<Vec<i32>> {
        raw.split(',')
            .map(|x| x.trim())
            .filter(|x| !x.is_empty())
            .map(|x| {
                x.parse::<i32>().map_err(|e| {
                    PyValueError::new_err(format!(
                        "expected a comma-separated list of integers, got '{}': {}",
                        raw, e
                    ))
                })
            })
            .collect()
    }

    fn parse_u32_csv(raw: &str) -> PyResult<Vec<u32>> {
        raw.split(',')
            .map(|x| x.trim())
            .filter(|x| !x.is_empty())
            .map(|x| {
                x.parse::<u32>().map_err(|e| {
                    PyValueError::new_err(format!(
                        "expected a comma-separated list of non-negative integers, got '{}': {}",
                        raw, e
                    ))
                })
            })
            .collect()
    }

    fn get_kwarg_i32_list(
        kwargs: Option<&Bound<'_, PyDict>>,
        key: &str,
    ) -> PyResult<Option<Vec<i32>>> {
        let Some(kwargs) = kwargs else {
            return Ok(None);
        };
        match kwargs.get_item(key)? {
            None => Ok(None),
            Some(v) => {
                if let Ok(val) = v.extract::<i32>() {
                    Ok(Some(vec![val]))
                } else if let Ok(vals) = v.extract::<Vec<i32>>() {
                    Ok(Some(vals))
                } else if let Ok(vals) = v.extract::<Vec<i64>>() {
                    let vals = vals
                        .into_iter()
                        .map(|x| {
                            i32::try_from(x).map_err(|_| {
                                PyValueError::new_err(format!(
                                    "{key} contains a value outside the i32 range"
                                ))
                            })
                        })
                        .collect::<PyResult<Vec<_>>>()?;
                    Ok(Some(vals))
                } else if let Ok(s) = v.extract::<String>() {
                    Ok(Some(parse_i32_csv(&s)?))
                } else {
                    Err(PyValueError::new_err(format!(
                        "{key} must be an int, a comma-separated string, or a list of ints"
                    )))
                }
            }
        }
    }

    fn get_kwarg_u32_list(
        kwargs: Option<&Bound<'_, PyDict>>,
        key: &str,
    ) -> PyResult<Option<Vec<u32>>> {
        let Some(kwargs) = kwargs else {
            return Ok(None);
        };
        match kwargs.get_item(key)? {
            None => Ok(None),
            Some(v) => {
                if let Ok(val) = v.extract::<u32>() {
                    Ok(Some(vec![val]))
                } else if let Ok(val) = v.extract::<u64>() {
                    Ok(Some(vec![u32::try_from(val).map_err(|_| {
                        PyValueError::new_err(format!(
                            "{key} contains a value outside the u32 range"
                        ))
                    })?]))
                } else if let Ok(vals) = v.extract::<Vec<u32>>() {
                    Ok(Some(vals))
                } else if let Ok(vals) = v.extract::<Vec<u64>>() {
                    let vals = vals
                        .into_iter()
                        .map(|x| {
                            u32::try_from(x).map_err(|_| {
                                PyValueError::new_err(format!(
                                    "{key} contains a value outside the u32 range"
                                ))
                            })
                        })
                        .collect::<PyResult<Vec<_>>>()?;
                    Ok(Some(vals))
                } else if let Ok(s) = v.extract::<String>() {
                    Ok(Some(parse_u32_csv(&s)?))
                } else {
                    Err(PyValueError::new_err(format!(
                        "{key} must be an int, a comma-separated string, or a list of ints"
                    )))
                }
            }
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
        // Prefer explicit keywords.
        if let Some(s) = get_kwarg_string(kwargs, "settings_path")? {
            return Ok(PathBuf::from(s));
        }
        if let Some(s) = get_kwarg_string(kwargs, "settings")? {
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
            "settings_path not provided (expected kwarg settings_path/settings or positional arg)",
        ))
    }

    /// Python: gen_settings(..., execution_mode=..., prob_ops=..., prob_k=..., prob_seed_mode=...)
    #[pyfunction]
    #[pyo3(signature = (*args, **kwargs))]
    pub fn gen_settings(
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<bool> {
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

        // IMPORTANT:
        // The benchmark runner treats falsy returns as failure.
        // Returning `True` here avoids a successful settings patch being interpreted as `False`.
        Ok(true)
    }

    /// Python: calibrate_settings(..., execution_mode=..., prob_ops=..., prob_k=..., prob_seed_mode=...)
    ///
    /// IMPORTANT:
    /// The minimal Dark-EZKL Python wheel does not embed the full proving/calibration engine.
    /// To make this API behave like users expect, we:
    ///   1) patch the settings.json with probabilistic args (if provided), then
    ///   2) shell out to the real `ezkl calibrate-settings` CLI, and
    ///   3) return `True` iff the subprocess succeeded.
    ///
    /// This fixes benchmark runners that do:
    ///   ok = ezkl.calibrate_settings(...)
    ///   if not ok: raise RuntimeError(...)
    #[pyfunction]
    #[pyo3(signature = (*args, **kwargs))]
    pub fn calibrate_settings(
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<bool> {
        let settings_path = if let Some(s) =
            get_any_kwarg_string(kwargs, &["settings_path", "settings", "output"])?
        {
            Some(PathBuf::from(s))
        } else if args.len() >= 3 {
            maybe_positional_string(args, 2)?.map(PathBuf::from)
        } else if args.len() >= 2 {
            maybe_positional_string(args, 1)?.map(PathBuf::from)
        } else if args.len() == 1 {
            maybe_positional_string(args, 0)?.map(PathBuf::from)
        } else {
            None
        };

        let execution_mode = get_kwarg_string(kwargs, "execution_mode")?;
        let prob_ops = get_kwarg_ops(kwargs, "prob_ops")?;
        let prob_k = get_kwarg_u32(kwargs, "prob_k")?;
        let prob_seed_mode = get_kwarg_string(kwargs, "prob_seed_mode")?;

        // First patch probabilistic settings into settings.json so the CLI calibration
        // sees the intended execution mode / Freivalds parameters.
        if let Some(settings_path) = settings_path.as_deref() {
            update_settings_file(
                settings_path,
                execution_mode.as_deref(),
                prob_ops,
                prob_k,
                prob_seed_mode.as_deref(),
            )
            .map_err(PyRuntimeError::new_err)?;
        }

        let model = get_any_kwarg_string(kwargs, &["model", "model_path", "onnx_path"])?.or(
            if args.len() >= 1 {
                maybe_positional_string(args, 0)?
            } else {
                None
            },
        );

        let data = get_any_kwarg_string(
            kwargs,
            &[
                "data",
                "data_path",
                "input",
                "input_path",
                "calibration_data",
            ],
        )?
        .or(if args.len() >= 3 {
            maybe_positional_string(args, 1)?
        } else {
            None
        });

        let target = get_any_kwarg_string(kwargs, &["target"])?.or(if args.len() >= 4 {
            maybe_positional_string(args, 3)?
        } else {
            None
        });

        let lookup_safety_margin = get_kwarg_f64(kwargs, "lookup_safety_margin")?;
        let scales = get_kwarg_i32_list(kwargs, "scales")?;
        let scale_rebase_multiplier = get_kwarg_u32_list(kwargs, "scale_rebase_multiplier")?
            .or(get_kwarg_u32_list(kwargs, "scale_rebase_multipliers")?);
        let max_logrows = get_kwarg_u32(kwargs, "max_logrows")?;

        // Backwards-compatible patch-only mode:
        // if the caller only supplied a settings path, preserve the old behavior and
        // simply return True after updating the JSON file.
        let (Some(model), Some(data), Some(settings_path)) = (model, data, settings_path) else {
            return Ok(true);
        };

        let mut cli_args = vec![
            "calibrate-settings".into(),
            "-M".into(),
            model,
            "-D".into(),
            data,
            "-O".into(),
            settings_path.display().to_string(),
        ];

        if let Some(target) = target
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
        {
            cli_args.push("--target".into());
            cli_args.push(target);
        }

        if let Some(lookup_safety_margin) = lookup_safety_margin {
            cli_args.push("--lookup-safety-margin".into());
            cli_args.push(lookup_safety_margin.to_string());
        }

        if let Some(scales) = scales.filter(|v| !v.is_empty()) {
            cli_args.push("--scales".into());
            cli_args.push(
                scales
                    .into_iter()
                    .map(|x| x.to_string())
                    .collect::<Vec<_>>()
                    .join(","),
            );
        }

        if let Some(scale_rebase_multiplier) = scale_rebase_multiplier.filter(|v| !v.is_empty()) {
            cli_args.push("--scale-rebase-multiplier".into());
            cli_args.push(
                scale_rebase_multiplier
                    .into_iter()
                    .map(|x| x.to_string())
                    .collect::<Vec<_>>()
                    .join(","),
            );
        }

        if let Some(max_logrows) = max_logrows {
            cli_args.push("--max-logrows".into());
            cli_args.push(max_logrows.to_string());
        }

        run_ezkl_cli_success(&cli_args)
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

    fn get_any_kwarg_string(
        kwargs: Option<&Bound<'_, PyDict>>,
        keys: &[&str],
    ) -> PyResult<Option<String>> {
        for key in keys {
            if let Some(v) = get_kwarg_string(kwargs, key)? {
                return Ok(Some(v));
            }
        }
        Ok(None)
    }

    fn maybe_positional_string_any(
        args: &Bound<'_, PyTuple>,
        idxs: &[usize],
    ) -> PyResult<Option<String>> {
        for idx in idxs {
            if let Some(v) = maybe_positional_string(args, *idx)? {
                return Ok(Some(v));
            }
        }
        Ok(None)
    }

    fn get_kwarg_bool(kwargs: Option<&Bound<'_, PyDict>>, key: &str) -> PyResult<Option<bool>> {
        let Some(kwargs) = kwargs else {
            return Ok(None);
        };
        match kwargs.get_item(key)? {
            None => Ok(None),
            Some(v) => Ok(Some(v.extract::<bool>()?)),
        }
    }

    fn run_ezkl_cli(args: &[String]) -> PyResult<String> {
        let out = Command::new("ezkl").args(args).output().map_err(|e| {
            PyRuntimeError::new_err(format!(
                "failed to execute `ezkl {}` via subprocess: {e}\n\
                     Hint: ensure the ezkl CLI is installed and on PATH.",
                args.join(" ")
            ))
        })?;

        let stdout = String::from_utf8_lossy(&out.stdout).to_string();
        let stderr = String::from_utf8_lossy(&out.stderr).to_string();

        if out.status.success() {
            return Ok(stdout.trim().to_string());
        }

        Err(PyRuntimeError::new_err(format!(
            "`ezkl {}` failed (exit={:?}).\n\nstdout:\n{}\n\nstderr:\n{}",
            args.join(" "),
            out.status.code(),
            stdout,
            stderr
        )))
    }

    fn run_ezkl_cli_success(args: &[String]) -> PyResult<bool> {
        // IMPORTANT:
        // A number of real ezkl CLI subcommands return structured JSON on stdout on success
        // (e.g. gen-witness returns a witness payload, prove returns a proof payload).
        //
        // The previous wrapper incorrectly treated "non-empty stdout" as failure unless it
        // literally equaled "true", which caused successful commands to return `False`
        // to Python and break benchmark pipelines.
        //
        // Success is determined by the subprocess exit code. `run_ezkl_cli()` already raises
        // on non-zero exit status, so if we get here the command succeeded.
        let _ = run_ezkl_cli(args)?;
        Ok(true)
    }

    #[pyfunction]
    #[pyo3(signature = (*args, **kwargs))]
    pub fn compile_circuit(
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<bool> {
        let model = get_any_kwarg_string(kwargs, &["model", "onnx_path"])?
            .or(maybe_positional_string_any(args, &[0])?)
            .ok_or_else(|| PyValueError::new_err("compile_circuit requires model=<path>"))?;

        let compiled_circuit = get_any_kwarg_string(
            kwargs,
            &[
                "compiled_circuit",
                "compiled_model",
                "compiled_path",
                "output",
            ],
        )?
        .or(maybe_positional_string_any(args, &[1])?)
        .ok_or_else(|| {
            PyValueError::new_err(
                "compile_circuit requires compiled_circuit=<path> (or compiled_model/output)",
            )
        })?;

        let settings_path = get_any_kwarg_string(kwargs, &["settings_path", "settings"])?
            .or(maybe_positional_string_any(args, &[2])?)
            .ok_or_else(|| {
                PyValueError::new_err("compile_circuit requires settings_path=<path>")
            })?;

        run_ezkl_cli_success(&vec![
            "compile-circuit".into(),
            "-M".into(),
            model,
            "--compiled-circuit".into(),
            compiled_circuit,
            "-S".into(),
            settings_path,
        ])
    }

    #[pyfunction]
    #[pyo3(signature = (*args, **kwargs))]
    pub fn setup(args: &Bound<'_, PyTuple>, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<bool> {
        let compiled_circuit = get_any_kwarg_string(
            kwargs,
            &[
                "compiled_circuit",
                "compiled_model",
                "model",
                "compiled_path",
            ],
        )?
        .or(maybe_positional_string_any(args, &[0])?)
        .ok_or_else(|| PyValueError::new_err("setup requires compiled_circuit=<path>"))?;

        let vk_path = get_any_kwarg_string(kwargs, &["vk_path", "vk"])?
            .or(maybe_positional_string_any(args, &[1])?)
            .ok_or_else(|| PyValueError::new_err("setup requires vk_path=<path>"))?;

        let pk_path = get_any_kwarg_string(kwargs, &["pk_path", "pk"])?
            .or(maybe_positional_string_any(args, &[2])?)
            .ok_or_else(|| PyValueError::new_err("setup requires pk_path=<path>"))?;

        let srs_path = get_any_kwarg_string(kwargs, &["srs_path", "srs"])?
            .or(maybe_positional_string_any(args, &[3])?);

        let witness = get_any_kwarg_string(kwargs, &["witness", "witness_path"])?
            .or(maybe_positional_string_any(args, &[4])?);

        let mut cli_args = vec![
            "setup".into(),
            "-M".into(),
            compiled_circuit,
            "--vk-path".into(),
            vk_path,
            "--pk-path".into(),
            pk_path,
        ];

        if let Some(srs_path) = srs_path {
            cli_args.push("--srs-path".into());
            cli_args.push(srs_path);
        }

        if let Some(witness) = witness {
            cli_args.push("-W".into());
            cli_args.push(witness);
        }

        run_ezkl_cli_success(&cli_args)
    }

    #[pyfunction]
    #[pyo3(signature = (*args, **kwargs))]
    pub fn gen_witness(
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<bool> {
        let compiled_circuit = get_any_kwarg_string(
            kwargs,
            &[
                "compiled_circuit",
                "compiled_model",
                "model",
                "compiled_path",
            ],
        )?
        .or(maybe_positional_string_any(args, &[0])?)
        .ok_or_else(|| PyValueError::new_err("gen_witness requires compiled_circuit=<path>"))?;

        let data = get_any_kwarg_string(kwargs, &["data", "data_path", "input", "input_path"])?
            .or(maybe_positional_string_any(args, &[1])?)
            .ok_or_else(|| PyValueError::new_err("gen_witness requires data=<path-or-json>"))?;

        let output = get_any_kwarg_string(kwargs, &["output", "witness", "witness_path"])?
            .or(maybe_positional_string_any(args, &[2])?)
            .ok_or_else(|| PyValueError::new_err("gen_witness requires output=<path>"))?;

        let vk_path = get_any_kwarg_string(kwargs, &["vk_path", "vk"])?
            .or(maybe_positional_string_any(args, &[3])?);

        let srs_path = get_any_kwarg_string(kwargs, &["srs_path", "srs"])?
            .or(maybe_positional_string_any(args, &[4])?);

        let mut cli_args = vec![
            "gen-witness".into(),
            "-M".into(),
            compiled_circuit,
            "-D".into(),
            data,
            "--output".into(),
            output,
        ];

        if let Some(vk_path) = vk_path {
            cli_args.push("--vk-path".into());
            cli_args.push(vk_path);
        }
        if let Some(srs_path) = srs_path {
            cli_args.push("--srs-path".into());
            cli_args.push(srs_path);
        }

        run_ezkl_cli_success(&cli_args)
    }

    #[pyfunction]
    #[pyo3(signature = (*args, **kwargs))]
    pub fn prove(args: &Bound<'_, PyTuple>, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<bool> {
        let compiled_circuit = get_any_kwarg_string(
            kwargs,
            &[
                "compiled_circuit",
                "compiled_model",
                "model",
                "compiled_path",
            ],
        )?
        .or(maybe_positional_string_any(args, &[0])?)
        .ok_or_else(|| PyValueError::new_err("prove requires compiled_circuit=<path>"))?;

        let witness = get_any_kwarg_string(kwargs, &["witness", "witness_path"])?
            .or(maybe_positional_string_any(args, &[1])?)
            .ok_or_else(|| PyValueError::new_err("prove requires witness=<path>"))?;

        let pk_path = get_any_kwarg_string(kwargs, &["pk_path", "pk"])?
            .or(maybe_positional_string_any(args, &[2])?)
            .ok_or_else(|| PyValueError::new_err("prove requires pk_path=<path>"))?;

        let proof_path = get_any_kwarg_string(kwargs, &["proof_path", "proof"])?
            .or(maybe_positional_string_any(args, &[3])?)
            .ok_or_else(|| PyValueError::new_err("prove requires proof_path=<path>"))?;

        let srs_path = get_any_kwarg_string(kwargs, &["srs_path", "srs"])?
            .or(maybe_positional_string_any(args, &[4])?);

        let check_mode = get_any_kwarg_string(kwargs, &["check_mode"])?
            .or(maybe_positional_string_any(args, &[5])?);

        let mut cli_args = vec![
            "prove".into(),
            "-M".into(),
            compiled_circuit,
            "--witness".into(),
            witness,
            "--pk-path".into(),
            pk_path,
            "--proof-path".into(),
            proof_path,
        ];

        if let Some(srs_path) = srs_path {
            cli_args.push("--srs-path".into());
            cli_args.push(srs_path);
        }
        if let Some(check_mode) = check_mode {
            cli_args.push("--check-mode".into());
            cli_args.push(check_mode);
        }

        run_ezkl_cli_success(&cli_args)
    }

    #[pyfunction]
    #[pyo3(signature = (*args, **kwargs))]
    pub fn verify(args: &Bound<'_, PyTuple>, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<bool> {
        let proof_path = get_any_kwarg_string(kwargs, &["proof_path", "proof"])?
            .or(maybe_positional_string_any(args, &[0])?)
            .ok_or_else(|| PyValueError::new_err("verify requires proof_path=<path>"))?;

        let settings_path = get_any_kwarg_string(kwargs, &["settings_path", "settings"])?
            .or(maybe_positional_string_any(args, &[1])?)
            .ok_or_else(|| PyValueError::new_err("verify requires settings_path=<path>"))?;

        let vk_path = get_any_kwarg_string(kwargs, &["vk_path", "vk"])?
            .or(maybe_positional_string_any(args, &[2])?)
            .ok_or_else(|| PyValueError::new_err("verify requires vk_path=<path>"))?;

        let srs_path = get_any_kwarg_string(kwargs, &["srs_path", "srs"])?
            .or(maybe_positional_string_any(args, &[3])?);

        let reduced_srs = get_kwarg_bool(kwargs, "reduced_srs")?;

        let mut cli_args = vec![
            "verify".into(),
            "--proof-path".into(),
            proof_path,
            "--settings-path".into(),
            settings_path,
            "--vk-path".into(),
            vk_path,
        ];

        if let Some(srs_path) = srs_path {
            cli_args.push("--srs-path".into());
            cli_args.push(srs_path);
        }
        if reduced_srs.unwrap_or(false) {
            cli_args.push("--reduced-srs".into());
        }

        let out = run_ezkl_cli(&cli_args)?;
        let s = out.trim().to_ascii_lowercase();
        Ok(s.is_empty() || s == "true" || s == "\"true\"")
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
    pub fn get_srs(
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<bool> {
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
        m.add_function(wrap_pyfunction!(compile_circuit, m)?)?;
        m.add_function(wrap_pyfunction!(setup, m)?)?;
        m.add_function(wrap_pyfunction!(gen_witness, m)?)?;
        m.add_function(wrap_pyfunction!(prove, m)?)?;
        m.add_function(wrap_pyfunction!(verify, m)?)?;
        m.add_function(wrap_pyfunction!(get_srs, m)?)?;
        Ok(())
    }
}

#[cfg(feature = "python-bindings")]
#[allow(unused_imports)]
pub use pyo3_bindings::{
    calibrate_settings, compile_circuit, gen_settings, gen_witness, get_srs, prove, register,
    setup, verify,
};

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
