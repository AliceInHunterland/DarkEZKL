#[cfg(all(feature = "colored_json", not(target_arch = "wasm32")))]
use colored_json::prelude::ToColoredJson;
#[derive(Debug, Clone)]
#[cfg_attr(
    all(feature = "tabled", not(target_arch = "wasm32")),
    derive(tabled::Tabled)
)]
/// Accuracy tearsheet
pub struct AccuracyResults {
    mean_error: f32,
    median_error: f32,
    max_error: f32,
    min_error: f32,
    mean_abs_error: f32,
    median_abs_error: f32,
    max_abs_error: f32,
    min_abs_error: f32,
    mean_squared_error: f32,
    mean_percent_error: f32,
    mean_abs_percent_error: f32,
}

impl AccuracyResults {
    /// Create a new accuracy results struct
    pub fn new(
        mut original_preds: Vec<crate::tensor::Tensor<f32>>,
        mut calibrated_preds: Vec<crate::tensor::Tensor<f32>>,
    ) -> Result<Self, EZKLError> {
        use std::cmp::Ordering;

        let mut errors = vec![];
        let mut abs_errors = vec![];
        let mut squared_errors = vec![];
        let mut percentage_errors = vec![];
        let mut abs_percentage_errors = vec![];

        for (original, calibrated) in original_preds.iter_mut().zip(calibrated_preds.iter_mut()) {
            original.flatten();
            calibrated.flatten();
            let error = (original.clone() - calibrated.clone())?;
            let abs_error = error.map(|x| x.abs());
            let squared_error = error.map(|x| x.powi(2));
            let percentage_error = error.enum_map(|i, x| {
                // if everything is 0 then we can't divide by 0 so we just return 0
                let res = if original[i] == 0.0 && x == 0.0 {
                    0.0
                } else {
                    x / original[i]
                };
                Ok::<f32, crate::tensor::TensorError>(res)
            })?;
            let abs_percentage_error = percentage_error.map(|x| x.abs());

            errors.extend(error);
            abs_errors.extend(abs_error);
            squared_errors.extend(squared_error);
            percentage_errors.extend(percentage_error);
            abs_percentage_errors.extend(abs_percentage_error);
        }

        if errors.is_empty()
            || abs_errors.is_empty()
            || squared_errors.is_empty()
            || percentage_errors.is_empty()
            || abs_percentage_errors.is_empty()
        {
            return Err(EZKLError::msg(
                "cannot compute accuracy: empty predictions (no elements after flattening)",
            ));
        }

        // Sort for proper median computation.
        errors.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
        abs_errors.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));

        let median_from_sorted = |v: &[f32]| -> f32 {
            let n = v.len();
            if n == 0 {
                return 0.0;
            }
            if n % 2 == 1 {
                v[n / 2]
            } else {
                (v[n / 2 - 1] + v[n / 2]) / 2.0
            }
        };

        let mean_percent_error =
            percentage_errors.iter().sum::<f32>() / percentage_errors.len() as f32;
        let mean_abs_percent_error =
            abs_percentage_errors.iter().sum::<f32>() / abs_percentage_errors.len() as f32;
        let mean_error = errors.iter().sum::<f32>() / errors.len() as f32;
        let median_error = median_from_sorted(&errors);
        let max_error = *errors
            .iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal))
            .unwrap();
        let min_error = *errors
            .iter()
            .min_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal))
            .unwrap();

        let mean_abs_error = abs_errors.iter().sum::<f32>() / abs_errors.len() as f32;
        let median_abs_error = median_from_sorted(&abs_errors);
        let max_abs_error = *abs_errors
            .iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal))
            .unwrap();
        let min_abs_error = *abs_errors
            .iter()
            .min_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal))
            .unwrap();

        let mean_squared_error = squared_errors.iter().sum::<f32>() / squared_errors.len() as f32;

        Ok(Self {
            mean_error,
            median_error,
            max_error,
            min_error,
            mean_abs_error,
            median_abs_error,
            max_abs_error,
            min_abs_error,
            mean_squared_error,
            mean_percent_error,
            mean_abs_percent_error,
        })
    }
}

/// Calibrate the circuit parameters to a given a dataset
#[allow(trivial_casts)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn calibrate(
    model_path: PathBuf,
    data: String,
    settings_path: PathBuf,
    target: CalibrationTarget,
    lookup_safety_margin: f64,
    scales: Option<Vec<crate::Scale>>,
    scale_rebase_multiplier: Vec<u32>,
    max_logrows: Option<u32>,
) -> Result<GraphSettings, EZKLError> {
    use std::collections::HashMap;

    use crate::fieldutils::IntegerRep;

    let data = GraphData::from_str(&data)?;
    // load the pre-generated settings
    let settings = GraphSettings::load(&settings_path)?;
    // now retrieve the run args
    // we load the model to get the input and output shapes

    let model = Model::from_run_args(&settings.run_args, &model_path)?;

    let input_shapes = model.graph.input_shapes()?;

    let chunks = data.split_into_batches(input_shapes)?;
    log::info!("num calibration batches: {}", chunks.len());

    log::debug!("running onnx predictions...");
    let original_predictions = Model::run_onnx_predictions(
        &settings.run_args,
        &model_path,
        &chunks,
        model.graph.input_shapes()?,
    )?;

    // Default scale search band:
    // - If the caller passes `--scales`, we only try those.
    // - Otherwise:
    //   * Always include the scales already present in settings.json (common in benchmarks).
    //   * For `--target accuracy`, also include the historic (11..14) window.
    //   * For `--target resources`, do NOT expand the search window unless scales appear "unset" (0,0),
    //     to avoid an O(|range|^2) blow-up on large models.
    let mut range: Vec<crate::Scale> = if let Some(scales) = scales {
        scales
    } else {
        let mut r = vec![settings.run_args.input_scale, settings.run_args.param_scale];

        let scales_unset = settings.run_args.input_scale == 0 && settings.run_args.param_scale == 0;
        if matches!(target, CalibrationTarget::Accuracy) || scales_unset {
            r.extend(11..14);
        }

        r
    };
    range.sort();
    range.dedup();

    let mut found_params: Vec<GraphSettings> = vec![];

    // 2 x 2 grid
    let range_grid = range
        .iter()
        .cartesian_product(range.iter())
        .map(|(a, b)| (*a, *b))
        .collect::<Vec<(crate::Scale, crate::Scale)>>();

    // remove all entries where input_scale > param_scale
    let mut range_grid = range_grid
        .into_iter()
        .filter(|(a, b)| a <= b)
        .collect::<Vec<(crate::Scale, crate::Scale)>>();

    // if all integers
    let all_scale_0 = model
        .graph
        .get_input_types()?
        .iter()
        .all(|t| t.is_integer());
    if all_scale_0 {
        // set all a values to 0 then dedup
        range_grid = range_grid
            .iter()
            .map(|(_, b)| (0, *b))
            .sorted()
            .dedup()
            .collect::<Vec<(crate::Scale, crate::Scale)>>();
    }

    let range_grid = range_grid
        .iter()
        .cartesian_product(scale_rebase_multiplier.iter())
        .map(|(a, b)| (*a, *b))
        .collect::<Vec<((crate::Scale, crate::Scale), u32)>>();

    let total_cases = range_grid.len();

    eprintln!(
        "[calibrate-settings] target={}, calibration_batches={}, scale_pairs_to_try={}. \
         Hint: pass --scales <csv> to restrict scale search (faster on large models).",
        target,
        chunks.len(),
        total_cases
    );

    let mut forward_pass_res = HashMap::new();

    let pb = init_bar(total_cases as u64);
    pb.set_message("calibrating...");

    let mut num_failed = 0;
    let mut num_passed = 0;
    let mut failure_reasons = vec![];

    // If decomposition is too small (base^legs), calibration can fail at layout time with:
    //   "[tensor] decomposition error: integer X is too large to be represented by base 128 and n <legs>"
    //
    // During calibration we can safely auto-increase `decomp_legs` (it only affects proving resources)
    // until the forward pass succeeds.
    const AUTO_DECOMP_LEGS_MAX: usize = 8;
    let is_decomp_overflow = |s: &str| -> bool {
        s.contains("decomposition error") && s.contains("too large to be represented")
    };

    // If `calc_min_logrows` fails with `ExtendedKTooLarge`, the circuit is too "tall" (too many rows)
    // to fit within the max allowed k (typically 26 on BN254 when accounting for quotient polynomial
    // degree). A pragmatic way to reduce required rows is to widen the layout by increasing
    // `num_inner_cols` (more parallelism across columns).
    //
    // We keep this bounded to avoid accidentally allocating an extreme number of advice columns.
    const AUTO_NUM_INNER_COLS_MAX_DEFAULT: usize = 64;
    let auto_num_inner_cols_max: usize = std::env::var("EZKL_AUTO_NUM_INNER_COLS_MAX")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(AUTO_NUM_INNER_COLS_MAX_DEFAULT)
        .max(1);

    // This matches the effective cap enforced inside `GraphCircuit::calc_min_logrows`
    // (MAX_PUBLIC_SRS min user max_logrows).
    let effective_max_k: u32 = max_logrows
        .unwrap_or(crate::graph::MAX_PUBLIC_SRS)
        .min(crate::graph::MAX_PUBLIC_SRS);

    for (case_idx, ((input_scale, param_scale), scale_rebase_multiplier)) in
        range_grid.into_iter().enumerate()
    {
        let case_no = case_idx + 1;

        // One line per case so users can see progress even if stdout is captured.
        eprintln!(
            "[calibrate-settings] case {}/{}: input_scale={}, param_scale={}, rebase_multiplier={}, decomp_base={}, decomp_legs={}, num_inner_cols={}",
            case_no,
            total_cases,
            input_scale,
            param_scale,
            scale_rebase_multiplier,
            settings.run_args.decomp_base,
            settings.run_args.decomp_legs,
            settings.run_args.num_inner_cols
        );

        pb.set_message(format!(
            "i-scale: {}, p-scale: {}, rebase-(x): {}, fail: {}, pass: {}",
            input_scale.to_string(),
            param_scale.to_string(),
            scale_rebase_multiplier.to_string(),
            num_failed.to_string(),
            num_passed.to_string()
        ));

        let key = (input_scale, param_scale, scale_rebase_multiplier);
        forward_pass_res.insert(key, vec![]);

        let mut local_run_args = RunArgs {
            input_scale,
            param_scale,
            scale_rebase_multiplier,
            lookup_range: (IntegerRep::MIN, IntegerRep::MAX),
            ..settings.run_args.clone()
        };

        // Keep stdout clean (the CLI prints a JSON blob on stdout for machine parsing).
        // Set `EZKL_GAG_STDOUT=0` if you want to see stdout from underlying libraries.
        #[cfg(unix)]
        let _stdout_gag: Option<gag::Gag> = {
            let raw = std::env::var("EZKL_GAG_STDOUT").unwrap_or_else(|_| "1".to_string());
            let raw_lc = raw.trim().to_lowercase();
            let enabled = !matches!(raw_lc.as_str(), "0" | "false" | "no");
            if enabled {
                gag::Gag::stdout().ok()
            } else {
                None
            }
        };

        // Try a forward pass. If we hit a decomposition overflow, retry with more `decomp_legs`.
        let start_legs = local_run_args.decomp_legs;
        let max_legs = std::cmp::max(start_legs, AUTO_DECOMP_LEGS_MAX);

        let mut circuit: Option<GraphCircuit> = None;
        let mut candidate_err: Option<String> = None;

        for legs in start_legs..=max_legs {
            local_run_args.decomp_legs = legs;

            // Reset results for this key in case we're retrying.
            if let Some(v) = forward_pass_res.get_mut(&key) {
                v.clear();
            }

            let mut attempt_circuit = match GraphCircuit::from_run_args(&local_run_args, &model_path)
            {
                Ok(c) => c,
                Err(e) => {
                    let msg = format!("circuit creation from run args failed: {}", e);

                    if is_decomp_overflow(&msg) && legs < max_legs {
                        eprintln!(
                            "[calibrate-settings] case {}/{}: decomposition overflow at decomp_legs={} during circuit creation; retrying with decomp_legs={}",
                            case_no,
                            total_cases,
                            legs,
                            legs + 1
                        );
                        continue;
                    }

                    candidate_err = Some(msg);
                    break;
                }
            };

            // Forward pass through all calibration batches.
            let fwd_start = Instant::now();
            let mut last_heartbeat = Instant::now();
            let mut forward_err: Option<String> = None;

            for (batch_idx, chunk) in chunks.iter().enumerate() {
                if last_heartbeat.elapsed() >= Duration::from_secs(30) {
                    eprintln!(
                        "[calibrate-settings] case {}/{} still running (elapsed={}s) — batch {}/{} (decomp_legs={})",
                        case_no,
                        total_cases,
                        fwd_start.elapsed().as_secs(),
                        batch_idx + 1,
                        chunks.len(),
                        legs
                    );
                    last_heartbeat = Instant::now();
                }

                let mut data = match attempt_circuit.load_graph_input(chunk) {
                    Ok(d) => d,
                    Err(e) => {
                        forward_err = Some(format!("failed to load circuit inputs: {}", e));
                        break;
                    }
                };

                let forward_res = match attempt_circuit.forward::<KZGCommitmentScheme<Bn256>>(
                    &mut data,
                    None,
                    None,
                    RegionSettings::all_true(local_run_args.decomp_base, local_run_args.decomp_legs),
                ) {
                    Ok(r) => r,
                    Err(e) => {
                        forward_err = Some(format!("failed to forward: {}", e));
                        break;
                    }
                };

                match forward_pass_res.get_mut(&key) {
                    Some(v) => v.push(forward_res),
                    None => {
                        forward_err = Some("key not found".to_string());
                        break;
                    }
                }
            }

            if let Some(e) = forward_err {
                // Decomposition overflow: retry with larger legs.
                if is_decomp_overflow(&e) && legs < max_legs {
                    eprintln!(
                        "[calibrate-settings] case {}/{}: decomposition overflow at decomp_legs={}; retrying with decomp_legs={}",
                        case_no,
                        total_cases,
                        legs,
                        legs + 1
                    );
                    continue;
                }

                candidate_err = Some(e);
                break;
            }

            // Success.
            if legs != start_legs {
                eprintln!(
                    "[calibrate-settings] case {}/{}: succeeded after increasing decomp_legs {} -> {}",
                    case_no, total_cases, start_legs, legs
                );
            }

            circuit = Some(attempt_circuit);
            candidate_err = None;
            break;
        }

        if let Some(e) = candidate_err {
            log::error!("calibration candidate failed: {:?}", e);
            pb.inc(1);
            num_failed += 1;
            failure_reasons.push(format!(
                "i-scale: {}, p-scale: {}, rebase-(x): {}, inner_cols: {}, decomp_legs: {}, reason: {}",
                input_scale.to_string(),
                param_scale.to_string(),
                scale_rebase_multiplier.to_string(),
                local_run_args.num_inner_cols.to_string(),
                local_run_args.decomp_legs.to_string(),
                e
            ));
            continue;
        }

        let mut circuit = match circuit {
            Some(c) => c,
            None => {
                // Should be unreachable, but keep a clear error if it ever happens.
                let e = "internal error: calibration succeeded but circuit is None".to_string();
                log::error!("{}", e);
                pb.inc(1);
                num_failed += 1;
                failure_reasons.push(format!(
                    "i-scale: {}, p-scale: {}, rebase-(x): {}, inner_cols: {}, decomp_legs: {}, reason: {}",
                    input_scale.to_string(),
                    param_scale.to_string(),
                    scale_rebase_multiplier.to_string(),
                    local_run_args.num_inner_cols.to_string(),
                    local_run_args.decomp_legs.to_string(),
                    e
                ));
                continue;
            }
        };

        let result = forward_pass_res
            .get(&key)
            .ok_or_else(|| EZKLError::msg("key not found"))?;

        let min_lookup_range = result
            .iter()
            .map(|x| x.min_lookup_inputs)
            .min()
            .unwrap_or(0);

        let max_lookup_range = result
            .iter()
            .map(|x| x.max_lookup_inputs)
            .max()
            .unwrap_or(0);

        let max_range_size = result.iter().map(|x| x.max_range_size).max().unwrap_or(0);

        let observed_lookup_range = (min_lookup_range, max_lookup_range);
        let observed_max_range_abs = max_range_size;

        // Try to compute the minimal viable `logrows` for this candidate.
        // If it fails due to k being too large, retry by widening the circuit (num_inner_cols *= 2).
        let start_inner_cols = local_run_args.num_inner_cols;
        let mut last_required_k: Option<u32> = None;

        let res = loop {
            let res = circuit.calc_min_logrows(
                observed_lookup_range,
                observed_max_range_abs,
                max_logrows,
                lookup_safety_margin,
            );

            match res {
                Ok(()) => break Ok(()),
                Err(crate::graph::errors::GraphError::ExtendedKTooLarge(required_k)) => {
                    // Stop if we're already at (or above) our widening cap.
                    if local_run_args.num_inner_cols >= auto_num_inner_cols_max {
                        break Err(crate::graph::errors::GraphError::ExtendedKTooLarge(required_k));
                    }

                    // If required_k is not improving across retries, stop to avoid infinite loops.
                    if last_required_k == Some(required_k) {
                        break Err(crate::graph::errors::GraphError::ExtendedKTooLarge(required_k));
                    }
                    last_required_k = Some(required_k);

                    let prev_cols = local_run_args.num_inner_cols;
                    let next_cols =
                        std::cmp::min(prev_cols.saturating_mul(2), auto_num_inner_cols_max);

                    eprintln!(
                        "[calibrate-settings] case {}/{}: required logrows {} exceeds max {}; retrying with num_inner_cols {} -> {}",
                        case_no,
                        total_cases,
                        required_k,
                        effective_max_k,
                        prev_cols,
                        next_cols
                    );

                    local_run_args.num_inner_cols = next_cols;

                    // Rebuild the circuit so sizing metadata (num_rows / assignments / etc)
                    // is recomputed under the wider layout.
                    circuit = match GraphCircuit::from_run_args(&local_run_args, &model_path) {
                        Ok(c) => c,
                        Err(e) => break Err(e),
                    };

                    // Retry.
                    continue;
                }
                Err(e) => break Err(e),
            }
        };

        if res.is_ok() {
            if local_run_args.num_inner_cols != start_inner_cols {
                eprintln!(
                    "[calibrate-settings] case {}/{}: succeeded after increasing num_inner_cols {} -> {}",
                    case_no,
                    total_cases,
                    start_inner_cols,
                    local_run_args.num_inner_cols
                );
            }

            let new_settings = circuit.settings().clone();
            let found_run_args = new_settings.run_args.clone();

            let found_settings = GraphSettings {
                run_args: found_run_args,
                required_lookups: new_settings.required_lookups,
                required_range_checks: new_settings.required_range_checks,
                model_output_scales: new_settings.model_output_scales,
                model_input_scales: new_settings.model_input_scales,
                num_rows: new_settings.num_rows,
                total_assignments: new_settings.total_assignments,
                total_const_size: new_settings.total_const_size,
                dynamic_lookup_params: new_settings.dynamic_lookup_params,
                shuffle_params: new_settings.shuffle_params,
                einsum_params: new_settings.einsum_params,
                ..settings.clone()
            };

            found_params.push(found_settings.clone());

            let found_settings_json = found_settings.as_json()?;
            let found_settings_json = found_settings_json.to_string();

            #[cfg(all(feature = "colored_json", not(target_arch = "wasm32")))]
            {
                let maybe_colored = found_settings_json.to_colored_json_auto().ok();

                log::debug!(
                    "found settings: \n {}",
                    maybe_colored.unwrap_or_else(|| found_settings_json.clone())
                );
            }

            #[cfg(not(all(feature = "colored_json", not(target_arch = "wasm32"))))]
            {
                log::debug!("found settings: \n {}", found_settings_json);
            }

            num_passed += 1;
        } else if let Err(res) = res {
            failure_reasons.push(format!(
                "i-scale: {}, p-scale: {}, rebase-(x): {}, inner_cols: {}, decomp_legs: {}, reason: {}",
                input_scale.to_string(),
                param_scale.to_string(),
                scale_rebase_multiplier.to_string(),
                local_run_args.num_inner_cols.to_string(),
                local_run_args.decomp_legs.to_string(),
                res.to_string()
            ));
            num_failed += 1;
        }

        pb.inc(1);
    }

    pb.finish_with_message("Calibration Done.");

    if found_params.is_empty() {
        // NOTE:
        // Historically we only logged per-candidate failure reasons via `log::error!`, but if the
        // CLI is run without a logger configured those details are lost and users only see the
        // generic message below. Surface the reasons in the returned error instead so callers
        // (including benchmark runners) can diagnose failures.
        let mut msg = String::from(
            "calibration failed, could not find any suitable parameters given the calibration dataset",
        );

        if !failure_reasons.is_empty() {
            const MAX_REASONS: usize = 20;

            msg.push_str("\n\nFailure reasons (first ");
            msg.push_str(&std::cmp::min(MAX_REASONS, failure_reasons.len()).to_string());
            msg.push_str("):\n");

            for (i, reason) in failure_reasons.iter().take(MAX_REASONS).enumerate() {
                msg.push_str(&format!("  {}) {}\n", i + 1, reason));
            }

            if failure_reasons.len() > MAX_REASONS {
                msg.push_str(&format!(
                    "  ... ({} more)\n",
                    failure_reasons.len() - MAX_REASONS
                ));
            }
        }

        return Err(EZKLError::msg(msg));
    }

    log::debug!("Found {} sets of parameters", found_params.len());

    let mut best_params = match target {
        CalibrationTarget::Resources { .. } => {
            let mut param_iterator = found_params.iter().sorted_by_key(|p| p.run_args.logrows);

            let min_logrows = param_iterator
                .next()
                .ok_or_else(|| EZKLError::msg("no params found"))?
                .run_args
                .logrows;

            found_params
                .iter()
                .filter(|p| p.run_args.logrows == min_logrows)
                .max_by_key(|p| {
                    (
                        p.run_args.input_scale,
                        p.run_args.param_scale,
                        p.run_args.scale_rebase_multiplier,
                    )
                })
                .ok_or_else(|| EZKLError::msg("no params found"))?
                .clone()
        }
        CalibrationTarget::Accuracy => {
            let mut param_iterator = found_params.iter().sorted_by_key(|p| {
                (
                    p.run_args.input_scale,
                    p.run_args.param_scale,
                    p.run_args.scale_rebase_multiplier,
                )
            });

            let last = param_iterator.next_back().ok_or_else(|| EZKLError::msg("no params found"))?;
            let max_scale = (
                last.run_args.input_scale,
                last.run_args.param_scale,
                last.run_args.scale_rebase_multiplier,
            );

            found_params
                .iter()
                .filter(|p| {
                    (
                        p.run_args.input_scale,
                        p.run_args.param_scale,
                        p.run_args.scale_rebase_multiplier,
                    ) == max_scale
                })
                .min_by_key(|p| p.run_args.logrows)
                .ok_or_else(|| EZKLError::msg("no params found"))?
                .clone()
        }
    };

    let outputs = forward_pass_res
        .get(&(
            best_params.run_args.input_scale,
            best_params.run_args.param_scale,
            best_params.run_args.scale_rebase_multiplier,
        ))
        .ok_or_else(|| EZKLError::msg("no params found"))?
        .iter()
        .map(|x| x.get_float_outputs(&best_params.model_output_scales))
        .collect::<Vec<_>>();

    let accuracy_res = AccuracyResults::new(
        original_predictions.into_iter().flatten().collect(),
        outputs.into_iter().flatten().collect(),
    )?;

    let tear_sheet = {
        #[cfg(all(feature = "tabled", not(target_arch = "wasm32")))]
        {
            tabled::Table::new(vec![accuracy_res.clone()]).to_string()
        }
        #[cfg(not(all(feature = "tabled", not(target_arch = "wasm32"))))]
        {
            format!("{:#?}", accuracy_res)
        }
    };

    log::warn!(
        "\n\n <------------- Numerical Fidelity Report (input_scale: {}, param_scale: {}, scale_input_multiplier: {}) ------------->\n\n{}\n\n",
        best_params.run_args.input_scale,
        best_params.run_args.param_scale,
        best_params.run_args.scale_rebase_multiplier,
        tear_sheet
    );

    if matches!(target, CalibrationTarget::Resources { col_overflow: true }) {
        let lookup_log_rows = best_params.lookup_log_rows_with_blinding();
        let module_log_row = best_params.module_constraint_logrows_with_blinding();
        let instance_logrows = best_params.log2_total_instances_with_blinding();
        let dynamic_lookup_logrows =
            best_params.min_dynamic_lookup_and_shuffle_logrows_with_blinding();

        let range_check_logrows = best_params.range_check_log_rows_with_blinding();

        let mut reduction = std::cmp::max(lookup_log_rows, module_log_row);
        reduction = std::cmp::max(reduction, range_check_logrows);
        reduction = std::cmp::max(reduction, instance_logrows);
        reduction = std::cmp::max(reduction, dynamic_lookup_logrows);
        reduction = std::cmp::max(reduction, crate::graph::MIN_LOGROWS);

        log::info!(
            "logrows > bits, shrinking logrows: {} -> {}",
            best_params.run_args.logrows,
            reduction
        );

        best_params.run_args.logrows = reduction;
    }

    best_params.save(&settings_path)?;

    log::debug!("Saved parameters.");

    Ok(best_params)
}

pub(crate) fn mock(
    compiled_circuit_path: PathBuf,
    data_path: PathBuf,
) -> Result<String, EZKLError> {
    let mut circuit = GraphCircuit::load(compiled_circuit_path)?;

    let data = GraphWitness::from_path(data_path)?;

    circuit.load_graph_witness(&data)?;

    let public_inputs = circuit.prepare_public_inputs(&data)?;
    let circuit_settings = circuit.settings().clone();
    let public_input_columns = proof_instances(&circuit_settings, public_inputs);

    log::info!("Mock proof");

    let prover = halo2_proofs::dev::MockProver::run(
        circuit.settings().run_args.logrows,
        &circuit,
        public_input_columns,
    )
        .map_err(|e| ExecutionError::MockProverError(e.to_string()))?;

    prover.verify().map_err(ExecutionError::VerifyError)?;
    Ok(String::new())
}

pub(crate) fn compile_circuit(
    model_path: PathBuf,
    compiled_circuit: PathBuf,
    settings_path: PathBuf,
) -> Result<String, EZKLError> {
    let settings = GraphSettings::load(&settings_path)?;
    let circuit = GraphCircuit::from_settings(&settings, &model_path, CheckMode::UNSAFE)?;
    circuit.save(compiled_circuit)?;
    Ok(String::new())
}

pub(crate) fn setup(
    compiled_circuit: PathBuf,
    srs_path: Option<PathBuf>,
    vk_path: PathBuf,
    pk_path: PathBuf,
    witness: Option<PathBuf>,
    disable_selector_compression: bool,
) -> Result<String, EZKLError> {
    let mut circuit = GraphCircuit::load(compiled_circuit)?;

    if let Some(witness) = witness {
        let data = GraphWitness::from_path(witness)?;
        circuit.load_graph_witness(&data)?;
    }

    let logrows = circuit.settings().run_args.logrows;

    let params = load_params_prover::<KZGCommitmentScheme<Bn256>>(srs_path, logrows)?;
    let pk = create_keys::<KZGCommitmentScheme<Bn256>, GraphCircuit>(
        &circuit,
        &params,
        disable_selector_compression,
    )?;

    save_vk::<G1Affine>(&vk_path, pk.get_vk())?;
    save_pk::<G1Affine>(&pk_path, &pk)?;
    Ok(String::new())
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn prove(
    data_path: PathBuf,
    compiled_circuit_path: PathBuf,
    pk_path: PathBuf,
    proof_path: Option<PathBuf>,
    srs_path: Option<PathBuf>,
    check_mode: CheckMode,
) -> Result<Snark<Fr, G1Affine>, EZKLError> {
    let data = GraphWitness::from_path(data_path)?;
    let mut circuit = GraphCircuit::load(compiled_circuit_path)?;

    circuit.load_graph_witness(&data)?;

    let pretty_public_inputs = circuit.pretty_public_inputs(&data)?;
    let public_inputs = circuit.prepare_public_inputs(&data)?;

    let circuit_settings = circuit.settings().clone();
    let public_input_columns = proof_instances(&circuit_settings, public_inputs);

    let proof_split_commits: Option<ProofSplitCommit> = data.into();

    let logrows = circuit_settings.run_args.logrows;

    let pk = load_pk::<KZGCommitmentScheme<Bn256>, GraphCircuit>(pk_path, circuit.params())?;

    let params = load_params_prover::<KZGCommitmentScheme<Bn256>>(srs_path, logrows)?;

    type E = Challenge255<G1Affine>;
    type TW = Blake2bWrite<Vec<u8>, G1Affine, E>;
    type TR = Blake2bRead<Cursor<Vec<u8>>, G1Affine, E>;

    let mut snark = create_proof_circuit::<
        KZGCommitmentScheme<Bn256>,
        _,
        ProverSHPLONK<_>,
        VerifierSHPLONK<_>,
        KZGSingleStrategy<_>,
        E,
        TW,
        TR,
    >(
        circuit,
        public_input_columns,
        &params,
        &pk,
        check_mode,
        proof_split_commits,
        None,
    )?;

    snark.pretty_public_inputs = Some(pretty_public_inputs);

    if let Some(proof_path) = proof_path {
        snark.save(&proof_path)?;
    }

    Ok(snark)
}

fn proof_instances(
    circuit_settings: &GraphSettings,
    public_inputs: Vec<Fr>,
) -> Vec<Vec<Fr>> {
    // Halo2 expects zero instance columns for fully private/fixed circuits.
    if circuit_settings.total_instances().iter().sum::<usize>() == 0 {
        vec![]
    } else {
        vec![public_inputs]
    }
}

pub(crate) fn swap_proof_commitments_cmd(
    proof_path: PathBuf,
    witness: PathBuf,
) -> Result<Snark<Fr, G1Affine>, EZKLError> {
    let snark = Snark::load::<KZGCommitmentScheme<Bn256>>(&proof_path)?;
    let witness = GraphWitness::from_path(witness)?;
    let commitments = witness.get_polycommitments();

    type E = Challenge255<G1Affine>;
    type TW = Blake2bWrite<Vec<u8>, G1Affine, E>;

    let snark_new = swap_proof_commitments::<KZGCommitmentScheme<Bn256>, E, TW>(&snark, &commitments)?;

    if snark_new.proof != *snark.proof {
        log::warn!("swap proof has created a different proof");
    }

    snark_new.save(&proof_path)?;
    Ok(snark_new)
}

pub(crate) fn verify(
    proof_path: PathBuf,
    settings_path: PathBuf,
    vk_path: PathBuf,
    srs_path: Option<PathBuf>,
    reduced_srs: bool,
) -> Result<bool, EZKLError> {
    let circuit_settings = GraphSettings::load(&settings_path)?;

    let logrows = circuit_settings.run_args.logrows;

    let params: ParamsKZG<Bn256> = if reduced_srs {
        load_params_verifier::<KZGCommitmentScheme<Bn256>>(srs_path, 1)?
    } else {
        load_params_verifier::<KZGCommitmentScheme<Bn256>>(srs_path, logrows)?
    };

    type E = Challenge255<G1Affine>;
    type TR = Blake2bRead<Cursor<Vec<u8>>, G1Affine, E>;

    verify_commitment::<
        KZGCommitmentScheme<Bn256>,
        VerifierSHPLONK<'_, Bn256>,
        E,
        KZGSingleStrategy<_>,
        TR,
        GraphCircuit,
        _,
    >(proof_path, circuit_settings, vk_path, &params, logrows)
}

fn verify_commitment<
    'a,
    Scheme: CommitmentScheme,
    V: Verifier<'a, Scheme>,
    E: EncodedChallenge<Scheme::Curve>,
    Strategy: VerificationStrategy<'a, Scheme, V>,
    TR: TranscriptReadBuffer<Cursor<Vec<u8>>, Scheme::Curve, E>,
    C: Circuit<<Scheme as CommitmentScheme>::Scalar, Params = Params>,
    Params,
>(
    proof_path: PathBuf,
    settings: Params,
    vk_path: PathBuf,
    params: &'a Scheme::ParamsVerifier,
    logrows: u32,
) -> Result<bool, EZKLError>
where
    Scheme::Scalar: FromUniformBytes<64>
    + SerdeObject
    + Serialize
    + DeserializeOwned
    + WithSmallOrderMulGroup<3>,
    Scheme::Curve: SerdeObject + Serialize + DeserializeOwned,
    Scheme::ParamsVerifier: 'a,
{
    let proof = Snark::load::<Scheme>(&proof_path)?;

    let strategy = Strategy::new(params);
    let vk = load_vk::<Scheme, C>(vk_path, settings)?;
    let now = Instant::now();

    let result =
        verify_proof_circuit::<V, _, _, _, TR>(&proof, params, &vk, strategy, 1 << logrows);

    let elapsed = now.elapsed();
    log::info!("verify took {}.{}", elapsed.as_secs(), elapsed.subsec_millis());
    log::info!("verified: {}", result.is_ok());
    result
        .map_err(|e: halo2_proofs::plonk::Error| e.into())
        .map(|_| true)
}

/// helper function for load_params
pub(crate) fn load_params_verifier<Scheme: CommitmentScheme>(
    srs_path: Option<PathBuf>,
    logrows: u32,
) -> Result<Scheme::ParamsVerifier, EZKLError> {
    let srs_path = get_srs_path(logrows, srs_path);
    let mut params = load_srs_verifier::<Scheme>(srs_path)?;
    if logrows < params.k() {
        log::info!("downsizing params to {} logrows", logrows);
        params.downsize(logrows);
    }
    Ok(params)
}

/// helper function for load_params
pub(crate) fn load_params_prover<Scheme: CommitmentScheme>(
    srs_path: Option<PathBuf>,
    logrows: u32,
) -> Result<Scheme::ParamsProver, EZKLError> {
    let srs_path = get_srs_path(logrows, srs_path);
    let mut params = load_srs_prover::<Scheme>(srs_path)?;
    if logrows < params.k() {
        log::info!("downsizing params to {} logrows", logrows);
        params.downsize(logrows);
    }
    Ok(params)
}
