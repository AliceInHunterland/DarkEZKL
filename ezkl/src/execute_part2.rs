// NOTE: `colored_json` is an optional dependency (enabled via the `colored_json` feature).
// This file must compile even when that feature is disabled (e.g. `--no-default-features`),
// so the import and all usages must be cfg-gated.
#[cfg(all(feature = "colored_json", not(target_arch = "wasm32")))]
use colored_json::prelude::ToColoredJson;

// Needed for `.cartesian_product()`, `.sorted()`, `.dedup()`, `.collect_vec()`, etc.
use itertools::Itertools;

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
            return Err(
                "cannot compute accuracy: empty predictions (no elements after flattening)"
                    .to_string()
                    .into(),
            );
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

    let range = if let Some(scales) = scales {
        scales
    } else {
        (11..14).collect::<Vec<crate::Scale>>()
    };

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

    let mut forward_pass_res = HashMap::new();

    let pb = init_bar(range_grid.len() as u64);
    pb.set_message("calibrating...");

    let mut num_failed = 0;
    let mut num_passed = 0;
    let mut failure_reasons = vec![];

    for ((input_scale, param_scale), scale_rebase_multiplier) in range_grid {
        // NOTE: avoid `colored` dependency methods (e.g. `.blue()`, `.red()`) here so this
        // code can compile in builds that don't enable optional coloring/logging features.
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

        let local_run_args = RunArgs {
            input_scale,
            param_scale,
            scale_rebase_multiplier,
            lookup_range: (IntegerRep::MIN, IntegerRep::MAX),
            ..settings.run_args.clone()
        };

        // if unix get a gag
        #[cfg(all(not(not(feature = "ezkl")), unix))]
        let _r = gag::Gag::stdout().ok();
        #[cfg(all(not(not(feature = "ezkl")), unix))]
        let _g = gag::Gag::stderr().ok();

        let mut circuit = match GraphCircuit::from_run_args(&local_run_args, &model_path) {
            Ok(c) => c,
            Err(e) => {
                log::error!("circuit creation from run args failed: {:?}", e);
                failure_reasons.push(format!(
                    "i-scale: {}, p-scale: {}, rebase-(x): {}, reason: {}",
                    input_scale.to_string(),
                    param_scale.to_string(),
                    scale_rebase_multiplier.to_string(),
                    e
                ));
                pb.inc(1);
                num_failed += 1;
                continue;
            }
        };

        let forward_res = chunks
            .iter()
            .map(|chunk| -> Result<(), String> {
                let chunk = chunk.clone();

                let data = circuit
                    .load_graph_input(&chunk)
                    .map_err(|e| format!("failed to load circuit inputs: {}", e))?;

                let forward_res = circuit
                    .forward::<KZGCommitmentScheme<Bn256>>(
                        &mut data.clone(),
                        None,
                        None,
                        RegionSettings::all_true(
                            settings.run_args.decomp_base,
                            settings.run_args.decomp_legs,
                        ),
                    )
                    .map_err(|e| format!("failed to forward: {}", e))?;

                // push result to the hashmap
                forward_pass_res
                    .get_mut(&key)
                    .ok_or("key not found")?
                    .push(forward_res);

                Ok(())
            })
            .collect::<Result<Vec<()>, String>>();

        match forward_res {
            Ok(_) => (),
            // typically errors will be due to the circuit overflowing the i64 limit
            Err(e) => {
                log::error!("forward pass failed: {:?}", e);
                pb.inc(1);
                num_failed += 1;
                failure_reasons.push(format!(
                    "i-scale: {}, p-scale: {}, rebase-(x): {}, reason: {}",
                    input_scale.to_string(),
                    param_scale.to_string(),
                    scale_rebase_multiplier.to_string(),
                    e
                ));
                continue;
            }
        }

        // drop the gag
        #[cfg(all(not(not(feature = "ezkl")), unix))]
        drop(_r);
        #[cfg(all(not(not(feature = "ezkl")), unix))]
        drop(_g);

        let result = forward_pass_res.get(&key).ok_or("key not found")?;

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

        let res = circuit.calc_min_logrows(
            (min_lookup_range, max_lookup_range),
            max_range_size,
            max_logrows,
            lookup_safety_margin,
        );

        if res.is_ok() {
            let new_settings = circuit.settings().clone();

            let found_run_args = RunArgs {
                input_scale: new_settings.run_args.input_scale,
                param_scale: new_settings.run_args.param_scale,
                lookup_range: new_settings.run_args.lookup_range,
                logrows: new_settings.run_args.logrows,
                scale_rebase_multiplier: new_settings.run_args.scale_rebase_multiplier,
                ..settings.run_args.clone()
            };

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

            // Pretty-print if optional `colored_json` dependency is enabled; otherwise fall back
            // to plain JSON. Importantly: do NOT propagate a `colored_json` formatting error via `?`
            // (it may not implement conversion into `EZKLError`).
            let found_settings_json = found_settings.as_json()?;
            let found_settings_json = found_settings_json.to_string();

            #[cfg(all(feature = "colored_json", not(target_arch = "wasm32")))]
            {
                // `colored_json::prelude::ToColoredJson` is implemented for `AsRef<str>` (e.g. `String` / `&str`),
                // so colorize the JSON string directly (no need to parse into `serde_json::Value`).
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
                "i-scale: {}, p-scale: {}, rebase-(x): {}, reason: {}",
                input_scale.to_string(),
                param_scale.to_string(),
                scale_rebase_multiplier.to_string(),
                res.to_string()
            ));
            num_failed += 1;
        }

        pb.inc(1);
    }

    pb.finish_with_message("Calibration Done.");

    if found_params.is_empty() {
        if !failure_reasons.is_empty() {
            log::error!("Calibration failed for the following reasons:");
            for reason in failure_reasons {
                log::error!("{}", reason);
            }
        }

        return Err("calibration failed, could not find any suitable parameters given the calibration dataset".into());
    }

    log::debug!("Found {} sets of parameters", found_params.len());

    // now find the best params according to the target
    let mut best_params = match target {
        CalibrationTarget::Resources { .. } => {
            let mut param_iterator = found_params.iter().sorted_by_key(|p| p.run_args.logrows);

            let min_logrows = param_iterator
                .next()
                .ok_or("no params found")?
                .run_args
                .logrows;

            // pick the ones that have the minimum logrows but also the largest scale:
            // this is the best tradeoff between resource usage and accuracy
            found_params
                .iter()
                .filter(|p| p.run_args.logrows == min_logrows)
                .max_by_key(|p| {
                    (
                        p.run_args.input_scale,
                        p.run_args.param_scale,
                        // we want the largest rebase multiplier as it means we can use less constraints
                        p.run_args.scale_rebase_multiplier,
                    )
                })
                .ok_or("no params found")?
                .clone()
        }
        CalibrationTarget::Accuracy => {
            let mut param_iterator = found_params.iter().sorted_by_key(|p| {
                (
                    p.run_args.input_scale,
                    p.run_args.param_scale,
                    // we want the largest rebase multiplier as it means we can use less constraints
                    p.run_args.scale_rebase_multiplier,
                )
            });

            let last = param_iterator.next_back().ok_or("no params found")?;
            let max_scale = (
                last.run_args.input_scale,
                last.run_args.param_scale,
                last.run_args.scale_rebase_multiplier,
            );

            // pick the ones that have the max scale but also the smallest logrows:
            // this is the best tradeoff between resource usage and accuracy
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
                .ok_or("no params found")?
                .clone()
        }
    };

    let outputs = forward_pass_res
        .get(&(
            best_params.run_args.input_scale,
            best_params.run_args.param_scale,
            best_params.run_args.scale_rebase_multiplier,
        ))
        .ok_or("no params found")?
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
    // mock should catch any issues by default so we set it to safe
    let mut circuit = GraphCircuit::load(compiled_circuit_path)?;

    let data = GraphWitness::from_path(data_path)?;

    circuit.load_graph_witness(&data)?;

    let public_inputs = circuit.prepare_public_inputs(&data)?;

    log::info!("Mock proof");

    let prover = halo2_proofs::dev::MockProver::run(
        circuit.settings().run_args.logrows,
        &circuit,
        vec![public_inputs],
    )
    .map_err(|e| ExecutionError::MockProverError(e.to_string()))?;

    prover.verify().map_err(ExecutionError::VerifyError)?;
    Ok(String::new())
}

#[cfg(all(feature = "eth", not(target_arch = "wasm32")))]
pub(crate) async fn create_evm_verifier(
    vk_path: PathBuf,
    srs_path: Option<PathBuf>,
    settings_path: PathBuf,
    sol_code_path: PathBuf,
    abi_path: PathBuf,
    reusable: bool,
) -> Result<String, EZKLError> {
    let settings = GraphSettings::load(&settings_path)?;
    let params =
        load_params_verifier::<KZGCommitmentScheme<Bn256>>(srs_path, settings.run_args.logrows)?;

    let num_instance = settings.total_instances();
    // create a scales array that is the same length as the number of instances, all populated with 0
    let scales = vec![0; num_instance.len()];
    // let poseidon_instance = settings.module_sizes.num_instances().iter().sum::<usize>();

    let vk = load_vk::<KZGCommitmentScheme<Bn256>, GraphCircuit>(vk_path, settings)?;
    log::trace!("params computed");

    let generator = halo2_solidity_verifier::SolidityGenerator::new(
        &params,
        &vk,
        halo2_solidity_verifier::BatchOpenScheme::Bdfg21,
        &num_instance,
        &scales,
        0,
        0,
    );
    let (verifier_solidity, name) = if reusable {
        (generator.render_separately()?.0, "Halo2VerifierReusable") // ignore the rendered vk artifact for now and generate it in create_evm_vka
    } else {
        (generator.render()?, "Halo2Verifier")
    };

    File::create(sol_code_path.clone())?.write_all(verifier_solidity.as_bytes())?;

    // fetch abi of the contract
    let (abi, _, _) = get_contract_artifacts(sol_code_path, name, 0).await?;
    // save abi to file
    serde_json::to_writer(std::fs::File::create(abi_path)?, &abi)?;

    Ok(String::new())
}

#[cfg(feature = "reusable-verifier")]
pub(crate) async fn create_evm_vka(
    vk_path: PathBuf,
    srs_path: Option<PathBuf>,
    settings_path: PathBuf,
    vka_path: PathBuf,
    decimals: usize,
) -> Result<String, EZKLError> {
    log::warn!("Reusable verifier support is experimental and may change in the future. Use at your own risk.");
    let settings = GraphSettings::load(&settings_path)?;
    let params =
        load_params_verifier::<KZGCommitmentScheme<Bn256>>(srs_path, settings.run_args.logrows)?;

    let num_poseidon_instance = settings.module_sizes.num_instances().iter().sum::<usize>();
    let num_fixed_point_instance = settings
        .model_instance_shapes
        .iter()
        .map(|x| x.iter().product::<usize>())
        .collect_vec();

    let scales = settings.get_model_instance_scales();
    let vk = load_vk::<KZGCommitmentScheme<Bn256>, GraphCircuit>(vk_path, settings)?;
    log::trace!("params computed");
    // assert that the decimals must be less than or equal to 38 to prevent overflow
    if decimals > 38 {
        return Err("decimals must be less than or equal to 38".into());
    }

    let generator = halo2_solidity_verifier::SolidityGenerator::new(
        &params,
        &vk,
        halo2_solidity_verifier::BatchOpenScheme::Bdfg21,
        &num_fixed_point_instance,
        &scales,
        decimals,
        num_poseidon_instance,
    );

    let vka_words: Vec<[u8; 32]> = generator.render_separately_vka_words()?.1;
    let serialized_vka_words = bincode::serialize(&vka_words).or_else(|e| {
        Err(EZKLError::from(format!(
            "Failed to serialize vka words: {}",
            e
        )))
    })?;

    File::create(vka_path.clone())?.write_all(&serialized_vka_words)?;

    // Load in the vka words and deserialize them and check that they match the original
    let bytes = std::fs::read(vka_path)?;
    let vka_buf: Vec<[u8; 32]> = bincode::deserialize(&bytes)
        .map_err(|e| EZKLError::from(format!("Failed to deserialize vka words: {e}")))?;
    if vka_buf != vka_words {
        return Err("vka words do not match".into());
    };

    Ok(String::new())
}
#[cfg(all(feature = "eth", not(target_arch = "wasm32")))]
pub(crate) async fn deploy_evm(
    sol_code_path: PathBuf,
    rpc_url: String,
    addr_path: PathBuf,
    runs: usize,
    private_key: Option<String>,
    contract: ContractType,
) -> Result<String, EZKLError> {
    let contract_name = match contract {
        ContractType::Verifier { reusable: false } => "Halo2Verifier",
        ContractType::Verifier { reusable: true } => "Halo2VerifierReusable",
    };
    let contract_address = deploy_contract_via_solidity(
        sol_code_path,
        &rpc_url,
        runs,
        private_key.as_deref(),
        contract_name,
    )
    .await?;

    log::info!("Contract deployed at: {:#?}", contract_address);

    let mut f = File::create(addr_path)?;
    write!(f, "{:#?}", contract_address)?;
    Ok(String::new())
}

#[cfg(all(feature = "reusable-verifier", not(target_arch = "wasm32")))]
pub(crate) async fn register_vka(
    rpc_url: String,
    rv_addr: H160Flag,
    vka_path: PathBuf,
    vka_digest_path: PathBuf,
    private_key: Option<String>,
) -> Result<String, EZKLError> {
    log::warn!("Reusable verifier support is experimental and may change in the future. Use at your own risk.");
    // Load the vka, which is bincode serialized, from the vka_path
    let bytes = std::fs::read(vka_path)?;
    let vka_buf: Vec<[u8; 32]> = bincode::deserialize(&bytes)
        .map_err(|e| EZKLError::from(format!("Failed to deserialize vka words: {e}")))?;
    let vka_digest = register_vka_via_rv(
        rpc_url.as_ref(),
        private_key.as_deref(),
        rv_addr.into(),
        &vka_buf,
    )
    .await?;

    log::info!("VKA digest: {:#?}", vka_digest);

    let mut f = File::create(vka_digest_path)?;
    write!(f, "{:#?}", vka_digest)?;
    Ok(String::new())
}

/// Encodes the calldata for the EVM verifier
/// TODO: Add a "RV address param" which will query the "RegisteredVKA" events to fetch the
/// VKA from the vka_digest.
#[cfg(all(feature = "eth", not(target_arch = "wasm32")))]
pub(crate) fn encode_evm_calldata(
    proof_path: PathBuf,
    calldata_path: PathBuf,
    vka_path: Option<PathBuf>,
) -> Result<Vec<u8>, EZKLError> {
    let snark = Snark::load::<KZGCommitmentScheme<Bn256>>(&proof_path)?;

    let flattened_instances = snark.instances.into_iter().flatten();

    // Load the vka, which is bincode serialized, from the vka_path
    let vka_buf: Option<Vec<[u8; 32]>> =
        match vka_path {
            Some(path) => {
                let bytes = std::fs::read(path)?;
                Some(bincode::deserialize(&bytes).map_err(|e| {
                    EZKLError::from(format!("Failed to deserialize vka words: {e}"))
                })?)
            }
            None => None,
        };

    let vka: Option<&[[u8; 32]]> = vka_buf.as_deref();
    let encoded = encode_calldata(vka, &snark.proof, &flattened_instances.collect::<Vec<_>>());

    log::debug!("Encoded calldata: {:?}", encoded);

    File::create(calldata_path)?.write_all(encoded.as_slice())?;

    Ok(encoded)
}

/// TODO: Add an optional vka_digest param that will allow us to fetch the associated VKA
/// from the RegisteredVKA events on the RV.
#[cfg(all(feature = "eth", not(target_arch = "wasm32")))]
pub(crate) async fn verify_evm(
    proof_path: PathBuf,
    addr_verifier: H160Flag,
    rpc_url: String,
    vka_path: Option<PathBuf>,
    encoded_calldata: Option<PathBuf>,
) -> Result<String, EZKLError> {
    let proof = Snark::load::<KZGCommitmentScheme<Bn256>>(&proof_path)?;

    let result = verify_proof_via_solidity(
        proof.clone(),
        addr_verifier.into(),
        vka_path.map(|s| s.into()),
        rpc_url.as_ref(),
        encoded_calldata.map(|s| s.into()),
    )
    .await?;

    log::info!("Solidity verification result: {}", result);

    if !result {
        return Err("Solidity verification failed".into());
    }

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
    // these aren't real values so the sanity checks are mostly meaningless

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

    let proof_split_commits: Option<ProofSplitCommit> = data.into();

    let logrows = circuit_settings.run_args.logrows;
    // creates and verifies the proof

    let pk = load_pk::<KZGCommitmentScheme<Bn256>, GraphCircuit>(pk_path, circuit.params())?;

    let params = load_params_prover::<KZGCommitmentScheme<Bn256>>(srs_path, logrows)?;
    let mut snark = create_proof_circuit::<
        KZGCommitmentScheme<Bn256>,
        _,
        ProverSHPLONK<_>,
        VerifierSHPLONK<_>,
        KZGSingleStrategy<_>,
        _,
        EvmTranscript<_, _, _, _>,
        EvmTranscript<_, _, _, _>,
    >(
        circuit,
        vec![public_inputs],
        &params,
        &pk,
        check_mode,
        proof_split_commits,
        None,
    )?;

    snark.pretty_public_inputs = pretty_public_inputs;

    if let Some(proof_path) = proof_path {
        snark.save(&proof_path)?;
    }

    Ok(snark)
}

pub(crate) fn swap_proof_commitments_cmd(
    proof_path: PathBuf,
    witness: PathBuf,
) -> Result<Snark<Fr, G1Affine>, EZKLError> {
    let snark = Snark::load::<KZGCommitmentScheme<Bn256>>(&proof_path)?;
    let witness = GraphWitness::from_path(witness)?;
    let commitments = witness.get_polycommitments();

    let snark_new = swap_proof_commitments::<
        KZGCommitmentScheme<Bn256>,
        _,
        EvmTranscript<G1Affine, _, _, _>,
    >(&snark, &commitments)?;

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
        // only need G_0 for the verification with shplonk
        load_params_verifier::<KZGCommitmentScheme<Bn256>>(srs_path, 1)?
    } else {
        load_params_verifier::<KZGCommitmentScheme<Bn256>>(srs_path, logrows)?
    };

    verify_commitment::<
        KZGCommitmentScheme<Bn256>,
        VerifierSHPLONK<'_, Bn256>,
        _,
        KZGSingleStrategy<_>,
        EvmTranscript<G1Affine, _, _, _>,
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
