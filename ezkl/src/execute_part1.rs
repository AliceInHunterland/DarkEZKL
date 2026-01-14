use crate::circuit::region::RegionSettings;
use crate::circuit::CheckMode;
use crate::commands::CalibrationTarget;
#[cfg(all(feature = "eth", not(target_arch = "wasm32")))]
use crate::eth::deploy_contract_via_solidity;
#[cfg(all(feature = "reusable-verifier", not(target_arch = "wasm32")))]
use crate::eth::register_vka_via_rv;
#[allow(unused_imports)]
#[cfg(all(feature = "eth", not(target_arch = "wasm32")))]
use crate::eth::{get_contract_artifacts, verify_proof_via_solidity};
use crate::graph::input::GraphData;
use crate::graph::{GraphCircuit, GraphSettings, GraphWitness, Model};
use crate::pfsys::{
    create_keys, load_pk, load_vk, save_params, save_pk, swap_proof_commitments, Snark,
};
use crate::pfsys::{create_proof_circuit, verify_proof_circuit, ProofSplitCommit};
use crate::pfsys::{encode_calldata, save_vk, srs::*};
use crate::tensor::TensorError;
use crate::RunArgs;
use crate::EZKL_BUF_CAPACITY;
use crate::{commands::*, EZKLError};
use colored::Colorize;
#[cfg(unix)]
use gag::Gag;
use halo2_proofs::dev::VerifyFailure;
#[cfg(feature = "gpu-accelerated")]
use halo2_proofs::icicle::try_load_and_set_backend_device;
use halo2_proofs::plonk::{self, Circuit};
use halo2_proofs::poly::commitment::Verifier;
use halo2_proofs::poly::commitment::{CommitmentScheme, Params};

use halo2_proofs::poly::kzg::commitment::KZGCommitmentScheme;
use halo2_proofs::poly::kzg::multiopen::{ProverSHPLONK, VerifierSHPLONK};
use halo2_proofs::poly::kzg::{
    commitment::ParamsKZG, strategy::SingleStrategy as KZGSingleStrategy,
};
use halo2_proofs::poly::VerificationStrategy;
use halo2_proofs::transcript::{EncodedChallenge, TranscriptReadBuffer};
#[cfg(all(feature = "eth", not(target_arch = "wasm32")))]
use halo2_solidity_verifier;
use halo2curves::bn256::{Bn256, Fr, G1Affine};
use halo2curves::ff::{FromUniformBytes, WithSmallOrderMulGroup};
use halo2curves::serde::SerdeObject;
#[cfg(feature = "gpu-accelerated")]
use icicle_runtime::{stream::IcicleStream, warmup};
use indicatif::{ProgressBar, ProgressStyle};
use instant::Instant;
use itertools::Itertools;
use lazy_static::lazy_static;
use log::debug;
use log::{info, trace, warn};
use serde::de::DeserializeOwned;
use serde::Serialize;
use snark_verifier::system::halo2::transcript::evm::EvmTranscript;
#[cfg(all(feature = "eth", not(target_arch = "wasm32")))]
use std::fs::File;
use std::io::BufWriter;
use std::io::Cursor;
#[cfg(all(feature = "eth", not(target_arch = "wasm32")))]
use std::io::Write;
use std::path::Path;
use std::path::PathBuf;
use std::time::Duration;
use tabled::Tabled;
use thiserror::Error;
use tract_onnx::prelude::IntoTensor;
use tract_onnx::prelude::Tensor as TractTensor;

lazy_static! {
    #[derive(Debug)]
    /// The path to the ezkl related data.
    pub static ref EZKL_REPO_PATH: String =
        std::env::var("EZKL_REPO_PATH").unwrap_or_else(|_|
            // $HOME/.ezkl/
            format!("{}/.ezkl", std::env::var("HOME").unwrap())
        );

    /// The path to the ezkl related data (SRS)
    pub static ref EZKL_SRS_REPO_PATH: String = format!("{}/srs", *EZKL_REPO_PATH);

}

/// Set the device used for computation.
#[cfg(feature = "gpu-accelerated")]
pub fn set_device() {
    if std::env::var("ICICLE_BACKEND_INSTALL_DIR").is_ok() {
        info!("Running with ICICLE GPU");
        try_load_and_set_backend_device("CUDA");
        match warmup(&IcicleStream::default()) {
            Ok(_) => info!("GPU warmed :)"),
            Err(e) => log::error!("GPU warmup failed: {:?}", e),
        }
    } else {
        info!("Running with CPU: 'ICICLE_BACKEND_INSTALL_DIR' not set");
        try_load_and_set_backend_device("CPU");
    }
}

/// A wrapper for execution errors
#[derive(Debug, Error)]
pub enum ExecutionError {
    /// verification failed
    #[error("verification failed:\n{}", .0.iter().map(|e| e.to_string()).collect::<Vec<_>>().join("\n"))]
    VerifyError(Vec<VerifyFailure>),
    /// Prover error
    #[error("[mock] {0}")]
    MockProverError(String),
}

lazy_static::lazy_static! {
    // read from env EZKL_WORKING_DIR var or default to current dir
    static ref WORKING_DIR: PathBuf = {
        let wd = std::env::var("EZKL_WORKING_DIR").unwrap_or_else(|_| ".".to_string());
        PathBuf::from(wd)
    };
}

/// Run an ezkl command with given args
pub async fn run(command: Commands) -> Result<String, EZKLError> {
    #[cfg(feature = "gpu-accelerated")]
    set_device();
    // set working dir
    std::env::set_current_dir(WORKING_DIR.as_path())?;

    match command {
        #[cfg(feature = "empty-cmd")]
        Commands::Empty => Ok(String::new()),
        Commands::GenSrs { srs_path, logrows } => gen_srs_cmd(srs_path, logrows as u32),
        Commands::GetSrs {
            srs_path,
            settings_path,
            logrows,
        } => get_srs_cmd(srs_path, settings_path, logrows).await,
        Commands::Table { model, args } => table(model.unwrap_or(DEFAULT_MODEL.into()), args),
        Commands::GenSettings {
            model,
            settings_path,
            args,
        } => gen_circuit_settings(
            model.unwrap_or(DEFAULT_MODEL.into()),
            settings_path.unwrap_or(DEFAULT_SETTINGS.into()),
            args,
        ),
        Commands::GenRandomData {
            model,
            data,
            variables,
            seed,
            min,
            max,
        } => gen_random_data(
            model.unwrap_or(DEFAULT_MODEL.into()),
            data.unwrap_or(DEFAULT_DATA.into()),
            variables,
            seed,
            min,
            max,
        ),
        Commands::CalibrateSettings {
            model,
            settings_path,
            data,
            target,
            lookup_safety_margin,
            scales,
            scale_rebase_multiplier,
            max_logrows,
        } => calibrate(
            model.unwrap_or(DEFAULT_MODEL.into()),
            data.unwrap_or(DEFAULT_DATA.into()),
            settings_path.unwrap_or(DEFAULT_SETTINGS.into()),
            target,
            lookup_safety_margin,
            scales,
            scale_rebase_multiplier,
            max_logrows,
        )
        .map(|e| serde_json::to_string(&e).unwrap()),
        Commands::GenWitness {
            data,
            compiled_circuit,
            output,
            vk_path,
            srs_path,
        } => gen_witness(
            compiled_circuit.unwrap_or(DEFAULT_COMPILED_CIRCUIT.into()),
            data.unwrap_or(DataField(DEFAULT_DATA.into())).to_string(),
            Some(output.unwrap_or(DEFAULT_WITNESS.into())),
            vk_path,
            srs_path,
        )
        .map(|e| serde_json::to_string(&e).unwrap()),
        Commands::Mock { model, witness } => mock(
            model.unwrap_or(DEFAULT_MODEL.into()),
            witness.unwrap_or(DEFAULT_WITNESS.into()),
        ),
        #[cfg(all(feature = "eth", not(target_arch = "wasm32")))]
        Commands::CreateEvmVerifier {
            vk_path,
            srs_path,
            settings_path,
            sol_code_path,
            abi_path,
            reusable,
        } => {
            create_evm_verifier(
                vk_path.unwrap_or(DEFAULT_VK.into()),
                srs_path,
                settings_path.unwrap_or(DEFAULT_SETTINGS.into()),
                sol_code_path.unwrap_or(DEFAULT_SOL_CODE.into()),
                abi_path.unwrap_or(DEFAULT_VERIFIER_ABI.into()),
                reusable.unwrap_or(DEFAULT_RENDER_REUSABLE.parse().unwrap()),
            )
            .await
        }
        #[cfg(all(feature = "eth", not(target_arch = "wasm32")))]
        Commands::EncodeEvmCalldata {
            proof_path,
            calldata_path,
            vka_path,
        } => encode_evm_calldata(
            proof_path.unwrap_or(DEFAULT_PROOF.into()),
            calldata_path.unwrap_or(DEFAULT_CALLDATA.into()),
            vka_path,
        )
        .map(|e| serde_json::to_string(&e).unwrap()),
        #[cfg(all(
            feature = "eth",
            feature = "reusable-verifier",
            not(target_arch = "wasm32")
        ))]
        Commands::CreateEvmVka {
            vk_path,
            srs_path,
            settings_path,
            vka_path,
            decimals,
        } => {
            create_evm_vka(
                vk_path.unwrap_or(DEFAULT_VK.into()),
                srs_path,
                settings_path.unwrap_or(DEFAULT_SETTINGS.into()),
                vka_path.unwrap_or(DEFAULT_VKA.into()),
                decimals.unwrap_or(DEFAULT_DECIMALS.parse().unwrap()),
            )
            .await
        }

        Commands::CompileCircuit {
            model,
            compiled_circuit,
            settings_path,
        } => compile_circuit(
            model.unwrap_or(DEFAULT_MODEL.into()),
            compiled_circuit.unwrap_or(DEFAULT_COMPILED_CIRCUIT.into()),
            settings_path.unwrap_or(DEFAULT_SETTINGS.into()),
        ),
        Commands::Setup {
            compiled_circuit,
            srs_path,
            vk_path,
            pk_path,
            witness,
            disable_selector_compression,
        } => setup(
            compiled_circuit.unwrap_or(DEFAULT_COMPILED_CIRCUIT.into()),
            srs_path,
            vk_path.unwrap_or(DEFAULT_VK.into()),
            pk_path.unwrap_or(DEFAULT_PK.into()),
            witness,
            disable_selector_compression
                .unwrap_or(DEFAULT_DISABLE_SELECTOR_COMPRESSION.parse().unwrap()),
        ),
        Commands::SwapProofCommitments {
            proof_path,
            witness_path,
        } => swap_proof_commitments_cmd(
            proof_path.unwrap_or(DEFAULT_PROOF.into()),
            witness_path.unwrap_or(DEFAULT_WITNESS.into()),
        )
        .map(|e| serde_json::to_string(&e).unwrap()),

        Commands::Prove {
            witness,
            compiled_circuit,
            pk_path,
            proof_path,
            srs_path,
            check_mode,
        } => prove(
            witness.unwrap_or(DEFAULT_WITNESS.into()),
            compiled_circuit.unwrap_or(DEFAULT_COMPILED_CIRCUIT.into()),
            pk_path.unwrap_or(DEFAULT_PK.into()),
            Some(proof_path.unwrap_or(DEFAULT_PROOF.into())),
            srs_path,
            check_mode.unwrap_or(DEFAULT_CHECKMODE.parse().unwrap()),
        )
        .map(|e| serde_json::to_string(&e).unwrap()),
        Commands::Verify {
            proof_path,
            settings_path,
            vk_path,
            srs_path,
            reduced_srs,
        } => verify(
            proof_path.unwrap_or(DEFAULT_PROOF.into()),
            settings_path.unwrap_or(DEFAULT_SETTINGS.into()),
            vk_path.unwrap_or(DEFAULT_VK.into()),
            srs_path,
            reduced_srs.unwrap_or(DEFAULT_USE_REDUCED_SRS_FOR_VERIFICATION.parse().unwrap()),
        )
        .map(|e| serde_json::to_string(&e).unwrap()),
        #[cfg(all(feature = "eth", not(target_arch = "wasm32")))]
        Commands::DeployEvm {
            sol_code_path,
            rpc_url,
            addr_path,
            optimizer_runs,
            private_key,
            #[cfg(all(feature = "reusable-verifier", not(target_arch = "wasm32")))]
            contract,
        } => {
            deploy_evm(
                sol_code_path.unwrap_or(DEFAULT_SOL_CODE.into()),
                rpc_url,
                addr_path.unwrap_or(DEFAULT_CONTRACT_ADDRESS.into()),
                optimizer_runs,
                private_key,
                #[cfg(all(feature = "reusable-verifier", not(target_arch = "wasm32")))]
                contract,
                #[cfg(not(all(feature = "reusable-verifier", not(target_arch = "wasm32"))))]
                ContractType::default(),
            )
            .await
        }
        #[cfg(all(feature = "eth", not(target_arch = "wasm32")))]
        Commands::VerifyEvm {
            proof_path,
            addr_verifier,
            rpc_url,
            vka_path,
            encoded_calldata,
        } => {
            verify_evm(
                proof_path.unwrap_or(DEFAULT_PROOF.into()),
                addr_verifier,
                rpc_url,
                vka_path,
                encoded_calldata,
            )
            .await
        }
        #[cfg(feature = "reusable-verifier")]
        Commands::RegisterVka {
            addr_verifier,
            vka_path,
            rpc_url,
            vka_digest_path,
            private_key,
        } => {
            register_vka(
                rpc_url,
                addr_verifier,
                vka_path.unwrap_or(DEFAULT_VKA.into()),
                vka_digest_path.unwrap_or(DEFAULT_VKA_DIGEST.into()),
                private_key,
            )
            .await
        }
        #[cfg(not(feature = "no-update"))]
        Commands::Update { version } => update_ezkl_binary(&version).map(|e| e.to_string()),
    }
}

#[cfg(not(feature = "no-update"))]
/// Assert that the version is valid
fn assert_version_is_valid(version: &str) -> Result<(), EZKLError> {
    let err_string = "Invalid version string. Must be in the format v0.0.0";
    if version.is_empty() {
        return Err(err_string.into());
    }
    // safe to unwrap since we know the length is not 0
    if !version.starts_with('v') {
        return Err(err_string.into());
    }

    semver::Version::parse(&version[1..])
        .map_err(|_| "Invalid version string. Must be in the format v0.0.0")?;

    Ok(())
}

#[cfg(not(feature = "no-update"))]
const INSTALL_BYTES: &[u8] = include_bytes!("../install_ezkl_cli.sh");

#[cfg(not(feature = "no-update"))]
fn update_ezkl_binary(version: &Option<String>) -> Result<String, EZKLError> {
    // run the install script with the version
    let install_script = std::str::from_utf8(INSTALL_BYTES)?;
    //  now run as sh script with the version as an argument

    // check if bash is installed
    let command = if std::process::Command::new("bash")
        .arg("--version")
        .status()
        .is_err()
    {
        log::warn!(
            "bash is not installed on this system, trying to run the install script with sh (may fail)"
        );
        "sh"
    } else {
        "bash"
    };

    let mut command = std::process::Command::new(command);
    let mut command = command.arg("-c").arg(install_script);

    if let Some(version) = version {
        assert_version_is_valid(version)?;
        command = command.arg(version)
    };
    let output = command.output()?;

    if output.status.success() {
        info!("updated binary");
        Ok("".to_string())
    } else {
        Err(format!(
            "failed to update binary: {}, {}",
            std::str::from_utf8(&output.stdout)?,
            std::str::from_utf8(&output.stderr)?
        )
        .into())
    }
}

/// Get the srs path
pub fn get_srs_path(logrows: u32, srs_path: Option<PathBuf>) -> PathBuf {
    if let Some(srs_path) = srs_path {
        srs_path
    } else {
        if !Path::new(&*EZKL_SRS_REPO_PATH).exists() {
            std::fs::create_dir_all(&*EZKL_SRS_REPO_PATH).unwrap();
        }
        Path::new(&*EZKL_SRS_REPO_PATH).join(format!("kzg{}.srs", logrows))
    }
}

fn srs_exists_check(logrows: u32, srs_path: Option<PathBuf>) -> bool {
    Path::new(&get_srs_path(logrows, srs_path)).exists()
}

pub(crate) fn gen_srs_cmd(srs_path: PathBuf, logrows: u32) -> Result<String, EZKLError> {
    let params = gen_srs::<KZGCommitmentScheme<Bn256>>(logrows);
    save_params::<KZGCommitmentScheme<Bn256>>(&srs_path, &params)?;
    Ok(String::new())
}

async fn fetch_srs(uri: &str) -> Result<Vec<u8>, EZKLError> {
    let pb = {
        let pb = init_spinner();
        pb.set_message("Downloading SRS (this may take a while) ...");
        pb
    };
    let client = reqwest::Client::new();
    // wasm doesn't require it to be mutable
    #[allow(unused_mut)]
    let mut resp = client.get(uri).body(vec![]).send().await?;
    let mut buf = vec![];
    while let Some(chunk) = resp.chunk().await? {
        buf.extend(chunk.to_vec());
    }

    pb.finish_with_message("SRS downloaded.");
    Ok(std::mem::take(&mut buf))
}

pub(crate) fn get_file_hash(path: &PathBuf) -> Result<String, EZKLError> {
    use std::io::Read;
    let file = std::fs::File::open(path)?;
    let mut reader = std::io::BufReader::new(file);
    let mut buffer = vec![];
    let bytes_read = reader.read_to_end(&mut buffer)?;
    info!(
        "read {} bytes from file (vector of len = {})",
        bytes_read,
        buffer.len()
    );

    let hash = sha256::digest(buffer);
    info!("file hash: {}", hash);

    Ok(hash)
}

fn check_srs_hash(logrows: u32, srs_path: Option<PathBuf>) -> Result<String, EZKLError> {
    let path = get_srs_path(logrows, srs_path);
    let hash = get_file_hash(&path)?;

    let predefined_hash = match crate::srs_sha::PUBLIC_SRS_SHA256_HASHES.get(&logrows) {
        Some(h) => h,
        None => return Err(format!("SRS (k={}) hash not found in public set", logrows).into()),
    };

    if hash != *predefined_hash {
        // delete file
        warn!("removing SRS file at {}", path.display());
        std::fs::remove_file(path)?;
        return Err(
            "SRS hash does not match the expected hash. Remote SRS may have been tampered with."
                .into(),
        );
    }
    Ok(hash)
}

pub(crate) async fn get_srs_cmd(
    srs_path: Option<PathBuf>,
    settings_path: Option<PathBuf>,
    logrows: Option<u32>,
) -> Result<String, EZKLError> {
    // logrows overrides settings

    let err_string = "You will need to provide a valid settings file to use the settings option. You should run gen-settings to generate a settings file (and calibrate-settings to pick optimal logrows).";

    let k = if let Some(k) = logrows {
        k
    } else if let Some(settings_p) = &settings_path {
        if settings_p.exists() {
            let settings = GraphSettings::load(settings_p)?;
            settings.run_args.logrows
        } else {
            return Err(err_string.into());
        }
    } else {
        return Err(err_string.into());
    };

    if !srs_exists_check(k, srs_path.clone()) {
        info!("SRS does not exist, downloading...");
        let srs_uri = format!("{}{}", PUBLIC_SRS_URL, k);
        let mut reader = Cursor::new(fetch_srs(&srs_uri).await?);
        // check the SRS
        let pb = init_spinner();
        pb.set_message("Validating SRS (this may take a while) ...");
        let params = ParamsKZG::<Bn256>::read(&mut reader)?;
        pb.finish_with_message("SRS validated.");

        info!("Saving SRS to disk...");
        let computed_srs_path = get_srs_path(k, srs_path.clone());
        let mut file = std::fs::File::create(&computed_srs_path)?;
        let mut buffer = BufWriter::with_capacity(*EZKL_BUF_CAPACITY, &mut file);
        params.write(&mut buffer)?;

        info!(
            "Saved SRS to {}.",
            computed_srs_path.as_os_str().to_str().unwrap_or("disk")
        );

        info!("SRS downloaded");
    } else {
        info!("SRS already exists at that path");
    };
    // check the hash
    check_srs_hash(k, srs_path.clone())?;

    Ok(String::new())
}

pub(crate) fn table(model: PathBuf, run_args: RunArgs) -> Result<String, EZKLError> {
    let model = Model::from_run_args(&run_args, &model)?;
    info!("\n {}", model.table_nodes());
    Ok(String::new())
}

pub(crate) fn gen_witness(
    compiled_circuit_path: PathBuf,
    data: String,
    output: Option<PathBuf>,
    vk_path: Option<PathBuf>,
    srs_path: Option<PathBuf>,
) -> Result<GraphWitness, EZKLError> {
    // these aren't real values so the sanity checks are mostly meaningless

    let mut circuit = GraphCircuit::load(compiled_circuit_path)?;
    let data = GraphData::from_str(&data)?;
    let settings = circuit.settings().clone();

    let vk = if let Some(vk) = vk_path {
        Some(load_vk::<KZGCommitmentScheme<Bn256>, GraphCircuit>(
            vk,
            settings.clone(),
        )?)
    } else {
        None
    };

    // NOTE: this used to be conditionally compiled, but the unconditional + cfg'd duplicate
    // definition caused a "redefined variable" compilation error on some targets/features.
    let mut input = circuit.load_graph_input(&data)?;

    // if any of the settings have kzg visibility then we need to load the srs

    let region_settings =
        RegionSettings::all_true(settings.run_args.decomp_base, settings.run_args.decomp_legs);

    let start_time = Instant::now();
    let witness = if settings.module_requires_polycommit() {
        if get_srs_path(settings.run_args.logrows, srs_path.clone()).exists() {
            let srs: ParamsKZG<Bn256> = load_params_prover::<KZGCommitmentScheme<Bn256>>(
                srs_path.clone(),
                settings.run_args.logrows,
            )?;
            circuit.forward::<KZGCommitmentScheme<_>>(
                &mut input,
                vk.as_ref(),
                Some(&srs),
                region_settings,
            )?
        } else {
            warn!("SRS for poly commit does not exist (will be ignored)");
            circuit.forward::<KZGCommitmentScheme<Bn256>>(
                &mut input,
                vk.as_ref(),
                None,
                region_settings,
            )?
        }
    } else {
        circuit.forward::<KZGCommitmentScheme<Bn256>>(
            &mut input,
            vk.as_ref(),
            None,
            region_settings,
        )?
    };

    // print each variable tuple (symbol, value) as symbol=value
    trace!(
        "witness generation {:?} took {:?}",
        circuit
            .settings()
            .run_args
            .variables
            .iter()
            .map(|v| { format!("{}={}", v.0, v.1) })
            .collect::<Vec<_>>(),
        start_time.elapsed()
    );

    if let Some(output_path) = output {
        witness.save(output_path)?;
    }

    // print the witness in debug
    debug!("witness: \n {}", witness.as_json()?.to_colored_json_auto()?);

    Ok(witness)
}

/// Generate a circuit settings file
pub(crate) fn gen_circuit_settings(
    model_path: PathBuf,
    params_output: PathBuf,
    run_args: RunArgs,
) -> Result<String, EZKLError> {
    let circuit = GraphCircuit::from_run_args(&run_args, &model_path)?;
    let params = circuit.settings();
    params.save(&params_output)?;
    Ok(String::new())
}

/// Generate a circuit settings file
pub(crate) fn gen_random_data(
    model_path: PathBuf,
    data_path: PathBuf,
    variables: Vec<(String, usize)>,
    seed: u64,
    min: Option<f32>,
    max: Option<f32>,
) -> Result<String, EZKLError> {
    let mut file = std::fs::File::open(&model_path).map_err(|e| {
        crate::graph::errors::GraphError::ReadWriteFileError(
            model_path.display().to_string(),
            e.to_string(),
        )
    })?;

    let (tract_model, _symbol_values) = Model::load_onnx_using_tract(&mut file, &variables)?;

    let input_facts = tract_model
        .input_outlets()
        .map_err(|e| EZKLError::from(e.to_string()))?
        .iter()
        .map(|&i| tract_model.outlet_fact(i))
        .collect::<tract_onnx::prelude::TractResult<Vec<_>>>()
        .map_err(|e| EZKLError::from(e.to_string()))?;

    let min = min.unwrap_or(0.0);
    let max = max.unwrap_or(1.0);

    /// Generates a random tensor of a given size and type.
    fn random(
        sizes: &[usize],
        datum_type: tract_onnx::prelude::DatumType,
        seed: u64,
        min: f32,
        max: f32,
    ) -> TractTensor {
        use rand::{Rng, SeedableRng};
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

        let mut tensor = TractTensor::zero::<f32>(sizes).unwrap();
        let slice = tensor.as_slice_mut::<f32>().unwrap();
        slice.iter_mut().for_each(|x| *x = rng.gen_range(min..max));
        tensor.cast_to_dt(datum_type).unwrap().into_owned()
    }

    fn tensor_for_fact(
        fact: &tract_onnx::prelude::TypedFact,
        seed: u64,
        min: f32,
        max: f32,
    ) -> TractTensor {
        if let Some(value) = &fact.konst {
            return value.clone().into_tensor();
        }

        random(
            fact.shape
                .as_concrete()
                .expect("Expected concrete shape, found: {fact:?}"),
            fact.datum_type,
            seed,
            min,
            max,
        )
    }

    let generated = input_facts
        .iter()
        .map(|v| tensor_for_fact(v, seed, min, max))
        .collect_vec();

    let data = GraphData::from_tract_data(&generated)?;

    data.save(data_path)?;

    Ok(String::new())
}

// not for wasm targets
pub(crate) fn init_spinner() -> ProgressBar {
    let pb = indicatif::ProgressBar::new_spinner();
    pb.set_draw_target(indicatif::ProgressDrawTarget::stdout());
    pb.enable_steady_tick(Duration::from_millis(200));
    pb.set_style(
        ProgressStyle::with_template("[{elapsed_precise}] {spinner:.blue} {msg}")
            .unwrap()
            .tick_strings(&[
                "------ - ✨ ",
                "------ - ⏳ ",
                "------ - 🌎 ",
                "------ - 🔎 ",
                "------ - 🥹 ",
                "------ - 🫠 ",
                "------ - 👾 ",
            ]),
    );
    pb
}

// not for wasm targets
pub(crate) fn init_bar(len: u64) -> ProgressBar {
    let pb = ProgressBar::new(len);
    pb.set_draw_target(indicatif::ProgressDrawTarget::stdout());
    pb.enable_steady_tick(Duration::from_millis(200));
    let sty = ProgressStyle::with_template(
        "[{elapsed_precise}] {bar:40.cyan/blue} {pos:>7}/{len:7} {msg}",
    )
    .unwrap()
    .progress_chars("##-");
    pb.set_style(sty);
    pb
}
