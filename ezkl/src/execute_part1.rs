use crate::circuit::region::RegionSettings;
use bytes::Bytes;
use crate::circuit::CheckMode;
use crate::CalibrationTarget;
#[allow(unused_imports)]
#[cfg(all(feature = "eth", not(target_arch = "wasm32")))]
use crate::eth::{get_contract_artifacts, verify_proof_via_solidity};
use crate::graph::input::GraphData;
use crate::graph::{GraphCircuit, GraphSettings, GraphWitness, Model};
use crate::pfsys::{
    create_keys, load_pk, load_vk, save_params, save_pk, swap_proof_commitments, Snark,
};
use crate::pfsys::{create_proof_circuit, verify_proof_circuit, ProofSplitCommit};
use crate::pfsys::{save_vk, srs::*};
use crate::RunArgs;
use crate::EZKL_BUF_CAPACITY;
use crate::{commands::*, EZKLError};
use halo2_proofs::dev::VerifyFailure;
#[cfg(feature = "gpu-accelerated")]
use halo2_proofs::icicle::try_load_and_set_backend_device;
use halo2_proofs::plonk::Circuit;
use halo2_proofs::poly::commitment::Verifier;
use halo2_proofs::poly::commitment::{CommitmentScheme, Params};

use halo2_proofs::poly::kzg::commitment::KZGCommitmentScheme;
use halo2_proofs::poly::kzg::multiopen::{ProverSHPLONK, VerifierSHPLONK};
use halo2_proofs::poly::kzg::{
    commitment::ParamsKZG, strategy::SingleStrategy as KZGSingleStrategy,
};
use halo2_proofs::poly::VerificationStrategy;
use halo2_proofs::transcript::{
    Blake2bRead, Blake2bWrite, Challenge255, EncodedChallenge, TranscriptReadBuffer,
};
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
use std::io::BufWriter;
use std::io::Cursor;
use std::io::IsTerminal;
use std::io::Write;
use std::path::Path;
use std::path::PathBuf;
use std::time::Duration;
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
        Commands::Table { model, args } => table(model.unwrap_or(crate::DEFAULT_MODEL.into()), args),
        Commands::GenSettings {
            model,
            settings_path,
            args,
        } => gen_circuit_settings(
            model.unwrap_or(crate::DEFAULT_MODEL.into()),
            settings_path.unwrap_or(crate::DEFAULT_SETTINGS.into()),
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
            model.unwrap_or(crate::DEFAULT_MODEL.into()),
            data.unwrap_or(crate::DEFAULT_DATA.into()),
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
            model.unwrap_or(crate::DEFAULT_MODEL.into()),
            data.unwrap_or(crate::DEFAULT_DATA.into()),
            settings_path.unwrap_or(crate::DEFAULT_SETTINGS.into()),
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
            compiled_circuit.unwrap_or(crate::DEFAULT_COMPILED_CIRCUIT.into()),
            data.unwrap_or(DataField(crate::DEFAULT_DATA.into())).to_string(),
            Some(output.unwrap_or(crate::DEFAULT_WITNESS.into())),
            vk_path,
            srs_path,
        )
        .map(|e| serde_json::to_string(&e).unwrap()),
        Commands::Mock { model, witness } => mock(
            model.unwrap_or(crate::DEFAULT_MODEL.into()),
            witness.unwrap_or(crate::DEFAULT_WITNESS.into()),
        ),

        Commands::CompileCircuit {
            model,
            compiled_circuit,
            settings_path,
        } => compile_circuit(
            model.unwrap_or(crate::DEFAULT_MODEL.into()),
            compiled_circuit.unwrap_or(crate::DEFAULT_COMPILED_CIRCUIT.into()),
            settings_path.unwrap_or(crate::DEFAULT_SETTINGS.into()),
        ),
        Commands::Setup {
            compiled_circuit,
            srs_path,
            vk_path,
            pk_path,
            witness,
            disable_selector_compression,
        } => setup(
            compiled_circuit.unwrap_or(crate::DEFAULT_COMPILED_CIRCUIT.into()),
            srs_path,
            vk_path.unwrap_or(crate::DEFAULT_VK.into()),
            pk_path.unwrap_or(crate::DEFAULT_PK.into()),
            witness,
            disable_selector_compression
                .unwrap_or(crate::DEFAULT_DISABLE_SELECTOR_COMPRESSION.parse().unwrap()),
        ),
        Commands::SwapProofCommitments {
            proof_path,
            witness_path,
        } => swap_proof_commitments_cmd(
            proof_path.unwrap_or(crate::DEFAULT_PROOF.into()),
            witness_path.unwrap_or(crate::DEFAULT_WITNESS.into()),
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
            witness.unwrap_or(crate::DEFAULT_WITNESS.into()),
            compiled_circuit.unwrap_or(crate::DEFAULT_COMPILED_CIRCUIT.into()),
            pk_path.unwrap_or(crate::DEFAULT_PK.into()),
            Some(proof_path.unwrap_or(crate::DEFAULT_PROOF.into())),
            srs_path,
            check_mode.unwrap_or(crate::DEFAULT_CHECKMODE.parse().unwrap()),
        )
        .map(|e| serde_json::to_string(&e).unwrap()),
        Commands::Verify {
            proof_path,
            settings_path,
            vk_path,
            srs_path,
            reduced_srs,
        } => verify(
            proof_path.unwrap_or(crate::DEFAULT_PROOF.into()),
            settings_path.unwrap_or(crate::DEFAULT_SETTINGS.into()),
            vk_path.unwrap_or(crate::DEFAULT_VK.into()),
            srs_path,
            reduced_srs.unwrap_or(crate::DEFAULT_USE_REDUCED_SRS_FOR_VERIFICATION.parse().unwrap()),
        )
        .map(|e| serde_json::to_string(&e).unwrap()),
        #[cfg(not(feature = "no-update"))]
        Commands::Update { version } => update_ezkl_binary(&version).map(|e| e.to_string()),
    }
}

#[cfg(not(feature = "no-update"))]
/// Assert that the version is valid
fn assert_version_is_valid(version: &str) -> Result<(), EZKLError> {
    let err_string = "Invalid version string. Must be in the format v0.0.0";
    if version.is_empty() {
        return Err(EZKLError::msg(err_string));
    }
    // safe to unwrap since we know the length is not 0
    if !version.starts_with('v') {
        return Err(EZKLError::msg(err_string));
    }

    semver::Version::parse(&version[1..])
        .map_err(|_| EZKLError::msg("Invalid version string. Must be in the format v0.0.0"))?;

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
        Err(EZKLError::msg(format!(
            "failed to update binary: {}, {}",
            std::str::from_utf8(&output.stdout)?,
            std::str::from_utf8(&output.stderr)?
        )))
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SrsSource {
    /// Only accept a public trusted-setup SRS (download + hash check).
    Public,
    /// Try `Public`, but fall back to `Dummy` on download/validation failures.
    Auto,
    /// Always generate a local (DUMMY) SRS (benchmarking only; insecure for production).
    Dummy,
}

impl SrsSource {
    fn from_env() -> Self {
        let raw = std::env::var("EZKL_SRS_SOURCE").unwrap_or_default();
        match raw.trim().to_lowercase().as_str() {
            "" | "public" | "download" => SrsSource::Public,
            "auto" => SrsSource::Auto,
            "dummy" | "local" | "gen" | "generate" => SrsSource::Dummy,
            other => {
                warn!(
                    "Unknown EZKL_SRS_SOURCE='{}' (expected public|auto|dummy); defaulting to 'public'",
                    other
                );
                SrsSource::Public
            }
        }
    }
}

fn ensure_parent_dir(path: &Path) -> Result<(), EZKLError> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() && !parent.exists() {
            std::fs::create_dir_all(parent)?;
        }
    }
    Ok(())
}

fn srs_dummy_marker_path(srs_path: &Path) -> PathBuf {
    PathBuf::from(format!("{}.dummy", srs_path.display()))
}

fn is_marked_dummy(srs_path: &Path) -> bool {
    srs_dummy_marker_path(srs_path).exists()
}

fn mark_srs_as_dummy(srs_path: &Path) -> Result<(), EZKLError> {
    let marker = srs_dummy_marker_path(srs_path);
    ensure_parent_dir(&marker)?;
    std::fs::write(
        marker,
        b"dummy srs: generated locally by ezkl (NOT from a public trusted setup)\n",
    )?;
    Ok(())
}

fn clear_dummy_marker(srs_path: &Path) -> Result<(), EZKLError> {
    let marker = srs_dummy_marker_path(srs_path);
    if marker.exists() {
        std::fs::remove_file(marker)?;
    }
    Ok(())
}

fn copy_srs_and_marker(from: &Path, to: &Path) -> Result<(), EZKLError> {
    if from == to {
        return Ok(());
    }

    ensure_parent_dir(to)?;

    // Prefer hard-link to avoid multi-GB copies (common in Docker bind-mount setups).
    // If it fails (e.g., cross-device), fall back to a real copy.
    if !to.exists() {
        if let Err(link_err) = std::fs::hard_link(from, to) {
            // Fall back to copy (EXDEV is common when crossing filesystems/mounts).
            std::fs::copy(from, to).map_err(|copy_err| {
                EZKLError::msg(format!(
                    "failed to hard-link SRS {} -> {}: {}; and failed to copy: {}",
                    from.display(),
                    to.display(),
                    link_err,
                    copy_err
                ))
            })?;
        }
    }

    // Carry dummy marker along (if present).
    if is_marked_dummy(from) {
        mark_srs_as_dummy(to)?;
    } else {
        clear_dummy_marker(to)?;
    }

    Ok(())
}

pub(crate) fn gen_srs_cmd(srs_path: PathBuf, logrows: u32) -> Result<String, EZKLError> {
    ensure_parent_dir(&srs_path)?;
    let params = gen_srs::<KZGCommitmentScheme<Bn256>>(logrows);
    save_params::<KZGCommitmentScheme<Bn256>>(&srs_path, &params)?;
    Ok(String::new())
}

fn env_u64(key: &str, default: u64) -> u64 {
    std::env::var(key)
        .ok()
        .and_then(|v| v.trim().parse::<u64>().ok())
        .unwrap_or(default)
}

fn mib(x: u64) -> f64 {
    (x as f64) / (1024.0 * 1024.0)
}

async fn download_srs_to_file(uri: &str, out_path: &Path) -> Result<(), EZKLError> {
    let pb = {
        let pb = init_spinner();
        pb.set_message(format!(
            "Downloading SRS (this may take a while) ... ({uri} -> {})",
            out_path.display()
        ));
        pb
    };

    let connect_timeout_s = env_u64("EZKL_SRS_CONNECT_TIMEOUT_SECS", 15);
    let stall_timeout_s = env_u64("EZKL_SRS_STALL_TIMEOUT_SECS", 60);
    let progress_every_s = env_u64("EZKL_SRS_PROGRESS_EVERY_SECS", 30);

    let client = reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(connect_timeout_s))
        .build()?;

    // NOTE: Some hosts/WAFs block non-browser user agents. Setting a conservative UA helps.
    #[allow(unused_mut)]
    let mut resp = client
        .get(uri)
        .header(
            reqwest::header::USER_AGENT,
            "Mozilla/5.0 (X11; Linux x86_64) dark-ezkl/bench (reqwest)",
        )
        .body(vec![])
        .send()
        .await?;

    let status = resp.status();
    if !status.is_success() {
        let body = resp.bytes().await.unwrap_or_default();
        let snippet_len = std::cmp::min(2048, body.len());
        let snippet = String::from_utf8_lossy(&body[..snippet_len]);
        pb.finish_and_clear();

        return Err(EZKLError::msg(format!(
            "failed to download SRS from {uri} (HTTP {status}).\n\
             Hint: set EZKL_SRS_SOURCE=auto (fallback to dummy) or EZKL_SRS_SOURCE=dummy (no download).\n\
             Response body (first {snippet_len} bytes):\n{snippet}"
        )));
    }

    ensure_parent_dir(out_path)?;
    
    // Immediately report that download is starting
    eprintln!("[get-srs] streaming download from {uri} to {}", out_path.display());

    // Stream body to disk (avoid buffering multi-GB SRS in RAM).
    let file = std::fs::File::create(out_path)?;
    let mut writer = BufWriter::with_capacity(EZKL_BUF_CAPACITY, file);

    let total = resp.content_length();
    let start = Instant::now();
    let mut downloaded: u64 = 0;
    let mut last_report = Instant::now();
    
    // Report initial state
    if let Some(t) = total {
        eprintln!("[get-srs] starting download: {:.1} MiB total", mib(t));
    } else {
        eprintln!("[get-srs] starting download: size unknown");
    }

    loop {
        let next: Result<Option<Bytes>, reqwest::Error> = tokio::time::timeout(
            Duration::from_secs(stall_timeout_s),
            resp.chunk(),
        )
        .await
        .map_err(|_| {
            pb.finish_and_clear();
            EZKLError::msg(format!(
                "SRS download stalled for >= {stall_timeout_s}s from {uri}.\n\
                 Hint: set EZKL_SRS_SOURCE=dummy to skip download."
            ))
        })?;

        let Some(chunk) = next? else {
            break;
        };

        writer.write_all(&chunk)?;
        downloaded += chunk.len() as u64;

        if last_report.elapsed() >= Duration::from_secs(progress_every_s) {
            if let Some(t) = total {
                eprintln!(
                    "[get-srs] downloaded {:.1}/{:.1} MiB ({:.1}%) in {:.0}s",
                    mib(downloaded),
                    mib(t),
                    (downloaded as f64 * 100.0) / (t as f64),
                    start.elapsed().as_secs_f64()
                );
            } else {
                eprintln!(
                    "[get-srs] downloaded {:.1} MiB in {:.0}s",
                    mib(downloaded),
                    start.elapsed().as_secs_f64()
                );
            }
            last_report = Instant::now();
        }
    }

    writer.flush()?;
    pb.finish_with_message("SRS downloaded.");
    Ok(())
}

pub(crate) fn get_file_hash(path: &PathBuf) -> Result<String, EZKLError> {
    // Streaming hash (do not read multi-GB files into RAM).
    let hash = sha256::try_digest(path.as_path()).map_err(|e| {
        EZKLError::msg(format!(
            "failed to compute sha256 of {}: {e}",
            path.display()
        ))
    })?;
    info!("file hash: {}", hash);

    Ok(hash)
}

fn enforce_public_srs_hash(logrows: u32, path: &PathBuf) -> Result<String, EZKLError> {
    if is_marked_dummy(path.as_path()) {
        return Err(EZKLError::msg(format!(
            "SRS at {} is marked as DUMMY ({} exists).\n\
             Refusing to treat it as a public trusted-setup SRS.\n\
             Delete it or set EZKL_SRS_SOURCE=auto|dummy.",
            path.display(),
            srs_dummy_marker_path(path.as_path()).display(),
        )));
    }

    let hash = get_file_hash(path)?;

    let predefined_hash = match crate::srs_sha::PUBLIC_SRS_SHA256_HASHES.get(&logrows) {
        Some(h) => h,
        None => {
            return Err(EZKLError::msg(format!(
                "SRS (k={}) hash not found in public set",
                logrows
            )))
        }
    };

    if hash != *predefined_hash {
        warn!("removing SRS file at {}", path.display());
        std::fs::remove_file(path)?;
        // best-effort: also remove marker if it exists
        let _ = clear_dummy_marker(path.as_path());

        return Err(EZKLError::msg(
            "SRS hash does not match the expected hash. Remote SRS may have been tampered with.",
        ));
    }

    Ok(hash)
}

fn warn_if_public_srs_hash_mismatch(logrows: u32, path: &PathBuf) -> Result<(), EZKLError> {
    if is_marked_dummy(path.as_path()) {
        warn!(
            "SRS at {} is marked as DUMMY ({} exists); skipping public hash check.",
            path.display(),
            srs_dummy_marker_path(path.as_path()).display(),
        );
        return Ok(());
    }

    let Some(expected) = crate::srs_sha::PUBLIC_SRS_SHA256_HASHES.get(&logrows) else {
        warn!(
            "No expected public SRS hash known for k={}; skipping hash check.",
            logrows
        );
        return Ok(());
    };

    let actual = get_file_hash(path)?;
    if actual != *expected {
        warn!(
            "SRS hash mismatch for k={}: expected {}, got {}. Proceeding anyway (non-strict mode).",
            logrows, expected, actual
        );
    }
    Ok(())
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
            return Err(EZKLError::msg(err_string));
        }
    } else {
        return Err(EZKLError::msg(err_string));
    };

    let srs_source = SrsSource::from_env();

    // Always use global cache (~/.ezkl/srs) as the canonical storage, and copy to the
    // user-requested `--srs-path` if provided.
    let cache_path = get_srs_path(k, None);
    let out_path = get_srs_path(k, srs_path.clone());

    let ensure_out_from_cache = || -> Result<(), EZKLError> {
        if out_path == cache_path {
            return Ok(());
        }
        if Path::new(&out_path).exists() {
            return Ok(());
        }
        if !Path::new(&cache_path).exists() {
            return Err(EZKLError::msg(format!(
                "internal error: cache_path {} does not exist after SRS creation",
                cache_path.display()
            )));
        }

        info!(
            "Materializing SRS {} from cache {}",
            out_path.display(),
            cache_path.display()
        );
        copy_srs_and_marker(cache_path.as_path(), out_path.as_path())?;
        Ok(())
    };

    let auto_hash_check = || -> bool {
        let raw = std::env::var("EZKL_SRS_AUTO_HASH_CHECK").unwrap_or_default();
        matches!(raw.trim().to_lowercase().as_str(), "1" | "true" | "yes")
    };

    let maybe_check_cache = || -> Result<(), EZKLError> {
        match srs_source {
            SrsSource::Public => enforce_public_srs_hash(k, &cache_path).map(|_| ()),
            SrsSource::Auto => {
                if is_marked_dummy(cache_path.as_path()) {
                    warn!(
                        "Using locally-generated (dummy) cached SRS at {} ({} exists).",
                        cache_path.display(),
                        srs_dummy_marker_path(cache_path.as_path()).display(),
                    );
                    Ok(())
                } else if auto_hash_check() {
                    warn_if_public_srs_hash_mismatch(k, &cache_path)
                } else {
                    // Skip expensive multi-GB hash scans by default in auto mode.
                    Ok(())
                }
            }
            SrsSource::Dummy => Ok(()),
        }
    };

    // Prefer cache as canonical storage.
    if Path::new(&cache_path).exists() {
        maybe_check_cache()?;
        ensure_out_from_cache()?;
        return Ok(String::new());
    }

    // Cache missing, but output exists: try to backfill cache.
    if Path::new(&out_path).exists() {
        info!(
            "SRS exists at requested path {} but cache {} is missing; backfilling cache.",
            out_path.display(),
            cache_path.display()
        );

        if let Err(e) = copy_srs_and_marker(out_path.as_path(), cache_path.as_path()) {
            warn!(
                "Failed to backfill global SRS cache {} from {}: {} (continuing with out_path only).",
                cache_path.display(),
                out_path.display(),
                e
            );
            return Ok(String::new());
        }

        // If strict hash check fails, remove both cache and out_path to avoid "hard-link keeps corrupted copy alive".
        if let Err(e) = maybe_check_cache() {
            if srs_source == SrsSource::Public {
                let _ = std::fs::remove_file(&out_path);
                let _ = clear_dummy_marker(out_path.as_path());
            }
            return Err(e);
        }

        return Ok(String::new());
    }

    // Cache doesn't exist: materialize it.
    match srs_source {
        SrsSource::Dummy => {
            eprintln!(
                "EZKL_SRS_SOURCE=dummy: generating local/dummy SRS for k={} at {}.\n\
                 This is NOT a trusted-setup SRS and must not be used in production.",
                k,
                cache_path.display()
            );
            let _ = gen_srs_cmd(cache_path.clone(), k)?;
            mark_srs_as_dummy(cache_path.as_path())?;
        }
        SrsSource::Public | SrsSource::Auto => {
            // Try public download, validate, save.
            let srs_uri = format!("{}{}", PUBLIC_SRS_URL, k);
            let tmp_path = PathBuf::from(format!("{}.download", cache_path.display()));

            if tmp_path.exists() {
                let _ = std::fs::remove_file(&tmp_path);
            }

            let dl = download_srs_to_file(&srs_uri, &tmp_path).await;
            if let Err(e) = dl {
                let _ = std::fs::remove_file(&tmp_path);
                if srs_source == SrsSource::Auto {
                    eprintln!(
                        "Public SRS download failed: {e}\n\
                         Falling back to a local/dummy SRS because EZKL_SRS_SOURCE=auto.\n\
                         This is NOT a trusted-setup SRS and must not be used in production."
                    );
                    let _ = gen_srs_cmd(cache_path.clone(), k)?;
                    mark_srs_as_dummy(cache_path.as_path())?;
                    ensure_out_from_cache()?;
                    return Ok(String::new());
                } else {
                    return Err(e);
                }
            }

            // Move into place atomically.
            ensure_parent_dir(&cache_path)?;
            std::fs::rename(&tmp_path, &cache_path).map_err(|e| {
                EZKLError::msg(format!(
                    "failed to move downloaded SRS into cache {}: {e}",
                    cache_path.display()
                ))
            })?;

            // Public download should not be marked as dummy.
            clear_dummy_marker(cache_path.as_path())?;

            // Strict hash check only in Public mode (Auto skips by default; can opt-in via EZKL_SRS_AUTO_HASH_CHECK=1).
            maybe_check_cache()?;
        }
    }

    ensure_out_from_cache()?;
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
        circuit.forward::<KZGCommitmentScheme<Bn256>>(&mut input, vk.as_ref(), None, region_settings)?
    };

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
        .map_err(|e| EZKLError::msg(e.to_string()))?
        .iter()
        .map(|&i| tract_model.outlet_fact(i))
        .collect::<tract_onnx::prelude::TractResult<Vec<_>>>()
        .map_err(|e| EZKLError::msg(e.to_string()))?;

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
    // Write progress output to stderr so stdout can remain machine-readable (e.g., JSON).
    pb.set_draw_target(indicatif::ProgressDrawTarget::stderr());

    // Avoid spamming output when stderr is not a TTY (e.g., piped/captured by a parent process).
    if std::io::stderr().is_terminal() {
        pb.enable_steady_tick(Duration::from_millis(200));
    }
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
    // Write progress output to stderr so stdout can remain machine-readable (e.g., JSON).
    pb.set_draw_target(indicatif::ProgressDrawTarget::stderr());

    // Avoid spamming output when stderr is not a TTY (e.g., piped/captured by a parent process).
    if std::io::stderr().is_terminal() {
        pb.enable_steady_tick(Duration::from_millis(200));
    }
    let sty = ProgressStyle::with_template(
        "[{elapsed_precise}] {bar:40.cyan/blue} {pos:>7}/{len:7} {msg}",
    )
    .unwrap()
    .progress_chars("##-");
    pb.set_style(sty);
    pb
}
