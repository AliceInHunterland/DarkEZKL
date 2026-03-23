#![cfg(all(feature = "ezkl", not(target_arch = "wasm32")))]

use ezkl::graph::GraphSettings;
use lazy_static::lazy_static;
use std::env::var;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Once;
use tempdir::TempDir;

static COMPILE: Once = Once::new();
static PY_SETUP: Once = Once::new();

const TEST_BINARY: &str = "test-runs/ezkl";

lazy_static! {
    static ref CARGO_TARGET_DIR: String =
        var("CARGO_TARGET_DIR").unwrap_or_else(|_| "./target".to_string());
}

fn ezkl_bin() -> String {
    format!("{}/{}", *CARGO_TARGET_DIR, TEST_BINARY)
}

fn run_cmd(mut cmd: Command, what: &str) {
    let status = cmd
        .status()
        .unwrap_or_else(|e| panic!("failed to run {what}: {e}"));
    assert!(status.success(), "{what} failed with status: {status}");
}

fn init_binary() {
    COMPILE.call_once(|| {
        // Build the CLI binary used by integration tests.
        // Keep it consistent with existing integration test build strategy.
        let mut args: Vec<String> = vec![
            "build".to_string(),
            "--profile=test-runs".to_string(),
            "--bin".to_string(),
            "ezkl".to_string(),
        ];

        // Mirror feature-gated builds so `test-runs/ezkl` matches the crate's build.
        // (These are compile-time cfgs.)
        let mut features: Vec<&str> = Vec::new();
        if cfg!(feature = "gpu-accelerated") {
            // ezkl uses "icicle" feature for GPU acceleration in the CLI build.
            features.push("icicle");
        }
        if cfg!(feature = "macos-metal") {
            features.push("macos-metal");
        }
        if cfg!(feature = "reusable-verifier") {
            features.push("reusable-verifier");
        }

        if !features.is_empty() {
            args.push("--features".to_string());
            args.push(features.join(","));
        }

        run_cmd(
            Command::new("cargo").args(&args),
            "cargo build ezkl (test-runs profile)",
        );
    });
}

fn ensure_python_deps() {
    PY_SETUP.call_once(|| {
        // Prefer to avoid pip if deps already exist.
        let check = Command::new("python")
            .args(["-c", "import numpy, onnx"])
            .status()
            .map(|s| s.success())
            .unwrap_or(false);

        if check {
            return;
        }

        // Minimal deps needed to generate the ONNX model in-test.
        // (We intentionally do NOT require torch here.)
        run_cmd(
            Command::new("pip")
                .args(["install", "--quiet", "numpy", "onnx"])
                .stdout(std::process::Stdio::null()),
            "pip install numpy onnx",
        );
    });
}

fn write_matmul_onnx_and_input(model_dir: &Path) {
    fs::create_dir_all(model_dir).expect("failed to create model dir");

    // Generate a minimal ONNX MatMul model:
    //   output = MatMul(input, W)
    // where W is a constant initializer.
    //
    // Also write input.json in EZKL's expected format:
    //   { "input_data": [ [ ...flattened... ] ] }
    let py = r#"
import json
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper

# input: [1, 2]
X = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 2])
# output: [1, 2]
Y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 2])

# constant weight: [2, 2]
W = np.array([[1.0, 2.0],
              [3.0, 4.0]], dtype=np.float32)
W_init = numpy_helper.from_array(W, name="W")

node = helper.make_node("MatMul", inputs=["input", "W"], outputs=["output"])

graph = helper.make_graph(
    [node],
    "ezkl_probabilistic_matmul",
    [X],
    [Y],
    [W_init],
)

model = helper.make_model(
    graph,
    opset_imports=[helper.make_operatorsetid("", 10)]
)

onnx.checker.check_model(model)
onnx.save(model, "network.onnx")

# deterministic input
x = np.array([[0.25, -0.75]], dtype=np.float32).reshape(-1).tolist()
json.dump({"input_data": [x]}, open("input.json", "w"))
"#;

    run_cmd(
        Command::new("python")
            .current_dir(model_dir)
            .args(["-c", py]),
        "python generate matmul network.onnx + input.json",
    );
}

fn init_params(settings_path: &Path) -> u32 {
    let settings_str = fs::read_to_string(settings_path)
        .unwrap_or_else(|_| panic!("failed to read {settings_path:?}"));
    let settings: GraphSettings =
        serde_json::from_str(&settings_str).expect("failed to parse settings.json");
    let logrows = settings.run_args.logrows;

    run_cmd(
        Command::new(ezkl_bin()).args(["get-srs", "--logrows", &format!("{logrows}")]),
        "ezkl get-srs",
    );

    logrows
}

#[test]
fn probabilistic_matmul_end_to_end() {
    init_binary();
    ensure_python_deps();

    let tmp = TempDir::new("ezkl_probabilistic_matmul").expect("failed to create tempdir");
    let model_dir: PathBuf = tmp.path().join("matmul");

    write_matmul_onnx_and_input(&model_dir);

    let onnx_path = model_dir.join("network.onnx");
    let input_path = model_dir.join("input.json");
    let settings_path = model_dir.join("settings.json");
    let compiled_path = model_dir.join("network.compiled");
    let witness_path = model_dir.join("witness.json");
    let pk_path = model_dir.join("key.pk");
    let vk_path = model_dir.join("key.vk");
    let proof_path = model_dir.join("proof.pf");

    // 1) gen-settings (with probabilistic mode enabled)
    run_cmd(
        Command::new(ezkl_bin()).args([
            "gen-settings",
            "-M",
            onnx_path.to_str().unwrap(),
            &format!("--settings-path={}", settings_path.to_str().unwrap()),
            "--logrows=15",
            "--num-inner-cols=2",
            "--input-visibility=public",
            "--param-visibility=fixed",
            "--output-visibility=public",
            "--execution-mode",
            "probabilistic",
            "--prob-ops",
            "matmul",
            "--prob-k",
            "8",
            "--prob-seed-mode",
            "fiat_shamir",
        ]),
        "ezkl gen-settings (probabilistic)",
    );

    // 2) calibrate-settings (small model; keep it fast)
    run_cmd(
        Command::new(ezkl_bin()).args([
            "calibrate-settings",
            "--data",
            input_path.to_str().unwrap(),
            "-M",
            onnx_path.to_str().unwrap(),
            &format!("--settings-path={}", settings_path.to_str().unwrap()),
            "--target=resources",
            "--lookup-safety-margin=2",
        ]),
        "ezkl calibrate-settings",
    );

    // 3) compile-circuit
    run_cmd(
        Command::new(ezkl_bin()).args([
            "compile-circuit",
            "-M",
            onnx_path.to_str().unwrap(),
            "--compiled-circuit",
            compiled_path.to_str().unwrap(),
            &format!("--settings-path={}", settings_path.to_str().unwrap()),
        ]),
        "ezkl compile-circuit",
    );

    // Ensure we have the SRS needed for this logrows.
    let _logrows = init_params(&settings_path);

    // 4) gen-witness
    run_cmd(
        Command::new(ezkl_bin()).args([
            "gen-witness",
            "-D",
            input_path.to_str().unwrap(),
            "-M",
            compiled_path.to_str().unwrap(),
            "-O",
            witness_path.to_str().unwrap(),
        ]),
        "ezkl gen-witness",
    );

    // 5) setup
    run_cmd(
        Command::new(ezkl_bin()).args([
            "setup",
            "-M",
            compiled_path.to_str().unwrap(),
            "--pk-path",
            pk_path.to_str().unwrap(),
            "--vk-path",
            vk_path.to_str().unwrap(),
            "--disable-selector-compression",
        ]),
        "ezkl setup",
    );

    // 6) prove
    run_cmd(
        Command::new(ezkl_bin()).args([
            "prove",
            "-W",
            witness_path.to_str().unwrap(),
            "-M",
            compiled_path.to_str().unwrap(),
            "--proof-path",
            proof_path.to_str().unwrap(),
            "--pk-path",
            pk_path.to_str().unwrap(),
            "--check-mode=safe",
        ]),
        "ezkl prove",
    );

    // 7) swap-proof-commitments (used by other integration tests before verify)
    run_cmd(
        Command::new(ezkl_bin()).args([
            "swap-proof-commitments",
            "--proof-path",
            proof_path.to_str().unwrap(),
            "--witness-path",
            witness_path.to_str().unwrap(),
        ]),
        "ezkl swap-proof-commitments",
    );

    // 8) verify
    run_cmd(
        Command::new(ezkl_bin()).args([
            "verify",
            &format!("--settings-path={}", settings_path.to_str().unwrap()),
            "--proof-path",
            proof_path.to_str().unwrap(),
            "--vk-path",
            vk_path.to_str().unwrap(),
        ]),
        "ezkl verify",
    );

    tmp.close().expect("failed to close tempdir");
}
