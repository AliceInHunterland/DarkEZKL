use super::*;

#[test]
fn test_graph_settings_serialization_roundtrip() {
    use crate::{
        CheckMode, ExecutionMode, ProbOp, ProbOps, ProbSeedMode, ProbabilisticSettings, RunArgs,
    };

    // Create a test GraphSettings with nested structs
    let original = GraphSettings {
        run_args: RunArgs {
            execution_mode: ExecutionMode::Probabilistic,
            prob_k: 20,
            prob_seed_mode: ProbSeedMode::FiatShamir,
            prob_ops: ProbOps(vec![ProbOp::MatMul, ProbOp::Gemm]),
            ..RunArgs::default()
        },
        execution_mode: ExecutionMode::Probabilistic,
        prob_k: 20,
        prob_ops: ProbOps(vec![ProbOp::MatMul, ProbOp::Gemm]),
        probabilistic_settings: ProbabilisticSettings {
            k_repetitions: 20,
            seed_mode: ProbSeedMode::FiatShamir,
        },
        num_rows: 1000,
        total_assignments: 500,
        total_const_size: 100,
        dynamic_lookup_params: DynamicLookupParams {
            total_dynamic_col_size: 42,
            max_dynamic_input_len: 128,
            num_dynamic_lookups: 5,
        },
        shuffle_params: ShuffleParams {
            num_shuffles: 3,
            total_shuffle_col_size: 256,
        },
        einsum_params: EinsumParams::default(),
        model_instance_shapes: vec![vec![1, 2, 3]],
        model_output_scales: vec![],
        model_input_scales: vec![],
        module_sizes: ModuleSizes::default(),
        required_lookups: vec![],
        required_range_checks: vec![],
        check_mode: CheckMode::SAFE,
        version: "1.0.0".to_string(),
        num_blinding_factors: Some(5),
        timestamp: Some(123456789),
        input_types: None,
        output_types: None,
    };

    // Test 1: JSON serialization roundtrip with flattened format
    let json_str = serde_json::to_string_pretty(&original).unwrap();
    println!("JSON serialized (flattened):\n{}", json_str);

    // Verify the JSON contains flattened fields
    assert!(json_str.contains("\"total_dynamic_col_size\": 42"));
    assert!(json_str.contains("\"max_dynamic_input_len\": 128"));
    assert!(json_str.contains("\"num_dynamic_lookups\": 5"));
    assert!(json_str.contains("\"num_shuffles\": 3"));
    assert!(json_str.contains("\"total_shuffle_col_size\": 256"));

    // Verify probabilistic execution config is present at top-level.
    assert!(json_str.contains("\"execution_mode\""));
    assert!(json_str.contains("\"prob_k\": 20"));
    assert!(json_str.contains("\"prob_ops\""));
    assert!(json_str.contains("matmul"));
    assert!(json_str.contains("\"probabilistic_settings\""));
    assert!(json_str.contains("\"k_repetitions\": 20"));
    assert!(json_str.contains("\"seed_mode\""));

    // (Back-compat) run_args still contains these fields too.
    assert!(json_str.contains("\"prob_seed_mode\""));

    // Verify the JSON does NOT contain nested structs
    assert!(!json_str.contains("\"dynamic_lookup_params\""));
    assert!(!json_str.contains("\"shuffle_params\""));

    // Deserialize from JSON
    let deserialized: GraphSettings = serde_json::from_str(&json_str).unwrap();
    assert_eq!(original, deserialized);

    // now do JSON bytes
    let json_bytes = serde_json::to_vec(&original).unwrap();
    let deserialized_from_bytes: GraphSettings = serde_json::from_slice(&json_bytes).unwrap();
    assert_eq!(original, deserialized_from_bytes);

    // Test 2: Bincode serialization roundtrip
    let bincode_data = bincode::serialize(&original).unwrap();
    let bincode_deserialized: GraphSettings = bincode::deserialize(&bincode_data).unwrap();
    assert_eq!(original, bincode_deserialized);

    // Test 3: Backwards compatibility - deserialize old nested format
    let old_format_json = r#"{
"run_args": {
    "tolerance": {
        "val": 0.0,
        "scale": 1.0
    },
    "input_scale": 0,
    "param_scale": 0,
    "scale_rebase_multiplier": 10,
    "lookup_range": [
        0,
        0
    ],
    "logrows": 6,
    "num_inner_cols": 2,
    "variables": [
        [
            "batch_size",
            1
        ]
    ],
    "input_visibility": "Private",
    "output_visibility": "Public",
    "param_visibility": "Private",
    "rebase_frac_zero_constants": false,
    "check_mode": "UNSAFE",
    "commitment": "KZG",
    "decomp_base": 128,
    "decomp_legs": 2,
    "bounded_log_lookup": false,
    "ignore_range_check_inputs_outputs": false,
    "disable_freivalds": false
},
"num_rows": 236,
"total_assignments": 472,
"total_const_size": 4,
"total_dynamic_col_size": 0,
"max_dynamic_input_len": 0,
"num_dynamic_lookups": 0,
"num_shuffles": 0,
"total_shuffle_col_size": 0,
"model_instance_shapes": [
    [
        1,
        4
    ]
],
"model_output_scales": [
    0
],
"model_input_scales": [
    0
],
"module_sizes": {
    "polycommit": [],
    "poseidon": [
        0,
        [
            0
        ]
    ]
},
"required_lookups": [],
"required_range_checks": [
    [
        -1,
        1
    ],
    [
        0,
        127
    ]
],
"check_mode": "UNSAFE",
"version": "0.0.0",
"num_blinding_factors": null,
"timestamp": 1741214578354
}"#;

    let _backwards_compatible: GraphSettings = serde_json::from_str(old_format_json).unwrap();
}
