//! OLMo 3-shaped inference compared to independent NumPy f32 logits.

use crate::logit_comparison::{Format, compare_with_reference};

#[test]
#[ignore = "production fixture currently requires a pre-norm decoder tensor inventory"]
fn olmo3_prefill_and_cached_decode_match_independent_logits() {
    let difference = compare_with_reference("olmo3", "tiny", Format::HuggingFace, "olmo3");
    eprintln!("OLMo 3 maximum logit difference: {difference:.3e}");
}

#[test]
fn olmo3_and_smollm2_unsupported_or_malformed_activations_fail_before_inference() {
    for path in [
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/fixtures/olmo3/tiny/config.json"
        ),
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/fixtures/smollm2/tiny/config.json"
        ),
    ] {
        let mut value: serde_json::Value =
            serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
        for activation in [
            serde_json::json!("gelu"),
            serde_json::json!("unknown"),
            serde_json::Value::Null,
        ] {
            value["hidden_act"] = activation;
            assert!(
                magnetar_loader_huggingface::parse_config(&serde_json::to_vec(&value).unwrap())
                    .is_err()
            );
        }
    }
}
