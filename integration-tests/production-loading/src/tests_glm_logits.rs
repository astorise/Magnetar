//! Tiny GLM-4.7-shaped checkpoints, compared with Transformers 5.23 logits
//! for prefill and five cached decode steps.

use crate::logit_comparison::{
    Format, compare_chunked_on, compare_on, compare_with_reference, cpu_providers,
};

#[test]
fn glm_grouped_routing_matches_transformers_for_top_one_and_top_two() {
    for variant in ["tiny-top1", "tiny-top2"] {
        let difference =
            compare_with_reference("glm", variant, Format::HuggingFace, "glm4_moe_lite");
        eprintln!("{variant}: max |logit difference| {difference:.3e}");
    }
}

#[test]
fn glm_grouped_routing_matches_transformers_on_both_cpu_providers() {
    for (provider, name) in cpu_providers() {
        for variant in ["tiny-top1", "tiny-top2"] {
            let difference = compare_on(
                provider.clone(),
                name,
                "glm",
                variant,
                Format::HuggingFace,
                "glm4_moe_lite",
            );
            eprintln!("{name} {variant}: max |logit difference| {difference:.3e}");
        }
    }
}

#[test]
fn glm_prefill_in_multiple_passes_matches_transformers() {
    for variant in ["tiny-top1", "tiny-top2"] {
        let (difference, chunk) = compare_chunked_on(
            std::sync::Arc::new(magnetar_runtime::ReferenceCpuProvider::new()),
            "magnetar-reference-cpu",
            "glm",
            variant,
            Format::HuggingFace,
            "glm4_moe_lite",
            2,
        );
        assert_eq!(chunk, 2);
        eprintln!("{variant}: chunked max |logit difference| {difference:.3e}");
    }
}
