//! Tiny Kimi-shaped FP8 checkpoints compared with the pinned upstream model.

use crate::logit_comparison::{Format, compare_on, compare_with_reference, cpu_providers};

#[test]
fn kimi_mixed_fp8_matches_pinned_upstream_for_top_one_and_top_two() {
    for variant in ["tiny-top1", "tiny-top2"] {
        let difference = compare_with_reference("kimi", variant, Format::HuggingFace, "kimi_k2");
        eprintln!("{variant}: max |logit difference| {difference:.3e}");
    }
}

#[test]
fn kimi_mixed_fp8_matches_pinned_upstream_on_both_cpu_providers() {
    for (provider, name) in cpu_providers() {
        for variant in ["tiny-top1", "tiny-top2"] {
            let difference = compare_on(
                provider.clone(),
                name,
                "kimi",
                variant,
                Format::HuggingFace,
                "kimi_k2",
            );
            eprintln!("{name} {variant}: max |logit difference| {difference:.3e}");
        }
    }
}
