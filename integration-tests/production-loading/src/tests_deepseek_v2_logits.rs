//! The DeepSeek Model Component (`components/deepseek`, checked in as
//! `magnetar-runtime/fixtures/components/deepseek-real.component.wasm`) run
//! by the Reference CPU Provider, against `transformers` 5.23's
//! `DeepseekV2ForCausalLM` in float32, on tiny random checkpoints
//! (`fixtures/deepseek_v2/`, see its README for how they were made). Each
//! has a dense first layer, then routed experts beside two ungated shared
//! experts, and multi-head latent attention in its expanded form.

use crate::logit_comparison::{Format, compare_with_reference};

fn compare(variant: &str) -> f32 {
    let difference =
        compare_with_reference("deepseek_v2", variant, Format::HuggingFace, "deepseek_v2");
    eprintln!("{variant}: max |logit difference| {difference:.3e}");
    difference
}

/// No low-rank query; YaRN with `mscale == mscale_all_dim` (the published
/// DeepSeek-V2 form: cosines and sines unscaled, the softmax rescaled).
#[test]
fn a_deepseek_v2_lite_shaped_bundle_matches_transformers() {
    compare("lite");
}

/// A low-rank query (`q_a_proj`, `q_a_norm`); YaRN with `mscale !=
/// mscale_all_dim`, so the cosines and sines are scaled too; a routed
/// scaling factor of 2.5.
#[test]
fn a_low_rank_query_bundle_with_scaled_routing_matches_transformers() {
    compare("qlora");
}

/// Latent attention with unscaled RoPE.
#[test]
fn a_deepseek_v2_bundle_without_rope_scaling_matches_transformers() {
    compare("plain");
}
