//! A tiny synthetic Llama with one block-scaled E4M3FN projection.

#[test]
fn block_scaled_fp8_prefill_and_cached_decode_match_transformers() {
    let component = crate::test_support::smollm2_component();
    let difference =
        crate::logit_comparison::compare_with_component("fp8", "tiny", "llama", component);
    eprintln!("Mixed FP8 Llama max logit difference: {difference:.3e}");
}

#[test]
fn fp8_latent_and_expert_weights_match_transformers() {
    let component = crate::test_support::real_component_for_architecture_family("deepseek_v2");
    let difference = crate::logit_comparison::compare_with_component(
        "fp8",
        "deepseek",
        "deepseek_v2",
        component,
    );
    eprintln!("Mixed FP8 DeepSeek max logit difference: {difference:.3e}");
}
