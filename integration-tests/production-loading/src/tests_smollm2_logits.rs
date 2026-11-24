//! Runs a tiny random SmolLM2-shaped Llama checkpoint through the real loader,
//! independently compiled SmolLM2 Component and Reference CPU. Its greedy
//! prefill and cached-decode logits are compared with Transformers' full
//! sequence recomputation; it is not the published model's weights.

#[test]
fn smollm2_tiny_prefill_and_cached_decode_match_transformers() {
    let component = crate::test_support::smollm2_component();
    let max_difference =
        crate::logit_comparison::compare_with_component("smollm2", "tiny", "llama", component);
    eprintln!("SmolLM2-shaped Llama max logit difference: {max_difference:.3e}");
}
