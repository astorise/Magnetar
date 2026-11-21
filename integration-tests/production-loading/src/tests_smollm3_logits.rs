//! A mixed RoPE/NoPE SmolLM3 checkpoint through ingestion, the real Wasm
//! component and CPU execution, compared with Transformers full recomputation.

#[test]
fn smollm3_mixed_rope_prefill_and_cached_decode_match_transformers() {
    let component = crate::test_support::smollm2_component();
    let difference =
        crate::logit_comparison::compare_with_component("smollm3", "tiny", "smollm3", component);
    eprintln!("SmolLM3 maximum logit difference: {difference:.3e}");
}
