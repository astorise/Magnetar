//! The Qwen Model Component (`components/qwen`, checked in as
//! `magnetar-runtime/fixtures/components/qwen-real.component.wasm`) run by
//! the Reference CPU Provider on a tiny random Qwen3.5 hybrid checkpoint
//! (`fixtures/qwen3_5/`, see its README), against `transformers` 5.23's
//! `Qwen3_5ForCausalLM` in float32 (`support-qwen38-flash-next`).
//!
//! Gated-delta linear-attention layers alternate with gated full-attention
//! layers. The Runtime keeps the mixers' convolution and recurrent states
//! across the prefill and every decode step, and KV history for the
//! attention layers only.

use crate::logit_comparison::{Format, compare_with_reference};

#[test]
fn a_qwen3_5_hybrid_bundle_matches_transformers() {
    let difference =
        compare_with_reference("qwen3_5", "hybrid", Format::HuggingFace, "qwen3_5_text");
    eprintln!("hybrid: max |logit difference| {difference:.3e}");
}
