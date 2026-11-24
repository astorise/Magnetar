//! Tiny Mistral-shaped full/sliding attention through real ingestion, a
//! Component Model artifact, the graph engine and the Reference CPU.

use crate::logit_comparison::{Format, compare_with_reference};

#[test]
fn mistral_prefill_and_cached_decode_match_independent_logits() {
    for variant in ["full", "window3"] {
        let difference = compare_with_reference("mistral", variant, Format::HuggingFace, "mistral");
        eprintln!("Mistral {variant}: maximum logit difference {difference:.3e}");
    }
}
