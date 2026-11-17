//! The Gemma 4 Model Component (`components/gemma`, checked in as
//! `magnetar-runtime/fixtures/components/gemma-real.component.wasm`) run by
//! the Reference CPU Provider, against `transformers` 5.23's
//! `Gemma4ForCausalLM` in float32, on tiny random checkpoints
//! (`fixtures/gemma4/`, see its README for how they were made).
//!
//! Each case ingests a checkpoint through a real loader, loads it with the
//! Component its architecture family selects, generates 12 greedy tokens
//! (one prefill and eleven decode steps with the KV cache) and compares
//! every step's top-8 logits with the reference: identical tokens, within
//! 1e-3. The quantized GGUF references are computed on the dequantized
//! weights, so they test the handling of the format, not the quantization
//! error.

use crate::logit_comparison::{Format, compare_on, compare_with_reference};
use magnetar_integration_resident_provider::ResidentReferenceCpuProvider;

fn compare(variant: &str, format: Format, family: &str) -> f32 {
    compare_with_reference("gemma4", variant, format, family)
}

/// Sliding and full layers with their own head widths, V from K, partial
/// RoPE, q/k/v norms, sandwich norms, `layer_scalar`, GeGLU and the
/// logit soft-cap, from a Hugging Face bundle with a nested text config.
#[test]
fn a_dense_gemma4_bundle_matches_transformers() {
    let difference = compare("dense", Format::HuggingFace, "gemma4");
    eprintln!("dense: max |logit difference| {difference:.3e}");
}

/// The same decoder as a `gemma4_unified` bundle (the 12B's model type),
/// whose vision tensor the loader leaves out.
#[test]
fn a_unified_gemma4_bundle_matches_transformers() {
    let difference = compare("unified", Format::HuggingFace, "gemma4_unified");
    eprintln!("unified: max |logit difference| {difference:.3e}");
}

/// Quantized `gemma4` GGUF files (`Q4_0` projections and experts, `Q8_0`
/// embedding and per-layer tables): the layout from the metadata, shared
/// KV with a wider MLP, per-layer input tables cut per layer, and fused
/// experts restacked with the router fold and per-expert scale.
#[test]
fn quantized_gemma4_gguf_files_match_transformers() {
    for variant in ["gguf-dense", "gguf-shared", "gguf-ple", "gguf-moe"] {
        let difference = compare(variant, Format::Gguf, "gemma4");
        eprintln!("{variant}: max |logit difference| {difference:.3e}");
    }
}

/// Every checkpoint above with every value held "on the Device"
/// ([`ResidentReferenceCpuProvider`]), so that no step of the Gemma 4
/// graph asks a device Provider for an intermediate value's host bytes.
#[test]
fn the_tiny_gemma4_checkpoints_match_transformers_with_resident_values() {
    for (variant, format, family) in [
        ("dense", Format::HuggingFace, "gemma4"),
        ("unified", Format::HuggingFace, "gemma4_unified"),
        ("gguf-dense", Format::Gguf, "gemma4"),
        ("gguf-shared", Format::Gguf, "gemma4"),
        ("gguf-ple", Format::Gguf, "gemma4"),
        ("gguf-moe", Format::Gguf, "gemma4"),
    ] {
        let difference = compare_on(
            std::sync::Arc::new(ResidentReferenceCpuProvider::new()),
            "magnetar-reference-cpu-resident",
            "gemma4",
            variant,
            format,
            family,
        );
        eprintln!("resident {variant}: max |logit difference| {difference:.3e}");
    }
}
