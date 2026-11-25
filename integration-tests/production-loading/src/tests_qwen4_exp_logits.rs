//! The Qwen Model Component (`components/qwen`, checked in as
//! `magnetar-runtime/fixtures/components/qwen-real.component.wasm`) run by
//! the Reference CPU Provider on a tiny random Qwen4-Exp checkpoint
//! (`fixtures/qwen4_exp/`, see its README), against `transformers` 5.23's
//! `Qwen4ExpForCausalLM` in float32 (`support-qwen38-flash-next`).
//!
//! Four residual streams mixed by learned gates, gated-delta linear
//! attention with a gated query attention, and a mixture of four experts
//! with a gated shared expert in every layer. The Runtime keeps the
//! mixers' convolution and recurrent states across the prefill and every
//! decode step, and KV history for the attention layers only.

use crate::logit_comparison::{Format, compare_on, compare_with_reference, cpu_providers};

#[test]
fn a_qwen4_exp_dense_bundle_matches_transformers() {
    let difference =
        compare_with_reference("qwen4_exp", "dense", Format::HuggingFace, "qwen4_exp_text");
    eprintln!("dense: max |logit difference| {difference:.3e}");
}

/// The linear layers' output gate a sigmoid (`output_gate_type`).
#[test]
fn a_qwen4_exp_sigmoid_gate_bundle_matches_transformers() {
    for (provider, label) in cpu_providers() {
        let difference = compare_on(
            provider,
            label,
            "qwen4_exp",
            "sigmoid",
            Format::HuggingFace,
            "qwen4_exp_text",
        );
        eprintln!("sigmoid gate, {label}: max |logit difference| {difference:.3e}");
    }
}

/// A hashed n-gram embedding added in the first layer, its token context
/// and convolution history carried as recurrent states.
#[test]
fn a_qwen4_exp_ngram_embedding_bundle_matches_transformers() {
    for (provider, label) in cpu_providers() {
        let difference = compare_on(
            provider,
            label,
            "qwen4_exp",
            "ple",
            Format::HuggingFace,
            "qwen4_exp_text",
        );
        eprintln!("n-gram embedding, {label}: max |logit difference| {difference:.3e}");
    }
}

/// An indexer that prunes: a query keeps at most 9 tokens (budget 8 in
/// blocks of 2), so from the tenth visible token attention is sparse in the
/// prefill and in every decode step. Each indexed layer keeps its raw index
/// keys as a KV store of its own.
#[test]
fn a_qwen4_exp_pruning_indexer_bundle_matches_transformers() {
    for (provider, label) in cpu_providers() {
        let difference = compare_on(
            provider,
            label,
            "qwen4_exp",
            "prune",
            Format::HuggingFace,
            "qwen4_exp_text",
        );
        eprintln!("pruning indexer, {label}: max |logit difference| {difference:.3e}");
    }
}

/// The same checkpoint on `providers/cpu`'s Reference CPU Provider and its
/// optimized CPU Provider.
#[test]
fn a_qwen4_exp_dense_bundle_matches_transformers_on_the_cpu_providers() {
    for (provider, label) in cpu_providers() {
        let difference = compare_on(
            provider,
            label,
            "qwen4_exp",
            "dense",
            Format::HuggingFace,
            "qwen4_exp_text",
        );
        eprintln!("{label}: max |logit difference| {difference:.3e}");
    }
}
