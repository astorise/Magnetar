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

use crate::logit_comparison::{
    Format, compare_on, compare_with_mapped_tables, compare_with_reference, cpu_providers,
};

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

/// The same checkpoint with its n-gram table mapped, never staged
/// (`support-qwen38-flash-next` task 4.4, design D8): the Runtime reads only
/// the rows each step names, hinting all of a step's rows to the mapping
/// before its layers run, and the logits are the dense table's.
#[test]
fn a_qwen4_exp_ngram_table_read_by_row_matches_transformers() {
    use std::sync::atomic::Ordering;
    for (provider, label) in cpu_providers() {
        let (difference, usage) = compare_with_mapped_tables(
            provider,
            label,
            "qwen4_exp",
            "ple",
            "qwen4_exp_text",
            ".ple.table",
        );
        let mapped = usage.mapped.lock().unwrap().clone();
        assert_eq!(
            mapped.len(),
            1,
            "{label}: the one n-gram table is mapped: {mapped:?}"
        );
        assert!(
            usage.read_whole.lock().unwrap().is_empty(),
            "{label}: no table is read whole: {:?}",
            usage.read_whole.lock().unwrap()
        );
        let reads = usage.reads.load(Ordering::Relaxed);
        let prefetches = usage.prefetches.load(Ordering::Relaxed);
        assert!(reads > 0, "{label}: rows are read from the mapping");
        // One hint per step (the prefill, then a decode step for every
        // generated token but the first), and in all `heads` rows per token
        // run and per n-gram layer: no row more than the steps name.
        let reference: serde_json::Value = serde_json::from_slice(
            &std::fs::read(format!(
                "{}/fixtures/qwen4_exp/ple/reference.json",
                env!("CARGO_MANIFEST_DIR")
            ))
            .unwrap(),
        )
        .unwrap();
        let prompt = reference["prompt_token_ids"].as_array().unwrap().len() as u64;
        let generated = reference["generated_token_ids"].as_array().unwrap().len() as u64;
        let heads = 2 * 8; // (ngram_size - 1) * heads_per_ngram
        assert_eq!(prefetches, generated, "{label}: one hint per step");
        assert_eq!(
            reads,
            (prompt + generated - 1) * heads,
            "{label}: `heads` rows per token run, no more"
        );
        eprintln!(
            "n-gram table by row, {label}: max |logit difference| {difference:.3e}, \
             {reads} row reads ({} bytes), {prefetches} prefetches",
            usage.bytes_read.load(Ordering::Relaxed)
        );
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

/// The `dense` checkpoint with its routed experts stacked, as `transformers`
/// holds them and other checkpoints store them (`mlp.experts.gate_up_proj`
/// and `mlp.experts.down_proj`): the same logits as with one tensor per expert.
#[test]
fn a_qwen4_exp_stacked_experts_bundle_matches_transformers() {
    for (provider, label) in cpu_providers() {
        let difference = compare_on(
            provider,
            label,
            "qwen4_exp",
            "stacked",
            Format::HuggingFace,
            "qwen4_exp_text",
        );
        eprintln!("stacked experts, {label}: max |logit difference| {difference:.3e}");
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
