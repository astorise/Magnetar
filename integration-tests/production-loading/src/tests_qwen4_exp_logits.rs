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
    Format, TableAccess, Timings, compare_on, compare_with_reference, compare_with_tables,
    cpu_providers,
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
        let (difference, usage, _) = compare_with_tables(
            provider,
            label,
            "qwen4_exp",
            "ple",
            "qwen4_exp_text",
            TableAccess::Mapped(".ple.table"),
            false,
        );
        let usage = usage.expect("the tables were counted");
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

/// The same checkpoint through the loader's real mapping, with nothing
/// standing between the Runtime and the loader (`support-qwen38-flash-next`
/// task 4.4): the loader maps the n-gram table from the bundle's weight
/// files, stacked from its shards, the Runtime never reads it whole, and the
/// logits are `transformers`'.
#[test]
fn a_qwen4_exp_ngram_table_is_mapped_from_the_bundle_files() {
    for (provider, label) in cpu_providers() {
        let (difference, usage, _) = compare_with_tables(
            provider,
            label,
            "qwen4_exp",
            "ple",
            "qwen4_exp_text",
            TableAccess::Observed(".ple.table"),
            false,
        );
        let usage = usage.expect("the tables were observed");
        let mapped = usage.mapped.lock().unwrap().clone();
        assert_eq!(
            mapped.len(),
            1,
            "{label}: the loader maps the one n-gram table: {mapped:?}"
        );
        assert!(
            usage.read_whole.lock().unwrap().is_empty(),
            "{label}: no table is read whole: {:?}",
            usage.read_whole.lock().unwrap()
        );
        eprintln!(
            "n-gram table mapped by the loader, {label}: max |logit difference| {difference:.3e}"
        );
    }
}

fn milliseconds(duration: std::time::Duration) -> f64 {
    duration.as_secs_f64() * 1e3
}

fn report(label: &str, what: &str, timings: &Timings) {
    let decode = &timings.steps[1..];
    let mean = decode.iter().map(|step| milliseconds(*step)).sum::<f64>() / decode.len() as f64;
    let worst = decode
        .iter()
        .map(|step| milliseconds(*step))
        .fold(0.0, f64::max);
    eprintln!(
        "{label:32} {what:20} ingest {:7.0} ms  load {:7.1} ms  prefill {:6.1} ms  decode mean {mean:6.2} ms  worst {worst:6.2} ms",
        milliseconds(timings.ingest),
        milliseconds(timings.load),
        milliseconds(timings.steps[0]),
    );
}

/// What reading an n-gram table by row costs, against the same table staged
/// whole (`support-qwen38-flash-next` task 4.4, design D8). Needs bundles of
/// tables of millions of rows, made where they are measured:
///
/// ```text
/// make_tiny_qwen4_exp.py <dir> big 120000   # 245 MB: under the operator's element cap
/// make_tiny_qwen4_exp.py <dir> big          # 820 MB: past it
/// qwen4_exp_reference_logits.py <dir> 12 <dir>/reference.json <prompt ids>
/// ```
///
/// `MAGNETAR_QWEN4_EXP_MID_PLE` names the first (read by row, cold and warm,
/// and staged whole) and `MAGNETAR_QWEN4_EXP_BIG_PLE` the second (only a
/// table read by row can run it: the Reference semantics refuse a tensor of
/// more than 67 million elements). Every run matches `transformers`; the
/// numbers go to the output.
#[test]
#[ignore = "needs MAGNETAR_QWEN4_EXP_MID_PLE and/or MAGNETAR_QWEN4_EXP_BIG_PLE: bundles of tables of millions of rows"]
fn an_ngram_table_read_by_row_is_measured_against_one_staged_whole() {
    let mid = std::env::var("MAGNETAR_QWEN4_EXP_MID_PLE").ok();
    let big = std::env::var("MAGNETAR_QWEN4_EXP_BIG_PLE").ok();
    assert!(
        mid.is_some() || big.is_some(),
        "name a bundle with MAGNETAR_QWEN4_EXP_MID_PLE or MAGNETAR_QWEN4_EXP_BIG_PLE"
    );
    for (provider, label) in cpu_providers() {
        for (directory, staged_too) in [(mid.as_deref(), true), (big.as_deref(), false)] {
            let Some(directory) = directory else { continue };
            let run = |access: TableAccess, cold: bool| {
                compare_with_tables(
                    provider.clone(),
                    label,
                    "qwen4_exp",
                    directory,
                    "qwen4_exp_text",
                    access,
                    cold,
                )
            };
            let size = if staged_too { "245 MB" } else { "820 MB" };

            let (_, usage, timings) = run(TableAccess::Observed(".ple.table"), true);
            let usage = usage.unwrap();
            assert_eq!(usage.mapped.lock().unwrap().len(), 1, "the table is mapped");
            assert!(usage.read_whole.lock().unwrap().is_empty());
            report(label, &format!("{size} by row, cold"), &timings);

            let (_, _, warm) = run(TableAccess::Observed(".ple.table"), false);
            report(label, &format!("{size} by row, warm"), &warm);

            if staged_too {
                let (_, usage, staged) = run(TableAccess::Dense(".ple.table"), false);
                let usage = usage.unwrap();
                assert!(usage.mapped.lock().unwrap().is_empty());
                assert_eq!(
                    usage.read_whole.lock().unwrap().len(),
                    1,
                    "the baseline reads the table whole"
                );
                report(label, &format!("{size} staged whole"), &staged);
            }
        }
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
