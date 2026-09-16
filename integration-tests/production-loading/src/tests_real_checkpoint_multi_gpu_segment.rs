//! `add-real-multi-device-model-instance-placement`, Phase C: the real,
//! public Qwen2.5-0.5B-Instruct checkpoint (the same one
//! `tests_real_checkpoint_smoke.rs` verifies), split by real decoder-layer
//! range across two real GPUs, generating real multi-step greedy text --
//! not a synthetic fixture, not a single Device.
//!
//! Two segment `ModelInstance`s, each loaded from the SAME real ingested
//! checkpoint but materializing only its own layer range's real weights
//! (`load_production_qwen_instance_segment_for_provider`, streamed
//! straight from the real Safetensors file -- the other segment's weights
//! are never fetched): real GPU 0 holds decoder layers `[0, mid)`, real
//! GPU 1 holds `[mid, num_hidden_layers)`. Each real generation step
//! (prefill, then every decode step) runs GPU 0's own segment graph
//! first, moves its real raw hidden-state output to GPU 1 through an
//! explicit Host round trip (`run_first_native_graph_segment_dispatch`'s
//! own doc comment on why not yet real peer-to-peer for this boundary),
//! then runs GPU 1's own segment graph to produce that step's real
//! logits -- greedy-argmax sampled here directly (this test's own
//! minimal decode loop, not the full production `GenerationParameters`/
//! `StopConditions` machinery `run_production_qwen_generation` wraps,
//! which is written against one full `ModelInstance`, not two segment
//! ones split across Providers).
//!
//! Compared against the exact same prompt run through the real, full,
//! unsegmented graph on one real GPU alone
//! (`run_production_qwen_generation_for_provider`, the same entrypoint
//! `tests_real_checkpoint_smoke.rs` already proves against this
//! checkpoint) -- the two-real-GPU segmented pipeline must generate
//! exactly the same real token ids.
//!
//! `#[ignore]`d and named `real_public_checkpoint_*` like its siblings so
//! it is picked up automatically by `gpu-runner-smoke.yml`'s existing
//! `real_public_checkpoint -- --include-ignored` step on the real
//! `arc-gpu-magnetar` two-GPU CI runner, which already downloads this
//! exact checkpoint for that step. Skips cleanly (returns without
//! assertions) on any host without two real CUDA devices.

use crate::test_support::register_real_qwen_component;
use crate::tests_real_checkpoint_smoke::{require_checkpoint_dir, verified_checkpoint_source};
use magnetar_loader_huggingface::{
    HuggingFaceIngestor, HuggingFaceTokenizer, parse_tokenizer_config,
};
use magnetar_runtime::model::ModelTrustStore;
use magnetar_runtime::production_model_ingestion::{
    ProductionModelArtifactIngestor, ProductionModelSource,
};
use magnetar_runtime::tokenizer::Tokenizer;
use magnetar_runtime::{
    E2eFixture, PromptInput, TokenId, TokenizationRequest,
    build_first_native_decode_graph_segment_for_config,
    build_first_native_prefill_graph_segment_for_config,
    load_production_qwen_instance_segment_for_provider, production_qwen_fixture,
    run_first_native_graph_segment_dispatch, run_production_qwen_generation_for_provider,
    tokenize_prompt_input,
};
use std::{fs, sync::Arc};

const REAL_VOCAB_SIZE: u64 = 151_936;
/// Bounded well below the existing 16-token CPU/CUDA smoke test: this
/// test's own real cost is roughly double per step (two real Providers
/// dispatched, one segment graph built and run on each, every step), so a
/// smaller real bound keeps a manual/nightly CI run's wall time
/// reasonable while still proving multiple real decode steps (not just
/// prefill).
const MAX_TOKENS: usize = 6;

fn argmax(logits: &[f32]) -> TokenId {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).expect("real logits are never NaN"))
        .map(|(index, _)| index as TokenId)
        .expect("real logits are never empty")
}

#[test]
#[ignore = "downloads/holds a ~1GB real checkpoint; manual/nightly two-real-GPU hardware profile only (add-real-multi-device-model-instance-placement)"]
fn real_public_checkpoint_multi_gpu_segment_decode_matches_full_graph_on_one_gpu() {
    let gpu0_probe = magnetar_provider_cuda::CudaProvider::new();
    if !gpu0_probe.is_available() {
        eprintln!("skipping: no compatible CUDA device found on this host at all");
        return;
    }
    let gpu1_probe =
        magnetar_provider_cuda::CudaProvider::for_device(1, "magnetar:provider/cuda:1");
    if !gpu1_probe.is_available() {
        eprintln!(
            "skipping: this host has only one real CUDA device -- a genuine second real GPU \
             is required to prove anything this test is specifically for"
        );
        return;
    }

    let dir = require_checkpoint_dir();
    let source: ProductionModelSource = verified_checkpoint_source(&dir);
    register_real_qwen_component();

    let ingested = HuggingFaceIngestor::new()
        .ingest(&source)
        .expect("real Qwen2.5-0.5B-Instruct bundle ingests");
    let num_hidden_layers = ingested
        .manifest
        .architecture_config
        .as_ref()
        .expect("real ingested manifest carries architecture_config")
        .num_hidden_layers;
    let mid = num_hidden_layers / 2;
    assert!(
        mid > 0 && mid < num_hidden_layers,
        "the real checkpoint must have at least 2 decoder layers to split into two segments"
    );

    let tokenizer_json = fs::read(dir.join("tokenizer.json")).expect("tokenizer.json readable");
    let tokenizer_config_bytes =
        fs::read(dir.join("tokenizer_config.json")).expect("tokenizer_config.json readable");
    let tokenizer_config =
        parse_tokenizer_config(&tokenizer_config_bytes).expect("real tokenizer_config.json parses");
    let real_tokenizer = HuggingFaceTokenizer::from_bytes(
        &tokenizer_json,
        Some(&tokenizer_config),
        "qwen2.5-0.5b-instruct",
        Some(REAL_VOCAB_SIZE),
    )
    .expect("real tokenizer.json loads and matches the declared vocab_size");
    let tokenizer_metadata = real_tokenizer.metadata().clone();
    let real_tokenizer: Arc<dyn Tokenizer + Send + Sync> = Arc::new(real_tokenizer);

    let trust_store =
        ModelTrustStore::default().trust_digest(ingested.manifest.id.digest.value.clone());

    // Ground truth: the real, full, unsegmented graph on real GPU 0 alone
    // -- the exact same entrypoint `tests_real_checkpoint_smoke.rs`
    // already proves against this checkpoint.
    let mut reference_manifest = ingested.manifest.clone();
    reference_manifest.generation = Some(magnetar_runtime::ModelGenerationDefaults {
        max_tokens: Some(MAX_TOKENS as u32),
        ..Default::default()
    });
    let reference_fixture = production_qwen_fixture(
        reference_manifest,
        tokenizer_metadata.clone(),
        Arc::clone(&real_tokenizer),
    )
    .expect("production fixture builds for the real single-GPU reference run");
    let reference_outcome = run_production_qwen_generation_for_provider(
        reference_fixture,
        ingested.payload_source.as_ref(),
        trust_store.clone(),
        "The capital of France is",
        Arc::new(magnetar_provider_cuda::CudaProvider::new()),
    )
    .expect("the real full-graph reference generation runs end to end on real GPU 0 alone");
    let reference_token_ids = reference_outcome.result.output.generated_token_ids.clone();
    assert_eq!(
        reference_token_ids.len(),
        MAX_TOKENS,
        "the real single-GPU reference must generate all requested tokens"
    );
    // Release the reference run's own real GPU 0 Runtime (and everything
    // it holds resident -- the full checkpoint's real weights) before the
    // segmented pipeline below opens its own, separate real GPU 0 Runtime
    // -- debugging real hardware contention between two concurrently live
    // CUDA contexts on the same physical Device, not something this
    // pipeline's own correctness should ever depend on.
    drop(reference_outcome);

    // The two-real-GPU segmented pipeline, against the same real ingested
    // checkpoint/tokenizer/prompt.
    let segment_fixture: E2eFixture = production_qwen_fixture(
        ingested.manifest.clone(),
        tokenizer_metadata,
        Arc::clone(&real_tokenizer),
    )
    .expect("production fixture builds for the segmented run");

    let prompt_tokens = tokenize_prompt_input(
        real_tokenizer.as_ref(),
        TokenizationRequest::new(PromptInput::PlainText("The capital of France is".into())),
        None,
    )
    .expect("the real prompt tokenizes through the real tokenizer");
    let prompt_len = prompt_tokens.token_ids.len() as u64;

    // Load each real segment Model Instance exactly once -- prefill and
    // every decode step below reuse the same real GPU 0 / real GPU 1
    // Runtime and Instance, never reloading weights per step.
    let gpu0_provider: Arc<dyn magnetar_runtime::provider::Provider> = Arc::new(gpu0_probe);
    let gpu1_provider: Arc<dyn magnetar_runtime::provider::Provider> = Arc::new(gpu1_probe);
    let gpu0_binding =
        magnetar_runtime::ProviderBinding::new(gpu0_provider.metadata().name.clone());
    let gpu1_binding =
        magnetar_runtime::ProviderBinding::new(gpu1_provider.metadata().name.clone());

    let mut gpu0_runtime = magnetar_runtime::Runtime::builder()
        .register_provider(gpu0_provider)
        .trust_store(trust_store.clone())
        .build()
        .expect("real GPU 0 Provider registers cleanly");
    magnetar_runtime::register_prepared_kernels_for_provider(
        &mut gpu0_runtime,
        &magnetar_provider_cuda::CudaProvider::new(),
    )
    .expect("real GPU 0 prepared kernels register cleanly");
    let mut gpu0_instance = load_production_qwen_instance_segment_for_provider(
        &mut gpu0_runtime,
        &segment_fixture.manifest,
        ingested.payload_source.as_ref(),
        &gpu0_binding,
        0,
        mid,
    )
    .expect("real GPU 0's own segment [0, mid) loads and materializes its real weights");

    let mut gpu1_runtime = magnetar_runtime::Runtime::builder()
        .register_provider(gpu1_provider)
        .trust_store(trust_store)
        .build()
        .expect("real GPU 1 Provider registers cleanly");
    magnetar_runtime::register_prepared_kernels_for_provider(
        &mut gpu1_runtime,
        &magnetar_provider_cuda::CudaProvider::for_device(1, "magnetar:provider/cuda:1"),
    )
    .expect("real GPU 1 prepared kernels register cleanly");
    let mut gpu1_instance = load_production_qwen_instance_segment_for_provider(
        &mut gpu1_runtime,
        &segment_fixture.manifest,
        ingested.payload_source.as_ref(),
        &gpu1_binding,
        mid,
        num_hidden_layers,
    )
    .expect(
        "real GPU 1's own segment [mid, num_hidden_layers) loads and materializes its real weights",
    );

    let gpu0_cache = magnetar_runtime::KvCacheId::new("multi-gpu-real-checkpoint-segment-0-cache")
        .expect("cache id is valid");
    let gpu1_cache = magnetar_runtime::KvCacheId::new("multi-gpu-real-checkpoint-segment-1-cache")
        .expect("cache id is valid");

    let mut generated_token_ids: Vec<TokenId> = Vec::with_capacity(MAX_TOKENS);
    let mut gpu0_layer_kv = magnetar_runtime::QwenLayerKvMap::new();
    let mut gpu1_layer_kv = magnetar_runtime::QwenLayerKvMap::new();

    for step in 0..MAX_TOKENS {
        let is_prefill = step == 0;
        let cached_token_count = prompt_len + step as u64;

        let (segment_one_graph, _definition, _instance) = if is_prefill {
            build_first_native_prefill_graph_segment_for_config(
                &segment_fixture.config,
                &segment_fixture.identity,
                prompt_len,
                0,
                mid,
            )
        } else {
            build_first_native_decode_graph_segment_for_config(
                &segment_fixture.config,
                &segment_fixture.identity,
                cached_token_count,
                0,
                mid,
            )
        }
        .expect("real GPU 0's own segment graph builds through the real Qwen Component");

        let step_token_ids: Vec<u32> = if is_prefill {
            prompt_tokens.token_ids.clone()
        } else {
            vec![
                *generated_token_ids
                    .last()
                    .expect("at least one token generated before decode"),
            ]
        };
        let gpu0_kv_history = if is_prefill {
            None
        } else {
            Some(&gpu0_layer_kv)
        };
        let gpu0_position = if is_prefill { 0 } else { cached_token_count };

        let gpu0_outcome = run_first_native_graph_segment_dispatch(
            gpu0_runtime,
            &segment_fixture,
            gpu0_instance,
            &gpu0_binding,
            &segment_one_graph,
            &gpu0_cache,
            &step_token_ids,
            0,
            None,
            gpu0_kv_history,
            Some(gpu0_position),
        )
        .expect("real GPU 0's own segment dispatch succeeds");
        gpu0_runtime = gpu0_outcome.runtime;
        gpu0_instance = gpu0_outcome.instance;
        gpu0_layer_kv = gpu0_outcome.layer_kv;
        let boundary_hidden = gpu0_outcome
            .bindings
            .get(&magnetar_runtime::TensorEdgeId::new("logits"))
            .expect("real GPU 0's own segment produced a logits-named (raw hidden-state) output")
            .clone();

        let (segment_two_graph, _definition, _instance) = if is_prefill {
            build_first_native_prefill_graph_segment_for_config(
                &segment_fixture.config,
                &segment_fixture.identity,
                prompt_len,
                mid,
                num_hidden_layers,
            )
        } else {
            build_first_native_decode_graph_segment_for_config(
                &segment_fixture.config,
                &segment_fixture.identity,
                cached_token_count,
                mid,
                num_hidden_layers,
            )
        }
        .expect("real GPU 1's own segment graph builds through the real Qwen Component");

        let gpu1_kv_history = if is_prefill {
            None
        } else {
            Some(&gpu1_layer_kv)
        };
        let gpu1_position = if is_prefill { 0 } else { cached_token_count };

        let gpu1_outcome = run_first_native_graph_segment_dispatch(
            gpu1_runtime,
            &segment_fixture,
            gpu1_instance,
            &gpu1_binding,
            &segment_two_graph,
            &gpu1_cache,
            &step_token_ids,
            mid,
            Some(boundary_hidden),
            gpu1_kv_history,
            Some(gpu1_position),
        )
        .expect("real GPU 1's own segment dispatch succeeds");
        gpu1_runtime = gpu1_outcome.runtime;
        gpu1_instance = gpu1_outcome.instance;
        gpu1_layer_kv = gpu1_outcome.layer_kv;
        let logits = gpu1_outcome
            .bindings
            .get(&magnetar_runtime::TensorEdgeId::new("logits"))
            .expect("real GPU 1's own segment produced the real logits output");

        // `logits.shape` is `[sequence_length, vocab_size]` -- prefill's
        // sequence_length is the real prompt length (every row a real
        // position's own distribution), so only the LAST row is this
        // step's actual next-token prediction. Sampling over the whole
        // flattened multi-row tensor (as if it were one row) is wrong for
        // any sequence_length > 1: the resulting flat index can exceed
        // vocab_size and is not a valid token id at all.
        let vocab_size = *logits
            .shape
            .last()
            .expect("logits tensor has a non-empty shape") as usize;
        let last_row_start = logits.data.len() - vocab_size;
        let next_token = argmax(&logits.data[last_row_start..]);
        generated_token_ids.push(next_token);
    }

    assert_eq!(
        generated_token_ids.len(),
        MAX_TOKENS,
        "the two-real-GPU segmented pipeline must generate all requested tokens"
    );
    assert_eq!(
        generated_token_ids, reference_token_ids,
        "the two-real-GPU segmented pipeline must select exactly the same real greedy tokens, \
         at every real step (prefill and every real decode step), as the real full graph \
         dispatched on one real GPU alone"
    );
    eprintln!(
        "real Qwen2.5-0.5B-Instruct multi-GPU segmented decode matched the real single-GPU \
         reference: {:?}",
        generated_token_ids
    );
}
