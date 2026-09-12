//! Real production Qwen loading end to end on real CUDA hardware (task
//! 12.3): the same real ingested bundle, real tokenizer, and real
//! compiled Qwen Component `tests_production_loading_e2e.rs` proves on
//! Reference CPU, instead dispatched through a real
//! `magnetar_provider_cuda::CudaProvider` via `run_production_qwen_
//! generation_for_provider` -- zero CUDA-specific code in
//! `magnetar-runtime` itself. Skips (returns early, does not fail) when
//! no CUDA-capable device is available, matching `integration-tests/
//! cuda-first-native`'s own pattern.

use crate::test_support::{VOCAB_SIZE, register_real_qwen_component, write_tiny_production_bundle};
use magnetar_loader_huggingface::HuggingFaceIngestor;
use magnetar_provider_cuda::CudaProvider;
use magnetar_runtime::model::ModelTrustStore;
use magnetar_runtime::production_model_ingestion::{
    ProductionModelArtifactIngestor, ProductionModelSource,
};
use magnetar_runtime::tokenizer::Tokenizer;
use magnetar_runtime::{ModelArtifactSource, production_qwen_fixture};
use std::sync::Arc;

#[test]
fn real_production_ingestion_generates_on_real_cuda_hardware() {
    let provider = CudaProvider::new();
    if !provider.is_available() {
        return;
    }
    register_real_qwen_component();

    let dir = tempfile::tempdir().unwrap();
    write_tiny_production_bundle(dir.path());

    let source = ProductionModelSource::authorized_local_bundle(
        ModelArtifactSource::Tachyon("tachyon-node-7:cuda-integration-test-bundle".into()),
        dir.path().to_path_buf(),
    );
    let mut ingested = HuggingFaceIngestor::new()
        .ingest(&source)
        .expect("real production ingestion succeeds");
    // Real multi-step decode (`implement-device-resident-multi-step-cuda-
    // decode`): historical KV concatenation across decode steps now
    // dispatches through a real, device-resident "concat" Kernel instead
    // of requiring host-readable tensor bytes, so this Provider genuinely
    // supports more than the one token prefill alone produces --
    // `max_tokens: 8` (matching the audit's own "8/16+ generated tokens"
    // bar) exercises 7 real decode steps beyond prefill, not just the
    // prefill-only shape task 12.3 originally proved. Each step's KV
    // pending-write and commit reuse the *same* stable resource identity
    // (replaced, not accumulated -- verified in isolation by
    // `copy_tensor_admitted_replaces_a_previous_allocation_at_the_same_
    // destination` on real hardware), so more steps here is real,
    // additional stress on that property, not merely a bigger number.
    ingested.manifest.generation = Some(magnetar_runtime::model::ModelGenerationDefaults {
        max_tokens: Some(8),
        ..Default::default()
    });

    let tokenizer_bytes = std::fs::read(dir.path().join("tokenizer.json")).unwrap();
    let real_tokenizer = magnetar_loader_huggingface::HuggingFaceTokenizer::from_bytes(
        &tokenizer_bytes,
        None,
        "cuda-integration-test-tokenizer",
        Some(VOCAB_SIZE),
    )
    .expect("real tokenizer.json loads");
    let tokenizer_metadata = real_tokenizer.metadata().clone();
    let real_tokenizer: Arc<dyn magnetar_runtime::tokenizer::Tokenizer + Send + Sync> =
        Arc::new(real_tokenizer);

    let trust_store =
        ModelTrustStore::default().trust_digest(ingested.manifest.id.digest.value.clone());
    let fixture = production_qwen_fixture(
        ingested.manifest.clone(),
        tokenizer_metadata,
        real_tokenizer,
    )
    .expect("production fixture builds from real ingested data");

    let outcome = magnetar_runtime::run_production_qwen_generation_for_provider(
        fixture,
        ingested.payload_source.as_ref(),
        trust_store,
        "hi",
        Arc::new(CudaProvider::new()),
    )
    .expect(
        "generation runs end to end on real CUDA hardware: real config.json -> real \
         tokenizer.json -> real Safetensors -> the real compiled Qwen Component -> real CUDA \
         Provider dispatch",
    );

    assert_eq!(
        outcome.result.output.generated_token_ids.len(),
        8,
        "real production ingestion + loading + generation on CUDA produced all 8 requested \
         tokens (prefill + 7 real decode steps), not just a prefill-only shape"
    );
    assert!(
        !outcome.text.starts_with("[generated token ids:"),
        "the real tokenizer decoded the generated tokens as real text: {}",
        outcome.text
    );
    // No silent Reference CPU fallback (task 12.3) is structural here, not
    // merely observed: this Runtime never registers a Reference CPU
    // Provider at all (only the CudaProvider passed to run_production_
    // qwen_generation_for_provider), so a successful dispatch could only
    // have gone through CUDA -- there was nothing else for it to fall
    // back to.

    // `close-tachyon-scope-audit-gaps` task 5.2: the shared generation
    // loop measures real wall-clock time identically for every Provider,
    // with zero CUDA-specific code -- proven here on real CUDA hardware
    // itself, not merely assumed by symmetry with the Reference CPU test.
    assert!(
        outcome.result.output.usage.tokens_per_second.is_some(),
        "real CUDA generation that produced at least one token must report real measured \
         tokens_per_second, not None"
    );
    assert!(
        outcome
            .result
            .output
            .usage
            .prefill_duration_millis
            .is_some(),
        "real CUDA generation must report real measured prefill duration"
    );
}

/// `expose-production-generation-parameters` verified on real CUDA
/// hardware, not only Reference CPU: non-greedy `GenerationParameters`
/// (temperature/top_p/seed) reach real CUDA dispatch without crashing or
/// hanging, and a caller-supplied `stop_conditions.stop_token_ids` entry
/// actually stops a real CUDA-dispatched generation early -- the same two
/// properties `tests_production_loading_e2e.rs`'s
/// `production_generation_request_forwards_non_greedy_sampling_parameters`/
/// `_honors_a_caller_supplied_stop_token_id` prove on Reference CPU.
/// Skips (does not fail) when no CUDA-capable device is available.
#[test]
fn production_generation_request_forwards_parameters_and_stop_conditions_on_real_cuda_hardware() {
    let provider = CudaProvider::new();
    if !provider.is_available() {
        return;
    }
    register_real_qwen_component();

    let dir = tempfile::tempdir().unwrap();
    write_tiny_production_bundle(dir.path());
    let source = ProductionModelSource::authorized_local_bundle(
        ModelArtifactSource::LocalPath(dir.path().to_path_buf()),
        dir.path().to_path_buf(),
    );
    let ingested = HuggingFaceIngestor::new()
        .ingest(&source)
        .expect("real production ingestion succeeds");

    let tokenizer_bytes = std::fs::read(dir.path().join("tokenizer.json")).unwrap();
    let load_tokenizer = || {
        let real_tokenizer = magnetar_loader_huggingface::HuggingFaceTokenizer::from_bytes(
            &tokenizer_bytes,
            None,
            "cuda-generation-parameters-test-tokenizer",
            Some(VOCAB_SIZE),
        )
        .expect("real tokenizer.json loads");
        let tokenizer_metadata = real_tokenizer.metadata().clone();
        let real_tokenizer: Arc<dyn magnetar_runtime::tokenizer::Tokenizer + Send + Sync> =
            Arc::new(real_tokenizer);
        (tokenizer_metadata, real_tokenizer)
    };

    let mut manifest = ingested.manifest.clone();
    manifest.generation = Some(magnetar_runtime::model::ModelGenerationDefaults {
        max_tokens: Some(4),
        ..Default::default()
    });
    let trust_store =
        ModelTrustStore::default().trust_digest(ingested.manifest.id.digest.value.clone());

    // Non-greedy sampling parameters must reach real CUDA dispatch without
    // crashing or hanging.
    let (tokenizer_metadata, real_tokenizer) = load_tokenizer();
    let non_greedy_fixture =
        production_qwen_fixture(manifest.clone(), tokenizer_metadata, real_tokenizer)
            .expect("production fixture builds from real ingested data");
    let non_greedy_outcome =
        magnetar_runtime::run_production_qwen_generation_for_provider_with_request(
            non_greedy_fixture,
            ingested.payload_source.as_ref(),
            trust_store.clone(),
            magnetar_runtime::ProductionGenerationRequest {
                prompt: magnetar_runtime::PromptInput::PlainText("hi".into()),
                parameters: magnetar_runtime::GenerationParameters {
                    temperature: 0.7,
                    top_p: Some(0.9),
                    seed: Some(42),
                    deterministic: true,
                    greedy: false,
                    sampling_enabled: true,
                    ..Default::default()
                },
                stop_conditions: magnetar_runtime::StopConditions::default(),
                max_new_tokens: None,
            },
            None,
            Arc::new(CudaProvider::new()),
        )
        .expect("non-greedy sampling parameters reach real CUDA dispatch");
    assert!(
        !non_greedy_outcome.result.output.generated_token_ids.is_empty(),
        "non-greedy production generation on real CUDA hardware produced at least one token"
    );

    // A caller-supplied stop_token_ids entry must actually stop a real
    // CUDA-dispatched generation, proving StopConditions reached it
    // instead of being replaced by StopConditions::default().
    let (tokenizer_metadata, real_tokenizer) = load_tokenizer();
    let baseline_fixture =
        production_qwen_fixture(manifest.clone(), tokenizer_metadata, real_tokenizer)
            .expect("production fixture builds from real ingested data");
    let baseline = magnetar_runtime::run_production_qwen_generation_for_provider_with_request(
        baseline_fixture,
        ingested.payload_source.as_ref(),
        trust_store.clone(),
        magnetar_runtime::ProductionGenerationRequest {
            prompt: magnetar_runtime::PromptInput::PlainText("hi".into()),
            parameters: magnetar_runtime::GenerationParameters::greedy(),
            stop_conditions: magnetar_runtime::StopConditions::default(),
            max_new_tokens: None,
        },
        None,
        Arc::new(CudaProvider::new()),
    )
    .expect("baseline production generation on real CUDA hardware succeeds");
    let first_token_id = *baseline
        .result
        .output
        .generated_token_ids
        .first()
        .expect("baseline CUDA generation produced at least one token");

    let (tokenizer_metadata, real_tokenizer) = load_tokenizer();
    let stopped_fixture = production_qwen_fixture(manifest, tokenizer_metadata, real_tokenizer)
        .expect("production fixture builds from real ingested data");
    let stopped = magnetar_runtime::run_production_qwen_generation_for_provider_with_request(
        stopped_fixture,
        ingested.payload_source.as_ref(),
        trust_store,
        magnetar_runtime::ProductionGenerationRequest {
            prompt: magnetar_runtime::PromptInput::PlainText("hi".into()),
            parameters: magnetar_runtime::GenerationParameters::greedy(),
            stop_conditions: magnetar_runtime::StopConditions {
                stop_token_ids: vec![first_token_id],
                ..Default::default()
            },
            max_new_tokens: None,
        },
        None,
        Arc::new(CudaProvider::new()),
    )
    .expect("stop-conditioned production generation on real CUDA hardware succeeds");
    assert_eq!(
        stopped.result.output.generated_token_ids,
        vec![first_token_id],
        "a caller-supplied stop_token_ids entry matching this deterministic greedy CUDA decode's \
         own first token must stop generation immediately after producing it"
    );
}

/// `stream-production-generation-events` verified on real CUDA hardware,
/// not only Reference CPU: streamed `Token` events (in production order)
/// followed by exactly one `Finished` event, and the concatenated text
/// deltas exactly reconstruct the non-streaming entry point's own decoded
/// text for the same request on the same real CUDA-dispatched weights.
/// Skips (does not fail) when no CUDA-capable device is available.
#[test]
fn production_generation_request_streaming_matches_non_streaming_on_real_cuda_hardware() {
    let provider = CudaProvider::new();
    if !provider.is_available() {
        return;
    }
    register_real_qwen_component();

    let dir = tempfile::tempdir().unwrap();
    write_tiny_production_bundle(dir.path());
    let source = ProductionModelSource::authorized_local_bundle(
        ModelArtifactSource::LocalPath(dir.path().to_path_buf()),
        dir.path().to_path_buf(),
    );
    let ingested = HuggingFaceIngestor::new()
        .ingest(&source)
        .expect("real production ingestion succeeds");

    let tokenizer_bytes = std::fs::read(dir.path().join("tokenizer.json")).unwrap();
    let load_tokenizer = || {
        let real_tokenizer = magnetar_loader_huggingface::HuggingFaceTokenizer::from_bytes(
            &tokenizer_bytes,
            None,
            "cuda-streaming-test-tokenizer",
            Some(VOCAB_SIZE),
        )
        .expect("real tokenizer.json loads");
        let tokenizer_metadata = real_tokenizer.metadata().clone();
        let real_tokenizer: Arc<dyn magnetar_runtime::tokenizer::Tokenizer + Send + Sync> =
            Arc::new(real_tokenizer);
        (tokenizer_metadata, real_tokenizer)
    };

    let mut manifest = ingested.manifest.clone();
    manifest.generation = Some(magnetar_runtime::model::ModelGenerationDefaults {
        max_tokens: Some(4),
        ..Default::default()
    });
    let trust_store =
        ModelTrustStore::default().trust_digest(ingested.manifest.id.digest.value.clone());
    let request = || magnetar_runtime::ProductionGenerationRequest {
        prompt: magnetar_runtime::PromptInput::PlainText("hi".into()),
        parameters: magnetar_runtime::GenerationParameters::greedy(),
        stop_conditions: magnetar_runtime::StopConditions::default(),
        max_new_tokens: None,
    };

    let (tokenizer_metadata, real_tokenizer) = load_tokenizer();
    let non_streaming_fixture =
        production_qwen_fixture(manifest.clone(), tokenizer_metadata, real_tokenizer)
            .expect("production fixture builds from real ingested data");
    let non_streaming = magnetar_runtime::run_production_qwen_generation_for_provider_with_request(
        non_streaming_fixture,
        ingested.payload_source.as_ref(),
        trust_store.clone(),
        request(),
        None,
        Arc::new(CudaProvider::new()),
    )
    .expect("non-streaming production generation on real CUDA hardware succeeds");

    let (tokenizer_metadata, real_tokenizer) = load_tokenizer();
    let streaming_fixture = production_qwen_fixture(manifest, tokenizer_metadata, real_tokenizer)
        .expect("production fixture builds from real ingested data");
    let mut events = Vec::new();
    let streamed = magnetar_runtime::run_production_qwen_generation_for_provider_streaming(
        streaming_fixture,
        ingested.payload_source.as_ref(),
        trust_store,
        request(),
        None,
        Arc::new(CudaProvider::new()),
        &mut |event| {
            events.push(event);
            std::ops::ControlFlow::Continue(())
        },
    )
    .expect("streaming production generation on real CUDA hardware succeeds");

    assert!(
        matches!(
            events.last(),
            Some(magnetar_runtime::GenerationStreamEvent::Finished { .. })
        ),
        "Finished must be the last delivered event on real CUDA hardware, got: {events:?}"
    );
    let delivered_token_ids: Vec<_> = events
        .iter()
        .filter_map(|event| match event {
            magnetar_runtime::GenerationStreamEvent::Token { token_id, .. } => Some(*token_id),
            magnetar_runtime::GenerationStreamEvent::Finished { .. } => None,
        })
        .collect();
    assert_eq!(
        delivered_token_ids, streamed.result.output.generated_token_ids,
        "delivered Token events must carry every generated token id, in production order, on \
         real CUDA hardware"
    );
    assert_eq!(
        streamed.result.output.generated_token_ids, non_streaming.result.output.generated_token_ids,
        "streaming and non-streaming must generate the same tokens for the same deterministic \
         greedy request on the same real CUDA-dispatched weights"
    );

    let reconstructed_text: String = events
        .iter()
        .filter_map(|event| match event {
            magnetar_runtime::GenerationStreamEvent::Token { text_delta, .. } => {
                text_delta.clone()
            }
            magnetar_runtime::GenerationStreamEvent::Finished { .. } => None,
        })
        .collect();
    assert_eq!(
        reconstructed_text, non_streaming.text,
        "concatenating every delivered text_delta in order must exactly reconstruct the \
         non-streaming entry point's own decoded text on real CUDA hardware"
    );
}

/// Task 12.7: a GPU CI job that dispatches this crate's tests must not be
/// able to report success while every CUDA-gated test above silently
/// skipped (returned early because no device was available) -- that
/// would make "0 real assertions ran" indistinguishable from "every
/// assertion passed". Mirrors `providers/cuda`'s own
/// `hardware_conformance_actually_ran_not_silently_skipped` exactly:
/// `#[ignore]`d by default so a GPU-less dev machine's plain `cargo test`
/// still passes cleanly, and run explicitly via `--include-ignored` only
/// by a job that guarantees a real GPU, where this test's failure means
/// the runner itself lost GPU access.
#[test]
#[ignore = "run explicitly via `cargo test -- --include-ignored` on a host guaranteed to have a GPU; every other test in this file already covers the GPU-less path"]
fn hardware_conformance_actually_ran_not_silently_skipped() {
    let provider = CudaProvider::new();
    assert!(
        provider.is_available(),
        "this test only runs where a compatible CUDA driver/device is guaranteed present -- \
         if it fails, the runner lost GPU access"
    );
}
