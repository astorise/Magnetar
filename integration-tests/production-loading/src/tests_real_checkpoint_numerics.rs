//! `qualify-real-checkpoint-performance` A02: checkpoint-bound token/logit
//! references, a declared per-dtype numerical profile, and streaming,
//! resume and cancellation lifecycle checks against Reference CPU and each
//! qualified Provider.
//!
//! The reference corpus for the pinned real checkpoint is checked in at
//! `fixtures/qualification/`. It is produced independently of Magnetar by
//! `tools/qualification/qwen2_reference_logits.py` (plain numpy, BF16
//! weights upcast to F32, Hugging Face Qwen2 semantics), so it can catch a
//! defect every Magnetar Provider shares -- comparing Providers only with
//! each other cannot. Every Provider run, Reference CPU included, is
//! compared against it, which also lets the GPU CI job check CUDA without
//! re-running Reference CPU. To regenerate it (only when the checkpoint or
//! prompt deliberately change):
//! `python3 -I tools/qualification/qwen2_reference_logits.py <checkpoint dir> \
//!    <revision> <weights sha256> 16 <corpus path> <prompt token ids...>`
//!
//! Lifecycle checks run for every selected Provider even when its
//! numerical comparison fails; each comparison's full differential
//! evidence (every step's differences) is written to
//! `MAGNETAR_QUALIFICATION_REPORT_DIR`, and the test fails at the end
//! naming every Provider that diverged.

use crate::test_support::{VOCAB_SIZE, register_real_qwen_component, write_tiny_production_bundle};
use crate::tests_real_checkpoint_smoke::{
    CHECKPOINT_REVISION, CHECKPOINT_WEIGHTS_SHA256, require_checkpoint_dir,
    verified_checkpoint_source,
};
use magnetar_inference_qualification::campaign::CheckpointIdentity;
use magnetar_inference_qualification::numerics::{
    LogitComparison, LogitRecorder, LogitReference, NumericalProfile, compare_logit_references,
};
use magnetar_loader_huggingface::{
    HuggingFaceIngestor, HuggingFaceTokenizer, parse_tokenizer_config,
};
use magnetar_runtime::model::ModelTrustStore;
use magnetar_runtime::production_model_ingestion::{
    ProductionModelArtifactIngestor, ProductionModelSource,
};
use magnetar_runtime::tokenizer::{EncodeInput, Tokenizer};
use magnetar_runtime::{
    FinishReason, GenerationParameters, GenerationStreamEvent, ModelArtifactSource,
    ProductionGenerationRequest, ProductionLoadedModel, PromptInput, Provider,
    ReferenceCpuProvider, StopConditions, production_model_fixture,
};
use std::{fs, ops::ControlFlow, path::PathBuf, sync::Arc};

const REAL_VOCAB_SIZE: u64 = 151_936;
const REAL_PROMPT: &str = "The capital of France is";
const REFERENCE_NEW_TOKENS: usize = 16;
/// Tokens generated before cancelling, and before resuming from a prompt
/// extended with them.
const LIFECYCLE_SPLIT: usize = 3;

fn reference_corpus_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("fixtures/qualification/qwen2.5-0.5b-instruct-7ae5576-independent-f32.json")
}

fn token_request(prompt_token_ids: &[u32], max_new_tokens: usize) -> ProductionGenerationRequest {
    ProductionGenerationRequest {
        prompt: PromptInput::TokenIds(prompt_token_ids.to_vec()),
        parameters: GenerationParameters::greedy(),
        stop_conditions: StopConditions::default(),
        max_new_tokens: Some(max_new_tokens),
        max_generation_millis: None,
    }
}

/// One greedy generation with every step's logits captured, plus the
/// stream events it delivered.
fn capture(
    model: &mut ProductionLoadedModel,
    checkpoint: &CheckpointIdentity,
    provider: &str,
    prompt_token_ids: &[u32],
    max_new_tokens: usize,
) -> (LogitReference, Vec<GenerationStreamEvent>) {
    let mut recorder = LogitRecorder::default();
    let mut events = Vec::new();
    let outcome = model
        .generate_streaming_with_logits(
            token_request(prompt_token_ids, max_new_tokens),
            None,
            &mut |event| {
                events.push(event);
                ControlFlow::Continue(())
            },
            &mut |logits| recorder.observe(logits),
        )
        .expect("generation with logit capture succeeds");
    let reference = recorder.finish(
        checkpoint.clone(),
        provider,
        "f32",
        prompt_token_ids.to_vec(),
        outcome.result.output.generated_token_ids.clone(),
    );
    (reference, events)
}

fn stream_token_ids(events: &[GenerationStreamEvent]) -> Vec<u32> {
    events
        .iter()
        .filter_map(|event| match event {
            GenerationStreamEvent::Token { token_id, .. } => Some(*token_id),
            GenerationStreamEvent::Finished { .. } => None,
        })
        .collect()
}

/// A02 lifecycle checks on one resident model: streaming order, logit
/// capture not perturbing generation, cancellation and resume.
fn check_lifecycle(
    model: &mut ProductionLoadedModel,
    checkpoint: &CheckpointIdentity,
    provider: &str,
    captured: &LogitReference,
    events: &[GenerationStreamEvent],
    profile: &NumericalProfile,
) {
    let generated = &captured.generated_token_ids;
    let prompt = &captured.prompt_token_ids;
    assert!(
        generated.len() > LIFECYCLE_SPLIT,
        "{provider}: too few tokens for lifecycle checks"
    );
    assert_eq!(
        captured.steps.len(),
        generated.len(),
        "{provider}: one logit step per token"
    );

    // Streaming: one Token per generated token in order, then exactly one
    // Finished, last.
    assert_eq!(
        &stream_token_ids(events),
        generated,
        "{provider}: streamed token order"
    );
    assert_eq!(
        events
            .iter()
            .filter(|event| matches!(event, GenerationStreamEvent::Finished { .. }))
            .count(),
        1,
        "{provider}: exactly one Finished event"
    );
    assert!(
        matches!(events.last(), Some(GenerationStreamEvent::Finished { .. })),
        "{provider}: Finished is last"
    );

    // Observing logits must not change generation.
    let mut plain_events = Vec::new();
    let plain = model
        .generate_streaming(token_request(prompt, generated.len()), None, &mut |event| {
            plain_events.push(event);
            ControlFlow::Continue(())
        })
        .expect("generation without logit capture succeeds");
    assert_eq!(
        &plain.result.output.generated_token_ids, generated,
        "{provider}: logit capture changed the generated tokens"
    );

    // Cancellation: stopping after LIFECYCLE_SPLIT tokens finishes cleanly
    // as Cancelled with exactly that prefix.
    let mut delivered = 0;
    let mut cancelled_events = Vec::new();
    let cancelled = model
        .generate_streaming(token_request(prompt, generated.len()), None, &mut |event| {
            if matches!(event, GenerationStreamEvent::Token { .. }) {
                delivered += 1;
            }
            cancelled_events.push(event);
            if delivered >= LIFECYCLE_SPLIT {
                ControlFlow::Break(())
            } else {
                ControlFlow::Continue(())
            }
        })
        .expect("cancelled generation finishes cleanly");
    assert_eq!(
        cancelled.result.output.finish_reason,
        FinishReason::Cancelled,
        "{provider}: cancellation finish reason"
    );
    assert_eq!(
        cancelled.result.output.generated_token_ids,
        generated[..LIFECYCLE_SPLIT],
        "{provider}: cancellation keeps exactly the delivered prefix"
    );
    assert_eq!(
        stream_token_ids(&cancelled_events),
        generated[..LIFECYCLE_SPLIT]
    );

    // Resume: a fresh request whose prompt is the original prompt plus the
    // committed prefix continues with the same tokens, and its logits
    // stay within the profile of the uninterrupted run's later steps.
    let mut resumed_prompt = prompt.clone();
    resumed_prompt.extend_from_slice(&generated[..LIFECYCLE_SPLIT]);
    let (resumed, _) = capture(
        model,
        checkpoint,
        provider,
        &resumed_prompt,
        generated.len() - LIFECYCLE_SPLIT,
    );
    assert_eq!(
        resumed.generated_token_ids,
        generated[LIFECYCLE_SPLIT..],
        "{provider}: resumed generation continues with the same tokens"
    );
    let mut expected_tail = captured.clone();
    expected_tail.prompt_token_ids = resumed_prompt;
    expected_tail.generated_token_ids = generated[LIFECYCLE_SPLIT..].to_vec();
    expected_tail.steps = captured.steps[LIFECYCLE_SPLIT..].to_vec();
    let resume_comparison = compare_logit_references(&expected_tail, &resumed, profile);
    assert!(
        resume_comparison.passed(),
        "{provider}: resumed logits drift from the uninterrupted run: {:?}",
        resume_comparison.failures
    );
}

fn record_comparison(comparison: &LogitComparison, label: &str) {
    if let Some(dir) = std::env::var_os("MAGNETAR_QUALIFICATION_REPORT_DIR").map(PathBuf::from) {
        fs::create_dir_all(&dir).expect("report directory is creatable");
        let path = dir.join(format!("logit-comparison-{label}.json"));
        fs::write(&path, serde_json::to_vec_pretty(comparison).unwrap())
            .expect("differential evidence is written");
        eprintln!("logit comparison evidence written to {}", path.display());
    }
}

/// Tasks 3.1-3.3 fast lane on the tiny synthetic bundle: two Reference CPU
/// captures are bit-identical, every lifecycle check passes, and a copy
/// whose logits drift beyond the F32 profile -- while sampling exactly the
/// same tokens, so the decoded text is identical -- fails and keeps its
/// differential evidence.
#[test]
fn fixture_logit_references_and_lifecycle_checks() {
    register_real_qwen_component();
    let dir = tempfile::tempdir().unwrap();
    write_tiny_production_bundle(dir.path());
    let source = ProductionModelSource::authorized_local_bundle(
        ModelArtifactSource::LocalPath(dir.path().to_path_buf()),
        dir.path().to_path_buf(),
    );
    let ingested = HuggingFaceIngestor::new().ingest(&source).unwrap();
    let tokenizer_bytes = fs::read(dir.path().join("tokenizer.json")).unwrap();
    let tokenizer = HuggingFaceTokenizer::from_bytes(
        &tokenizer_bytes,
        None,
        "numerics-fixture",
        Some(VOCAB_SIZE),
    )
    .unwrap();
    let prompt_token_ids = tokenizer
        .encode(EncodeInput::new("hi"))
        .expect("fixture prompt encodes")
        .token_ids;
    let tokenizer_metadata = tokenizer.metadata().clone();
    let tokenizer: Arc<dyn Tokenizer + Send + Sync> = Arc::new(tokenizer);
    let fixture =
        production_model_fixture(ingested.manifest.clone(), tokenizer_metadata, tokenizer).unwrap();
    let trust_store =
        ModelTrustStore::default().trust_digest(ingested.manifest.id.digest.value.clone());
    let checkpoint = CheckpointIdentity {
        name: "tiny-synthetic-production-bundle".into(),
        revision: None,
        weights_digest: Some(ingested.manifest.id.digest.value.clone()),
        digest_verified: false,
    };
    let mut model = ProductionLoadedModel::load(
        fixture,
        ingested.payload_source.as_ref(),
        trust_store,
        Arc::new(ReferenceCpuProvider::new()),
    )
    .unwrap();
    let profile = NumericalProfile::f32_cross_provider();

    let (first, events) = capture(
        &mut model,
        &checkpoint,
        "reference-cpu",
        &prompt_token_ids,
        6,
    );
    let (second, _) = capture(
        &mut model,
        &checkpoint,
        "reference-cpu",
        &prompt_token_ids,
        6,
    );
    let repeat = compare_logit_references(&first, &second, &profile);
    assert!(repeat.passed(), "{:?}", repeat.failures);
    assert!(
        repeat.steps.iter().all(|step| step.bit_identical),
        "Reference CPU is bit-reproducible on a resident instance"
    );
    assert_eq!(first.steps[0].vocab_size, VOCAB_SIZE as usize);

    check_lifecycle(
        &mut model,
        &checkpoint,
        "reference-cpu",
        &first,
        &events,
        &profile,
    );

    let mut drifted = second.clone();
    drifted.provider = "drifted".into();
    for entry in &mut drifted.steps[2].top {
        entry.logit += 2.0 * profile.max_abs_logit_difference;
    }
    drifted.steps[2].full_digest = "sha256:drifted".into();
    assert_eq!(drifted.generated_token_ids, first.generated_token_ids);
    let comparison = compare_logit_references(&first, &drifted, &profile);
    assert!(!comparison.passed(), "drift beyond tolerance must fail");
    assert!(
        comparison
            .failures
            .iter()
            .all(|failure| failure.starts_with("step 2:"))
    );
    assert_eq!(
        comparison.steps.len(),
        first.steps.len(),
        "evidence for every step is kept"
    );
    assert!(!comparison.steps[2].bit_identical);
    model.unload().unwrap();
}

/// Tasks 3.1/3.2 on the pinned real checkpoint: each selected Provider's
/// logits and tokens are compared with the checked-in independent corpus
/// under the F32 profile, and the lifecycle checks run on that Provider.
#[test]
#[ignore = "downloads/holds a ~1GB real checkpoint; manual/nightly qualification profile only"]
fn real_checkpoint_logits_match_the_reference_corpus() {
    let selection = std::env::var("MAGNETAR_QUALIFICATION_PROVIDERS")
        .unwrap_or_else(|_| "reference-cpu,cuda".into());
    let require_cuda = std::env::var("MAGNETAR_QUALIFICATION_REQUIRE_CUDA").as_deref() == Ok("1");
    let mut providers: Vec<(&str, Arc<dyn Provider>)> = Vec::new();
    for name in selection
        .split(',')
        .map(str::trim)
        .filter(|name| !name.is_empty())
    {
        match name {
            "reference-cpu" => {
                providers.push(("reference-cpu", Arc::new(ReferenceCpuProvider::new())))
            }
            "cuda" => {
                let cuda = magnetar_provider_cuda::CudaProvider::new();
                if cuda.is_available() {
                    providers.push(("cuda", Arc::new(cuda) as Arc<dyn Provider>));
                } else {
                    assert!(
                        !require_cuda,
                        "MAGNETAR_QUALIFICATION_REQUIRE_CUDA=1 but no CUDA Device"
                    );
                    eprintln!("CUDA numerical comparison skipped: no CUDA-capable Device");
                }
            }
            other => panic!("MAGNETAR_QUALIFICATION_PROVIDERS names unknown Provider {other:?}"),
        }
    }

    let dir = require_checkpoint_dir();
    let source = verified_checkpoint_source(&dir);
    register_real_qwen_component();
    let ingested = HuggingFaceIngestor::new()
        .ingest(&source)
        .expect("real Qwen2.5-0.5B-Instruct bundle ingests");
    let tokenizer_config =
        parse_tokenizer_config(&fs::read(dir.join("tokenizer_config.json")).unwrap()).unwrap();
    let tokenizer = HuggingFaceTokenizer::from_bytes(
        &fs::read(dir.join("tokenizer.json")).unwrap(),
        Some(&tokenizer_config),
        "qwen2.5-0.5b-instruct",
        Some(REAL_VOCAB_SIZE),
    )
    .unwrap();
    let prompt_token_ids = tokenizer
        .encode(EncodeInput::new(REAL_PROMPT))
        .expect("real prompt encodes")
        .token_ids;
    let tokenizer_metadata = tokenizer.metadata().clone();
    let tokenizer: Arc<dyn Tokenizer + Send + Sync> = Arc::new(tokenizer);
    let trust_store =
        ModelTrustStore::default().trust_digest(ingested.manifest.id.digest.value.clone());
    let checkpoint = CheckpointIdentity {
        name: "Qwen/Qwen2.5-0.5B-Instruct".into(),
        revision: Some(CHECKPOINT_REVISION.into()),
        weights_digest: Some(CHECKPOINT_WEIGHTS_SHA256.into()),
        digest_verified: true,
    };
    let profile = NumericalProfile::f32_cross_provider();
    let corpus: LogitReference = serde_json::from_slice(
        &fs::read(reference_corpus_path()).expect("the reference corpus is checked in"),
    )
    .expect("the reference corpus parses");
    assert_eq!(
        corpus.prompt_token_ids, prompt_token_ids,
        "corpus prompt is pinned"
    );
    assert_eq!(
        corpus.checkpoint.weights_digest.as_deref(),
        Some(CHECKPOINT_WEIGHTS_SHA256),
        "corpus is bound to the pinned checkpoint"
    );

    let mut diverged = Vec::new();
    for (provider_name, provider) in providers {
        let fixture = production_model_fixture(
            ingested.manifest.clone(),
            tokenizer_metadata.clone(),
            Arc::clone(&tokenizer),
        )
        .unwrap();
        let mut model = ProductionLoadedModel::load(
            fixture,
            ingested.payload_source.as_ref(),
            trust_store.clone(),
            provider,
        )
        .expect("the real checkpoint loads");
        let (captured, events) = capture(
            &mut model,
            &checkpoint,
            provider_name,
            &prompt_token_ids,
            REFERENCE_NEW_TOKENS,
        );

        let comparison = compare_logit_references(&corpus, &captured, &profile);
        record_comparison(&comparison, provider_name);
        eprintln!(
            "[{provider_name}] max |logit diff| vs independent corpus: {} over {} steps; \
             tokens {:?} (reference {:?})",
            comparison.max_abs_difference,
            comparison.steps.len(),
            captured.generated_token_ids,
            corpus.generated_token_ids,
        );
        if !comparison.passed() {
            diverged.push(format!("{provider_name}: {:?}", comparison.failures));
        }

        check_lifecycle(
            &mut model,
            &checkpoint,
            provider_name,
            &captured,
            &events,
            &profile,
        );
        model.unload().expect("the real checkpoint unloads");
    }
    assert!(
        diverged.is_empty(),
        "logits/tokens diverge from the independent reference corpus:\n{}",
        diverged.join("\n")
    );
}
