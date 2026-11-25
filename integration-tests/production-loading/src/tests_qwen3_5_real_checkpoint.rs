//! `support-qwen38-flash-next` task 2.7: the real Qwen3.5-0.8B-Base
//! checkpoint, pinned by revision and content digest, against a checked-in
//! `transformers` reference corpus.
//!
//! The corpus (`fixtures/qualification/qwen3.5-0.8b-base-dc7cdfe-transformers-f32.json`)
//! was produced independently of Magnetar by
//! `tools/qualification/qwen3_5_reference_logits.py`. That script runs
//! `transformers` 5.23.0's `Qwen3_5ForConditionalGeneration` in float32,
//! greedy, and recomputes the whole sequence at every step. Magnetar runs
//! one prefill and then decode steps that carry the gated-delta mixers'
//! recurrent states and the attention layers' KV history.
//!
//! To run:
//! 1. Download `config.json`, `tokenizer.json` and `tokenizer_config.json`
//!    from `Qwen/Qwen3.5-0.8B-Base` at the pinned revision, and its single
//!    `model.safetensors-00001-of-00001.safetensors` saved as
//!    `model.safetensors`, into one directory.
//! 2. `MAGNETAR_QWEN35_REAL_CHECKPOINT_DIR=<that directory>`, and optionally
//!    `MAGNETAR_QUALIFICATION_PROVIDERS` (comma-separated, default
//!    `reference-cpu,cpu-native`).
//! 3. `cargo test --release --manifest-path integration-tests/production-loading/Cargo.toml \
//!      qwen3_5_real_checkpoint -- --include-ignored --nocapture`

use crate::test_support::register_real_qwen_component;
use crate::tests_real_checkpoint_smoke::{RealCheckpointProfile, verified_profile_source};
use magnetar_inference_qualification::campaign::CheckpointIdentity;
use magnetar_inference_qualification::numerics::{
    LogitRecorder, LogitReference, NumericalProfile, compare_logit_references,
};
use magnetar_loader_huggingface::{
    HuggingFaceIngestor, HuggingFaceTokenizer, parse_tokenizer_config,
};
use magnetar_runtime::model::ModelTrustStore;
use magnetar_runtime::production_model_ingestion::ProductionModelArtifactIngestor;
use magnetar_runtime::tokenizer::{EncodeInput, Tokenizer};
use magnetar_runtime::{
    GenerationParameters, ProductionGenerationRequest, ProductionLoadedModel, PromptInput,
    Provider, ReferenceCpuProvider, StopConditions, production_model_fixture,
};
use std::{fs, ops::ControlFlow, path::PathBuf, sync::Arc};

/// <https://huggingface.co/Qwen/Qwen3.5-0.8B-Base/commit/dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68>:
/// 24 layers (three gated-delta linear-attention layers to each gated
/// full-attention layer), 1024 hidden, tied embeddings (Apache-2.0).
const QWEN3_5_0_8B_BASE: RealCheckpointProfile = RealCheckpointProfile {
    id: "qwen3.5-0.8b-base",
    repo: "Qwen/Qwen3.5-0.8B-Base",
    revision: "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68",
    weights_sha256: "sha256:c2b1e5a17d9c1e27685d92ed9b382911ebb99955ecd89052d1721241adfbab6c",
    vocab_size: 248_320,
};
const PROMPT: &str = "The capital of France is";
const NEW_TOKENS: usize = 16;

fn corpus_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("fixtures/qualification/qwen3.5-0.8b-base-dc7cdfe-transformers-f32.json")
}

fn providers() -> Vec<(&'static str, Arc<dyn Provider>)> {
    let selection = std::env::var("MAGNETAR_QUALIFICATION_PROVIDERS")
        .unwrap_or_else(|_| "reference-cpu,cpu-native".into());
    selection
        .split(',')
        .map(str::trim)
        .filter(|name| !name.is_empty())
        .map(|name| match name {
            "reference-cpu" => (
                "reference-cpu",
                Arc::new(ReferenceCpuProvider::new()) as Arc<dyn Provider>,
            ),
            "cpu-native" => {
                use magnetar_provider_cpu::native::{NativeCpuExecutor, NativeCpuProvider};
                (
                    "cpu-native",
                    Arc::new(NativeCpuProvider::with_executor(
                        NativeCpuExecutor::with_settings(true, false, 4),
                    )) as Arc<dyn Provider>,
                )
            }
            other => panic!("MAGNETAR_QUALIFICATION_PROVIDERS names unknown Provider {other:?}"),
        })
        .collect()
}

#[test]
#[ignore = "holds a 1.75 GB real checkpoint; manual or nightly qualification only"]
fn qwen3_5_real_checkpoint_logits_match_the_transformers_corpus() {
    let dir = PathBuf::from(
        std::env::var_os("MAGNETAR_QWEN35_REAL_CHECKPOINT_DIR").expect(
            "MAGNETAR_QWEN35_REAL_CHECKPOINT_DIR names the pinned Qwen3.5-0.8B-Base directory",
        ),
    );
    let source = verified_profile_source(&dir, &QWEN3_5_0_8B_BASE);
    register_real_qwen_component();
    let ingested = HuggingFaceIngestor::new()
        .ingest(&source)
        .expect("the real Qwen3.5-0.8B-Base bundle ingests");
    let config = ingested.manifest.architecture_config.clone().unwrap();
    assert!(config.linear_attention.is_some() && config.attention_output_gate);
    let tokenizer_config =
        parse_tokenizer_config(&fs::read(dir.join("tokenizer_config.json")).unwrap()).unwrap();
    let tokenizer = HuggingFaceTokenizer::from_bytes(
        &fs::read(dir.join("tokenizer.json")).unwrap(),
        Some(&tokenizer_config),
        QWEN3_5_0_8B_BASE.id,
        Some(QWEN3_5_0_8B_BASE.vocab_size),
    )
    .unwrap();
    let prompt_token_ids = tokenizer
        .encode(EncodeInput::new(PROMPT))
        .expect("the prompt encodes")
        .token_ids;
    let corpus: LogitReference =
        serde_json::from_slice(&fs::read(corpus_path()).expect("the corpus is checked in"))
            .expect("the corpus parses");
    assert_eq!(
        corpus.prompt_token_ids, prompt_token_ids,
        "the corpus prompt is pinned"
    );
    assert_eq!(
        corpus.checkpoint.weights_digest.as_deref(),
        Some(QWEN3_5_0_8B_BASE.weights_sha256),
        "the corpus is bound to the pinned checkpoint"
    );
    let tokenizer_metadata = tokenizer.metadata().clone();
    let tokenizer: Arc<dyn Tokenizer + Send + Sync> = Arc::new(tokenizer);
    let trust_store =
        ModelTrustStore::default().trust_digest(ingested.manifest.id.digest.value.clone());
    let checkpoint = CheckpointIdentity {
        name: QWEN3_5_0_8B_BASE.repo.into(),
        revision: Some(QWEN3_5_0_8B_BASE.revision.into()),
        weights_digest: Some(QWEN3_5_0_8B_BASE.weights_sha256.into()),
        digest_verified: true,
    };
    let profile = NumericalProfile::f32_cross_provider();

    let mut diverged = Vec::new();
    for (name, provider) in providers() {
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
        let started = std::time::Instant::now();
        let mut recorder = LogitRecorder::default();
        let outcome = model
            .generate_streaming_with_logits(
                ProductionGenerationRequest {
                    prompt: PromptInput::TokenIds(prompt_token_ids.clone()),
                    parameters: GenerationParameters::greedy(),
                    stop_conditions: StopConditions::default(),
                    max_new_tokens: Some(NEW_TOKENS),
                    max_generation_millis: None,
                },
                None,
                &mut |_| ControlFlow::Continue(()),
                &mut |logits| recorder.observe(logits),
            )
            .expect("generation with logit capture succeeds");
        let elapsed = started.elapsed();
        let captured = recorder.finish(
            checkpoint.clone(),
            name,
            "f32",
            prompt_token_ids.clone(),
            outcome.result.output.generated_token_ids.clone(),
        );
        let comparison = compare_logit_references(&corpus, &captured, &profile);
        eprintln!(
            "[{name}] max |logit diff| vs transformers: {:.3e} over {} steps in {elapsed:.1?}; \
             tokens {:?} (reference {:?})",
            comparison.max_abs_difference,
            comparison.steps.len(),
            captured.generated_token_ids,
            corpus.generated_token_ids,
        );
        if !comparison.passed() {
            diverged.push(format!("{name}: {:?}", comparison.failures));
        }
        model.unload().unwrap();
    }
    assert!(
        diverged.is_empty(),
        "diverged from transformers: {diverged:#?}"
    );
}
