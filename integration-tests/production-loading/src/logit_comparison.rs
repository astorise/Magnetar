//! Compares a checked-in tiny checkpoint's generation with the
//! `transformers` reference logits beside it
//! (`magnetar.inference-qualification.logit-reference/v1`).
//!
//! The checkpoint is ingested through a real loader and loaded with the real
//! Model Component its architecture family selects. It then generates as
//! many greedy tokens as the reference holds on the Reference CPU (one
//! prefill and then decode steps with the KV cache). Every step's top-8
//! logits are compared with the reference: identical tokens, within 1e-3.

use magnetar_inference_qualification::numerics::{
    LogitRecorder, LogitReference, NumericalProfile, compare_logit_references,
};
use magnetar_loader_gguf::{GgufIngestor, GgufTokenizer};
use magnetar_loader_huggingface::{HuggingFaceIngestor, HuggingFaceTokenizer};
use magnetar_runtime::model::ModelTrustStore;
use magnetar_runtime::production_model_ingestion::{
    ProductionModelArtifactIngestor, ProductionModelSource,
};
use magnetar_runtime::tokenizer::Tokenizer;
use magnetar_runtime::{
    GenerationParameters, ModelArtifactSource, ProductionGenerationRequest, ProductionLoadedModel,
    PromptInput, ReferenceCpuProvider, StopConditions, production_model_fixture,
};
use std::{fs, ops::ControlFlow, path::PathBuf, sync::Arc};

/// The checkpoint formats of the fixtures.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Format {
    HuggingFace,
    Gguf,
}

/// Runs the checkpoint in `fixtures/{family_directory}/{variant}` and returns
/// its largest logit difference from the reference beside it, after checking
/// the family its loader declares.
pub(crate) fn compare_with_reference(
    family_directory: &str,
    variant: &str,
    format: Format,
    family: &str,
) -> f32 {
    let directory = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("fixtures")
        .join(family_directory)
        .join(variant);
    let source = ProductionModelSource::authorized_local_bundle(
        ModelArtifactSource::LocalPath(directory.clone()),
        directory.clone(),
    );
    let reference: LogitReference = serde_json::from_slice(
        &fs::read(directory.join("reference.json")).expect("the reference is readable"),
    )
    .expect("the reference parses");

    let (ingested, tokenizer): (_, Arc<dyn Tokenizer + Send + Sync>) = match format {
        Format::HuggingFace => {
            let ingested = HuggingFaceIngestor::new()
                .ingest(&source)
                .unwrap_or_else(|error| panic!("{variant} ingests: {error}"));
            let bytes = fs::read(directory.join("tokenizer.json")).expect("tokenizer readable");
            let tokenizer = HuggingFaceTokenizer::from_bytes(&bytes, None, "tiny-gemma4", Some(64))
                .expect("the tokenizer builds");
            (ingested, Arc::new(tokenizer))
        }
        Format::Gguf => {
            let ingested = GgufIngestor::new()
                .ingest(&source)
                .unwrap_or_else(|error| panic!("{variant} ingests: {error}"));
            let bytes = fs::read(directory.join("model.gguf")).expect("GGUF readable");
            let artifact = magnetar_format_gguf::parse(&bytes).expect("the GGUF parses");
            let tokenizer =
                GgufTokenizer::from_gguf_metadata(&artifact.metadata, "tiny-gemma4", Some(64))
                    .expect("the tokenizer builds");
            (ingested, Arc::new(tokenizer))
        }
    };
    assert_eq!(ingested.manifest.architecture.family, family, "{variant}");

    let component = crate::test_support::real_component_for_architecture_family(
        &ingested.manifest.architecture.family,
    );
    let fixture = production_model_fixture(
        ingested.manifest.clone(),
        tokenizer.metadata().clone(),
        tokenizer,
    )
    .expect("the production fixture builds");
    let mut model = ProductionLoadedModel::load_with_component(
        fixture,
        ingested.payload_source.as_ref(),
        ModelTrustStore::default().trust_digest(ingested.manifest.id.digest.value.clone()),
        Arc::new(ReferenceCpuProvider::new()),
        Some(component),
    )
    .unwrap_or_else(|error| panic!("{variant} loads with the Gemma Component: {error}"));

    let mut recorder = LogitRecorder::default();
    let outcome = model
        .generate_streaming_with_logits(
            ProductionGenerationRequest {
                prompt: PromptInput::TokenIds(reference.prompt_token_ids.clone()),
                parameters: GenerationParameters::greedy(),
                stop_conditions: StopConditions::default(),
                max_new_tokens: Some(reference.steps.len()),
                max_generation_millis: None,
            },
            None,
            &mut |_| ControlFlow::Continue(()),
            &mut |logits| recorder.observe(logits),
        )
        .unwrap_or_else(|error| panic!("{variant} generates: {error}"));
    let candidate = recorder.finish(
        reference.checkpoint.clone(),
        "magnetar-reference-cpu",
        "f32",
        reference.prompt_token_ids.clone(),
        outcome.result.output.generated_token_ids.clone(),
    );
    let comparison = compare_logit_references(
        &reference,
        &candidate,
        &NumericalProfile {
            name: "gemma4-f32-vs-transformers".into(),
            dtype: "f32".into(),
            max_abs_logit_difference: 1e-3,
            require_identical_tokens: true,
            compared_top_k: 8,
        },
    );
    assert!(
        comparison.passed(),
        "{variant}: tokens {:?} against {:?}: {:#?}",
        candidate.generated_token_ids,
        reference.generated_token_ids,
        comparison.failures
    );
    comparison.max_abs_difference
}
