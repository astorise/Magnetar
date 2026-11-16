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
enum Format {
    HuggingFace,
    Gguf,
}

fn fixture(variant: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("fixtures/gemma4")
        .join(variant)
}

/// Runs `variant` and returns its largest logit difference from the
/// reference, after checking the family its loader declares.
fn compare(variant: &str, format: Format, family: &str) -> f32 {
    let directory = fixture(variant);
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
