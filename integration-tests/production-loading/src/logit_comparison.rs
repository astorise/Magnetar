//! Compares a checked-in tiny checkpoint's generation with the
//! `transformers` reference logits beside it
//! (`magnetar.inference-qualification.logit-reference/v1`).
//!
//! The checkpoint is ingested through a real loader and loaded with the real
//! Model Component its architecture family selects. It then generates as
//! many greedy tokens as the reference holds, on the Reference CPU or a
//! given Provider (one prefill and then decode steps with the KV cache).
//! Every step's top-8 logits are compared with the reference: identical
//! tokens, within 1e-3.

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
    GenerationParameters, KvPrecisionResolution, KvPrecisions, ModelArtifactSource,
    ProductionGenerationRequest, ProductionLoadedModel, PromptInput, Provider,
    ReferenceCpuProvider, StopConditions, production_model_fixture,
};
use std::{fs, ops::ControlFlow, path::PathBuf, sync::Arc};

/// The checkpoint formats of the fixtures.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Format {
    HuggingFace,
    Gguf,
}

/// Runs the checkpoint in `fixtures/{family_directory}/{variant}` on the
/// Reference CPU and returns its largest logit difference from the
/// reference beside it, after checking the family its loader declares.
pub(crate) fn compare_with_reference(
    family_directory: &str,
    variant: &str,
    format: Format,
    family: &str,
) -> f32 {
    compare_on(
        Arc::new(ReferenceCpuProvider::new()),
        "magnetar-reference-cpu",
        family_directory,
        variant,
        format,
        family,
    )
}

/// The two CPU Providers of `providers/cpu`, labelled: its Reference CPU
/// Provider (packed weights expanded to f32 at load) and the optimized CPU
/// Provider computing from the packed weights with f32 activations (its
/// int8 activations round by more than the comparison allows).
pub(crate) fn cpu_providers() -> [(Arc<dyn Provider>, &'static str); 2] {
    use magnetar_provider_cpu::native::{NativeCpuExecutor, NativeCpuProvider};
    [
        (
            Arc::new(magnetar_provider_cpu::ReferenceCpuProvider::new()),
            "magnetar-provider-cpu-reference",
        ),
        (
            Arc::new(NativeCpuProvider::with_executor(
                NativeCpuExecutor::with_settings(true, false, 2),
            )),
            "magnetar-cpu-native",
        ),
    ]
}

/// [`compare_with_reference`] on `provider` (`provider_label` names it in
/// the candidate document).
pub(crate) fn compare_on(
    provider: Arc<dyn Provider>,
    provider_label: &str,
    family_directory: &str,
    variant: &str,
    format: Format,
    family: &str,
) -> f32 {
    compare_configured(
        provider,
        provider_label,
        family_directory,
        variant,
        format,
        family,
        (|_| {}, None),
    )
    .0
}

/// [`compare_on`] with the prompt prefilled in passes of `chunk` rows
/// (`chunk-long-prefills`). Returns the largest logit difference and the
/// chunk length the model kept: `0` when it runs every prompt in one pass.
pub(crate) fn compare_chunked_on(
    provider: Arc<dyn Provider>,
    provider_label: &str,
    family_directory: &str,
    variant: &str,
    format: Format,
    family: &str,
    chunk: u64,
) -> (f32, u64) {
    let (difference, model) = compare_configured(
        provider,
        provider_label,
        family_directory,
        variant,
        format,
        family,
        (|model| model.set_prefill_chunk(chunk), None),
    );
    (difference, model.prefill_chunk())
}

/// [`compare_on`] on a model `configure` adjusted before generating; returns
/// the model with the largest logit difference.
fn compare_configured(
    provider: Arc<dyn Provider>,
    provider_label: &str,
    family_directory: &str,
    variant: &str,
    format: Format,
    family: &str,
    options: (
        impl FnOnce(&mut ProductionLoadedModel),
        Option<magnetar_runtime::ComponentDigest>,
    ),
) -> (f32, ProductionLoadedModel) {
    let (configure, component) = options;
    let (mut model, reference) = load_checkpoint(
        provider,
        family_directory,
        variant,
        format,
        family,
        component,
    );
    configure(&mut model);
    let mut recorder = LogitRecorder::default();
    let outcome = model
        .generate_streaming_with_logits(
            reference_request(&reference),
            None,
            &mut |_| ControlFlow::Continue(()),
            &mut |logits| recorder.observe(logits),
        )
        .unwrap_or_else(|error| panic!("{variant} generates: {error}"));
    let candidate = recorder.finish(
        reference.checkpoint.clone(),
        provider_label,
        "f32",
        reference.prompt_token_ids.clone(),
        outcome.result.output.generated_token_ids.clone(),
    );
    let comparison = compare_logit_references(
        &reference,
        &candidate,
        &NumericalProfile {
            name: format!("{family}-f32-vs-transformers"),
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
    (comparison.max_abs_difference, model)
}

/// Compare on the Reference CPU using an explicitly registered Component.
pub(crate) fn compare_with_component(
    family_directory: &str,
    variant: &str,
    family: &str,
    component: magnetar_runtime::ComponentDigest,
) -> f32 {
    compare_configured(
        Arc::new(ReferenceCpuProvider::new()),
        "magnetar-reference-cpu",
        family_directory,
        variant,
        Format::HuggingFace,
        family,
        (|_| {}, Some(component)),
    )
    .0
}

/// The reference's prompt, greedy, for as many tokens as it holds.
fn reference_request(reference: &LogitReference) -> ProductionGenerationRequest {
    ProductionGenerationRequest {
        prompt: PromptInput::TokenIds(reference.prompt_token_ids.clone()),
        parameters: GenerationParameters::greedy(),
        stop_conditions: StopConditions::default(),
        max_new_tokens: Some(reference.steps.len()),
        max_generation_millis: None,
    }
}

/// How a TurboQuant KV history changes a checkpoint's generation against
/// f32 stores (`turboquant-kv-history`).
pub(crate) struct KvPrecisionDrift {
    pub(crate) resolution: KvPrecisionResolution,
    /// The largest difference of any logit at any step over the steps
    /// both runs share a history for (until their tokens first differ).
    pub(crate) max_abs_difference: f32,
    /// How many leading generated tokens agree.
    pub(crate) agreeing_tokens: usize,
    pub(crate) steps: usize,
}

/// Runs the checkpoint of [`compare_on`] on `provider` with f32 KV stores,
/// then on a fresh `provider` from `make_provider` with `precisions`
/// requested, and measures the drift.
pub(crate) fn kv_precision_drift(
    make_provider: &dyn Fn() -> Arc<dyn Provider>,
    family_directory: &str,
    variant: &str,
    format: Format,
    family: &str,
    precisions: KvPrecisions,
) -> KvPrecisionDrift {
    let run = |precisions| {
        generate_logits(
            make_provider(),
            family_directory,
            variant,
            format,
            family,
            precisions,
        )
    };
    let (f32_steps, f32_tokens, _) = run(KvPrecisions::default());
    let (steps, tokens, resolution) = run(precisions);
    let agreeing_tokens = f32_tokens
        .iter()
        .zip(&tokens)
        .take_while(|(a, b)| a == b)
        .count();
    // Step `i`'s logits pick token `i`: compare until the first differing
    // token, after which the histories differ.
    let compared = (agreeing_tokens + 1).min(steps.len()).min(f32_steps.len());
    let max_abs_difference = f32_steps[..compared]
        .iter()
        .zip(&steps[..compared])
        .flat_map(|(a, b)| a.iter().zip(b).map(|(x, y)| (x - y).abs()))
        .fold(0.0_f32, f32::max);
    KvPrecisionDrift {
        resolution,
        max_abs_difference,
        agreeing_tokens,
        steps: tokens.len(),
    }
}

/// Every step's logits, the generated tokens and the KV precision
/// resolution of the checkpoint of [`compare_on`] on `provider` with
/// `precisions` requested.
pub(crate) fn generate_logits(
    provider: Arc<dyn Provider>,
    family_directory: &str,
    variant: &str,
    format: Format,
    family: &str,
    precisions: KvPrecisions,
) -> (Vec<Vec<f32>>, Vec<u32>, KvPrecisionResolution) {
    let (mut model, reference) =
        load_checkpoint(provider, family_directory, variant, format, family, None);
    model.request_kv_precisions(precisions);
    let mut steps: Vec<Vec<f32>> = Vec::new();
    let outcome = model
        .generate_streaming_with_logits(
            reference_request(&reference),
            None,
            &mut |_| ControlFlow::Continue(()),
            &mut |logits| steps.push(logits.to_vec()),
        )
        .unwrap_or_else(|error| panic!("{variant} generates: {error}"));
    (
        steps,
        outcome.result.output.generated_token_ids,
        model
            .kv_precision_resolution()
            .expect("the prefill resolved the KV precisions"),
    )
}

/// Loads `fixtures/{family_directory}/{variant}` on `provider` with the
/// Model Component its family selects, after checking that family, and
/// returns it with its reference.
fn load_checkpoint(
    provider: Arc<dyn Provider>,
    family_directory: &str,
    variant: &str,
    format: Format,
    family: &str,
    component: Option<magnetar_runtime::ComponentDigest>,
) -> (ProductionLoadedModel, LogitReference) {
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
            let tokenizer =
                HuggingFaceTokenizer::from_bytes(&bytes, None, format!("tiny-{family}"), Some(64))
                    .expect("the tokenizer builds");
            (ingested, Arc::new(tokenizer))
        }
        Format::Gguf => {
            let ingested = GgufIngestor::new()
                .ingest(&source)
                .unwrap_or_else(|error| panic!("{variant} ingests: {error}"));
            let bytes = fs::read(directory.join("model.gguf")).expect("GGUF readable");
            let artifact = magnetar_format_gguf::parse(&bytes).expect("the GGUF parses");
            let tokenizer = GgufTokenizer::from_gguf_metadata(
                &artifact.metadata,
                format!("tiny-{family}"),
                Some(64),
            )
            .expect("the tokenizer builds");
            (ingested, Arc::new(tokenizer))
        }
    };
    assert_eq!(ingested.manifest.architecture.family, family, "{variant}");

    let component = component.unwrap_or_else(|| {
        crate::test_support::real_component_for_architecture_family(
            &ingested.manifest.architecture.family,
        )
    });
    let fixture = production_model_fixture(
        ingested.manifest.clone(),
        tokenizer.metadata().clone(),
        tokenizer,
    )
    .expect("the production fixture builds");
    let model = ProductionLoadedModel::load_with_component(
        fixture,
        ingested.payload_source.as_ref(),
        ModelTrustStore::default().trust_digest(ingested.manifest.id.digest.value.clone()),
        provider,
        Some(component),
    )
    .unwrap_or_else(|error| panic!("{variant} loads with its Component: {error}"));
    (model, reference)
}
