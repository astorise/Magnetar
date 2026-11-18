//! Chunked prefill on CUDA against a prefill in one pass
//! (`chunk-long-prefills`). One real GGUF checkpoint is loaded once and
//! generates greedily from the same long prompt with every prompt row in
//! one pass, then in passes of each chunk length given. For each chunk
//! length the test reports TTFT, how many leading generated tokens agree
//! with the one-pass run, and the largest difference of the first step's
//! logits; it requires the same first token and first-step logits within
//! [`MAX_FIRST_LOGIT_DIFFERENCE`] of the one-pass run's.
//!
//! The prompt is a fixed English paragraph's tokens repeated to the prompt
//! length. The checkpoint is read as given: the workflow pins it by
//! revision.

use magnetar_loader_gguf::GgufIngestor;
use magnetar_runtime::model::ModelTrustStore;
use magnetar_runtime::production_model_ingestion::{
    ProductionModelArtifactIngestor, ProductionModelSource,
};
use magnetar_runtime::tokenizer::{EncodeInput, Tokenizer};
use magnetar_runtime::{
    GenerationParameters, GenerationStreamEvent, ModelArtifactSource, ProductionGenerationRequest,
    ProductionLoadedModel, PromptInput, Provider, StopConditions, production_model_fixture,
};
use std::{ops::ControlFlow, path::PathBuf, sync::Arc, time::Instant};

const PARAGRAPH: &str = "The history of the city stretches back more than two thousand \
    years. Its river carried traders, scholars and armies, and each left something \
    behind: a bridge, a library, a market square that still fills every morning. \
    Visitors walk the old streets, read the inscriptions on the walls, and ask how \
    so many generations managed to keep the place alive through floods, fires and wars.";

/// The largest first-step logit difference from the one-pass prefill a
/// chunked one may have. Passes of other row counts may take other Kernels
/// (and so round differently), never other positions or other keys: a
/// position or key off by one moves the logits by whole units.
const MAX_FIRST_LOGIT_DIFFERENCE: f32 = 0.5;

fn env_or(name: &str, default: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| default.into())
}

/// One generation's TTFT, tokens and first-step logits.
struct Run {
    ttft_ms: f64,
    tokens: Vec<u32>,
    first_logits: Vec<f32>,
}

#[test]
#[ignore = "needs a CUDA Device and a real checkpoint (gpu-runner-smoke kv group)"]
fn chunked_prefill_matches_one_pass_on_cuda() {
    let cuda = magnetar_provider_cuda::CudaProvider::new();
    if !cuda.is_available() {
        assert!(
            std::env::var("MAGNETAR_REQUIRE_CUDA").as_deref() != Ok("1"),
            "MAGNETAR_REQUIRE_CUDA=1 but no CUDA-capable Device is available"
        );
        eprintln!("skipped: no CUDA-capable Device");
        return;
    }
    let dir = PathBuf::from(
        std::env::var_os("MAGNETAR_CHUNKED_PREFILL_CHECKPOINT_DIR").expect(
            "MAGNETAR_CHUNKED_PREFILL_CHECKPOINT_DIR names the directory holding model.gguf",
        ),
    );
    let label = env_or("MAGNETAR_CHUNKED_PREFILL_LABEL", "checkpoint");
    let prompt_tokens: usize = env_or("MAGNETAR_CHUNKED_PREFILL_PROMPT_TOKENS", "3000")
        .parse()
        .expect("a token count");
    let chunks: Vec<u64> = env_or("MAGNETAR_CHUNKED_PREFILL_CHUNKS", "1024,512")
        .split(',')
        .map(|value| value.trim().parse().expect("a chunk length"))
        .collect();
    let new_tokens: usize = env_or("MAGNETAR_CHUNKED_PREFILL_NEW_TOKENS", "16")
        .parse()
        .expect("a token count");
    assert!(
        chunks
            .iter()
            .all(|chunk| *chunk > 0 && (*chunk as usize) < prompt_tokens),
        "every chunk must be shorter than the {prompt_tokens}-token prompt"
    );

    let source = ProductionModelSource::authorized_local_bundle(
        ModelArtifactSource::LocalPath(dir.clone()),
        dir.clone(),
    );
    let ingested = GgufIngestor::new()
        .ingest(&source)
        .expect("the checkpoint ingests");
    let vocab_size = ingested
        .manifest
        .architecture_config
        .as_ref()
        .expect("architecture config was normalized")
        .vocab_size;
    let tokenizer = magnetar_loader_gguf::load_gguf_tokenizer(
        &source,
        format!("{label}-tokenizer"),
        Some(vocab_size),
    )
    .expect("the tokenizer builds");
    let tokenizer_metadata = tokenizer.metadata().clone();
    let tokenizer: Arc<dyn Tokenizer + Send + Sync> = Arc::new(tokenizer);
    let encode = |special: bool| {
        let mut input = EncodeInput::new(PARAGRAPH);
        input.add_special_tokens = special;
        tokenizer
            .encode(input)
            .expect("the paragraph encodes")
            .token_ids
    };
    let (opening, body) = (encode(true), encode(false));
    let prompt: Vec<u32> = opening
        .iter()
        .chain(body.iter().cycle())
        .take(prompt_tokens)
        .copied()
        .collect();

    let mut manifest = ingested.manifest.clone();
    manifest.generation = Some(magnetar_runtime::model::ModelGenerationDefaults {
        max_tokens: Some(new_tokens as u32),
        ..Default::default()
    });
    let fixture =
        production_model_fixture(manifest.clone(), tokenizer_metadata, Arc::clone(&tokenizer))
            .expect("the production fixture builds");
    let component =
        crate::test_support::real_component_for_architecture_family(&manifest.architecture.family);
    let provider = Arc::new(magnetar_provider_cuda::CudaProvider::new());
    let mut model = ProductionLoadedModel::load_with_component(
        fixture,
        ingested.payload_source.as_ref(),
        ModelTrustStore::default().trust_digest(manifest.id.digest.value.clone()),
        provider as Arc<dyn Provider>,
        Some(component),
    )
    .expect("the checkpoint loads");

    let generate = |model: &mut ProductionLoadedModel, tokens: &[u32], max_new_tokens| {
        let mut first_logits: Vec<f32> = Vec::new();
        let mut ttft_ms = 0.0;
        let started = Instant::now();
        let outcome = model
            .generate_streaming_with_logits(
                ProductionGenerationRequest {
                    prompt: PromptInput::TokenIds(tokens.to_vec()),
                    parameters: GenerationParameters::greedy(),
                    stop_conditions: StopConditions::default(),
                    max_new_tokens: Some(max_new_tokens),
                    max_generation_millis: None,
                },
                None,
                &mut |event| {
                    if let GenerationStreamEvent::Token { .. } = event
                        && ttft_ms == 0.0
                    {
                        ttft_ms = started.elapsed().as_secs_f64() * 1_000.0;
                    }
                    ControlFlow::Continue(())
                },
                &mut |logits| {
                    if first_logits.is_empty() {
                        first_logits = logits.to_vec();
                    }
                },
            )
            .unwrap_or_else(|error| panic!("{label} generates: {error}"));
        Run {
            ttft_ms,
            tokens: outcome.result.output.generated_token_ids,
            first_logits,
        }
    };

    // Each setting's Kernels are compiled and its graphs built by a run at
    // the same length first, so the measured runs are warm.
    let measure = |model: &mut ProductionLoadedModel, chunk: u64| {
        model.set_prefill_chunk(chunk);
        assert_eq!(model.prefill_chunk(), chunk, "{label}: chunk {chunk}");
        generate(model, &prompt, 2);
        assert_eq!(
            model.prefill_chunk(),
            chunk,
            "{label}: the model declined chunk {chunk}"
        );
        generate(model, &prompt, new_tokens)
    };
    let one_pass = measure(&mut model, 0);
    eprintln!(
        "[chunked prefill] {label}: {prompt_tokens} prompt tokens in one pass: TTFT {:.1} ms, \
         generated token ids {:?}",
        one_pass.ttft_ms, one_pass.tokens
    );
    assert!(!one_pass.tokens.is_empty() && !one_pass.first_logits.is_empty());
    for chunk in chunks {
        let run = measure(&mut model, chunk);
        let agreeing = one_pass
            .tokens
            .iter()
            .zip(&run.tokens)
            .take_while(|(a, b)| a == b)
            .count();
        let difference = one_pass
            .first_logits
            .iter()
            .zip(&run.first_logits)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        eprintln!(
            "[chunked prefill] {label}: chunks of {chunk} rows ({} passes): TTFT {:.1} ms, \
             {agreeing}/{} leading tokens agree with one pass, max |first-step logit \
             difference| {difference:.3e}",
            (prompt_tokens as u64).div_ceil(chunk),
            run.ttft_ms,
            one_pass.tokens.len()
        );
        assert_eq!(run.first_logits.len(), one_pass.first_logits.len());
        assert_eq!(
            run.tokens.first(),
            one_pass.tokens.first(),
            "{label}: chunks of {chunk} change the first token"
        );
        assert!(
            difference.is_finite() && difference < MAX_FIRST_LOGIT_DIFFERENCE,
            "{label}: chunks of {chunk} move the first-step logits by {difference}"
        );
    }
}
