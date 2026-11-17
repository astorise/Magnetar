//! TurboQuant KV gains on CUDA (`turboquant-kv-history`), run by
//! `.github/workflows/turboquant-kv-gains.yml`. One real GGUF checkpoint is
//! loaded once per KV precision (f32, then each TurboQuant one) and
//! generates greedily at several context lengths. For each precision and
//! context the test reports:
//!
//! - the KV bytes per token, and how many tokens of KV history the VRAM
//!   left free after loading holds;
//! - whether the context ran (a run that fails, out of memory for example,
//!   is reported and the precision's longer contexts are skipped);
//! - warm TTFT and TPOT;
//! - against f32 at the same context: how many leading generated tokens
//!   agree, whether the first token is the same, and the largest difference
//!   of the first step's logits.
//!
//! The prompt is a fixed English paragraph's tokens repeated to the context
//! length. The checkpoint is read as given: the workflow pins it by
//! revision and checks its sha256.

use magnetar_loader_gguf::GgufIngestor;
use magnetar_runtime::model::ModelTrustStore;
use magnetar_runtime::production_model_ingestion::{
    ProductionModelArtifactIngestor, ProductionModelSource,
};
use magnetar_runtime::tokenizer::{EncodeInput, Tokenizer};
use magnetar_runtime::{
    GenerationParameters, GenerationStreamEvent, KvPrecisions, KvStorePrecision,
    ModelArtifactSource, ProductionGenerationRequest, ProductionLoadedModel, PromptInput, Provider,
    StopConditions, production_model_fixture,
};
use std::{ops::ControlFlow, path::PathBuf, sync::Arc, time::Instant};

const PARAGRAPH: &str = "The history of the city stretches back more than two thousand \
    years. Its river carried traders, scholars and armies, and each left something \
    behind: a bridge, a library, a market square that still fills every morning. \
    Visitors walk the old streets, read the inscriptions on the walls, and ask how \
    so many generations managed to keep the place alive through floods, fires and wars.";

/// One precision at one context.
struct Measure {
    context: u64,
    /// Why the run failed, when it did.
    error: Option<String>,
    ttft_ms: f64,
    tpot_ms: f64,
    tokens: Vec<u32>,
    first_logits: Vec<f32>,
}

/// One precision's runs.
struct PrecisionRuns {
    precision: KvStorePrecision,
    kv_bytes_per_token: u64,
    free_bytes_after_load: u64,
    measures: Vec<Measure>,
}

fn env_or(name: &str, default: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| default.into())
}

#[test]
#[ignore = "needs a CUDA Device and a real checkpoint (turboquant-kv-gains workflow)"]
fn turboquant_kv_gains_on_cuda() {
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
        std::env::var_os("MAGNETAR_KV_GAINS_CHECKPOINT_DIR")
            .expect("MAGNETAR_KV_GAINS_CHECKPOINT_DIR names the directory holding model.gguf"),
    );
    let label = env_or("MAGNETAR_KV_GAINS_LABEL", "checkpoint");
    let contexts: Vec<u64> = env_or("MAGNETAR_KV_GAINS_CONTEXTS", "512,4096")
        .split(',')
        .map(|value| value.trim().parse().expect("a context length"))
        .collect();
    let new_tokens: usize = env_or("MAGNETAR_KV_GAINS_NEW_TOKENS", "64")
        .parse()
        .expect("a token count");
    let precisions: Vec<KvStorePrecision> = env_or("MAGNETAR_KV_GAINS_PRECISIONS", "f32,tq4,tq3")
        .split(',')
        .map(|value| KvStorePrecision::parse(value).expect("f32, tq4 or tq3"))
        .collect();
    assert_eq!(
        precisions.first(),
        Some(&KvStorePrecision::F32),
        "f32 runs first: every other precision is compared with it"
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
    // The paragraph with its beginning-of-sequence token, then the
    // paragraph alone, repeated to each context length.
    let encode = |special: bool| {
        let mut input = EncodeInput::new(PARAGRAPH);
        input.add_special_tokens = special;
        tokenizer
            .encode(input)
            .expect("the paragraph encodes")
            .token_ids
    };
    let (opening, body) = (encode(true), encode(false));
    let prompt = |context: u64| -> Vec<u32> {
        opening
            .iter()
            .chain(body.iter().cycle())
            .take(context as usize)
            .copied()
            .collect()
    };

    let mut runs: Vec<PrecisionRuns> = Vec::new();
    for &precision in &precisions {
        let mut manifest = ingested.manifest.clone();
        manifest.generation = Some(magnetar_runtime::model::ModelGenerationDefaults {
            max_tokens: Some(new_tokens as u32),
            ..Default::default()
        });
        let fixture = production_model_fixture(
            manifest.clone(),
            tokenizer_metadata.clone(),
            Arc::clone(&tokenizer),
        )
        .expect("the production fixture builds");
        let component = crate::test_support::real_component_for_architecture_family(
            &manifest.architecture.family,
        );
        let provider = Arc::new(magnetar_provider_cuda::CudaProvider::new());
        let mut model = ProductionLoadedModel::load_with_component(
            fixture,
            ingested.payload_source.as_ref(),
            ModelTrustStore::default().trust_digest(manifest.id.digest.value.clone()),
            provider.clone() as Arc<dyn Provider>,
            Some(component),
        )
        .expect("the checkpoint loads");
        model.request_kv_precisions(KvPrecisions::uniform(precision));
        let free_bytes_after_load = provider
            .context()
            .and_then(|context| context.mem_get_info().ok())
            .map_or(0, |(free, _)| free as u64);
        let mut measures = Vec::new();
        let mut failed = false;
        for &context in &contexts {
            if failed {
                measures.push(Measure {
                    context,
                    error: Some("skipped after a shorter context failed".into()),
                    ttft_ms: 0.0,
                    tpot_ms: 0.0,
                    tokens: Vec::new(),
                    first_logits: Vec::new(),
                });
                continue;
            }
            let request = |max_new_tokens| ProductionGenerationRequest {
                prompt: PromptInput::TokenIds(prompt(context)),
                parameters: GenerationParameters::greedy(),
                stop_conditions: StopConditions::default(),
                max_new_tokens: Some(max_new_tokens),
                max_generation_millis: None,
            };
            // A warmup at the same length, so the measured run is warm.
            let warm = model.generate(request(2), None);
            let mut token_times = Vec::new();
            let mut first_logits: Vec<f32> = Vec::new();
            let started = Instant::now();
            let outcome = warm.and_then(|_| {
                model.generate_streaming_with_logits(
                    request(new_tokens),
                    None,
                    &mut |event| {
                        if let GenerationStreamEvent::Token { .. } = event {
                            token_times.push(started.elapsed().as_secs_f64() * 1_000.0);
                        }
                        ControlFlow::Continue(())
                    },
                    &mut |logits| {
                        if first_logits.is_empty() {
                            first_logits = logits.to_vec();
                        }
                    },
                )
            });
            match outcome {
                Ok(outcome) => {
                    let ttft = token_times.first().copied().unwrap_or(0.0);
                    let tpot = match token_times.len() {
                        0 | 1 => 0.0,
                        n => (token_times[n - 1] - ttft) / (n - 1) as f64,
                    };
                    measures.push(Measure {
                        context,
                        error: None,
                        ttft_ms: ttft,
                        tpot_ms: tpot,
                        tokens: outcome.result.output.generated_token_ids,
                        first_logits,
                    });
                }
                Err(error) => {
                    failed = true;
                    measures.push(Measure {
                        context,
                        error: Some(error.to_string()),
                        ttft_ms: 0.0,
                        tpot_ms: 0.0,
                        tokens: Vec::new(),
                        first_logits: Vec::new(),
                    });
                }
            }
        }
        let resolution = model
            .kv_precision_resolution()
            .expect("a prefill resolved the KV precisions");
        assert_eq!(
            resolution.resolved,
            KvPrecisions::uniform(precision),
            "{label}: {} stores were asked for: {:?}",
            precision.as_str(),
            resolution.fallbacks
        );
        let kv_bytes_per_token = model.kv_bytes_per_token();
        let _ = model.unload();
        runs.push(PrecisionRuns {
            precision,
            kv_bytes_per_token,
            free_bytes_after_load,
            measures,
        });
    }

    let report = gains_report(&label, new_tokens, &runs);
    eprintln!("{report}");
    if let Ok(path) = std::env::var("MAGNETAR_KV_GAINS_REPORT") {
        std::fs::write(&path, &report).expect("the report is writable");
    }
    let baseline = &runs[0];
    assert!(
        baseline.measures.first().is_some_and(|m| m.error.is_none()),
        "{label}: f32 must run at the shortest context: {:?}",
        baseline.measures.first().and_then(|m| m.error.clone())
    );
}

/// The gains of every precision against f32, as a Markdown report.
fn gains_report(label: &str, new_tokens: usize, runs: &[PrecisionRuns]) -> String {
    let baseline = &runs[0];
    let mut report = format!(
        "### TurboQuant KV gains: {label}\n\n\
         Greedy decoding of {new_tokens} tokens after a prompt of each context length, one RTX 3060.\n\n\
         | KV precision | KV bytes/token | vs f32 | Tokens of KV in the free VRAM after load |\n\
         |---|---|---|---|\n"
    );
    for run in runs {
        let ratio = baseline.kv_bytes_per_token as f64 / run.kv_bytes_per_token.max(1) as f64;
        report.push_str(&format!(
            "| {} | {} | {ratio:.2}x smaller | {} |\n",
            run.precision.as_str(),
            run.kv_bytes_per_token,
            run.free_bytes_after_load / run.kv_bytes_per_token.max(1)
        ));
    }
    report.push_str(
        "\n| KV precision | Context | Ran | Warm TTFT (ms) | TPOT (ms) | TPOT vs f32 | \
         Leading tokens = f32 | First token = f32 | Max \\|Δ logit\\| step 1 |\n\
         |---|---|---|---|---|---|---|---|---|\n",
    );
    for run in runs {
        for measure in &run.measures {
            let reference = baseline.measures.iter().find(|candidate| {
                candidate.context == measure.context && candidate.error.is_none()
            });
            let Some(error) = &measure.error else {
                let (agree, first, logit, speed) = match reference {
                    Some(reference) => {
                        let agree = reference
                            .tokens
                            .iter()
                            .zip(&measure.tokens)
                            .take_while(|(a, b)| a == b)
                            .count();
                        let first = reference.tokens.first() == measure.tokens.first();
                        let logit = reference
                            .first_logits
                            .iter()
                            .zip(&measure.first_logits)
                            .map(|(a, b)| (a - b).abs())
                            .fold(0.0_f32, f32::max);
                        let speed = if measure.tpot_ms > 0.0 {
                            format!("{:.2}x", reference.tpot_ms / measure.tpot_ms)
                        } else {
                            "-".into()
                        };
                        (
                            format!("{agree}/{}", reference.tokens.len()),
                            if first { "yes" } else { "no" }.to_string(),
                            format!("{logit:.3e}"),
                            speed,
                        )
                    }
                    None => ("-".into(), "-".into(), "-".into(), "-".into()),
                };
                report.push_str(&format!(
                    "| {} | {} | yes | {:.1} | {:.2} | {speed} | {agree} | {first} | {logit} |\n",
                    run.precision.as_str(),
                    measure.context,
                    measure.ttft_ms,
                    measure.tpot_ms,
                ));
                continue;
            };
            let reason: String = error.chars().take(80).collect();
            report.push_str(&format!(
                "| {} | {} | no: {} | - | - | - | - | - | - |\n",
                run.precision.as_str(),
                measure.context,
                reason.replace('|', "/")
            ));
        }
    }
    report
}
