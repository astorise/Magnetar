//! TurboQuant KV gains on CUDA (`turboquant-kv-history`), run by
//! `.github/workflows/turboquant-kv-gains.yml`. One real GGUF checkpoint is
//! loaded once per KV precision (f32, then each TurboQuant one, with or
//! without an exact head: `keep-turboquant-kv-head-exact`) and generates
//! greedily at several context lengths. For each precision and context the
//! test reports:
//!
//! - the KV bytes per token, and how many tokens of KV history the VRAM
//!   left free after loading holds;
//! - whether the context ran (a run that fails, out of memory for example,
//!   is reported and the precision's longer contexts are skipped);
//! - TTFT and TPOT, warm up to [`MAX_WARMED_CONTEXT`] tokens (a longer
//!   context's prefill dwarfs a first run's one-time costs, and warming it
//!   would double the run);
//! - against f32 at the same context: how many leading generated tokens
//!   agree, whether the first token is the same, and the largest difference
//!   of the first step's logits.
//!
//! The prompt is a fixed English paragraph's tokens repeated to the context
//! length. The checkpoint is read as given: the workflow pins it by
//! revision and checks its sha256. Each measure is printed as it ends and
//! the report is rewritten after each precision, so a run stopped by its
//! time limit keeps what it measured.

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

/// The longest context measured after a warmup at the same length.
const MAX_WARMED_CONTEXT: u64 = 4096;

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
    precision: KvPrecisions,
    kv_bytes_per_token: u64,
    /// `None` across several GPUs.
    free_bytes_after_load: Option<u64>,
    /// The rows of each prefill pass, `0` for one pass
    /// (`chunk-long-prefills`).
    prefill_chunk: u64,
    measures: Vec<Measure>,
}

fn env_or(name: &str, default: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| default.into())
}

/// A precision of `MAGNETAR_KV_GAINS_PRECISIONS`: `f32`, `tq4` or `tq3`
/// with the Runtime's default exact head, or `tq4@ROWS` / `tq3@ROWS` with
/// an exact head of `ROWS` rows (`@0` for none).
fn parse_precisions(value: &str) -> KvPrecisions {
    let (precision, exact_rows) = match value.trim().split_once('@') {
        Some((precision, rows)) => (precision, Some(rows.parse().expect("a row count"))),
        None => (value.trim(), None),
    };
    let precisions =
        KvPrecisions::uniform(KvStorePrecision::parse(precision).expect("f32, tq4 or tq3"));
    match exact_rows {
        Some(rows) => precisions.with_exact_rows(rows),
        None => precisions,
    }
}

/// How the report names `precisions`.
fn precision_label(precisions: &KvPrecisions) -> String {
    match (precisions.k, precisions.exact_rows) {
        (KvStorePrecision::F32, _) => "f32".into(),
        (precision, 0) => format!("{}, no exact head", precision.as_str()),
        (precision, rows) => format!("{}, exact head {rows}", precision.as_str()),
    }
}

#[test]
#[ignore = "needs a CUDA Device and a real checkpoint (turboquant-kv-gains workflow)"]
fn turboquant_kv_gains_on_cuda() {
    let devices: Option<Vec<usize>> = std::env::var("MAGNETAR_CUDA_DEVICES").ok().map(|list| {
        list.split(',')
            .map(|ordinal| {
                ordinal
                    .trim()
                    .parse()
                    .expect("MAGNETAR_CUDA_DEVICES lists GPU ordinals")
            })
            .collect()
    });
    let available = match &devices {
        Some(ordinals) => magnetar_provider_cuda::multi::CudaMultiProvider::for_devices(
            ordinals,
            "magnetar:provider/cuda",
        )
        .is_available(),
        None => magnetar_provider_cuda::CudaProvider::new().is_available(),
    };
    let gpus = match &devices {
        Some(ordinals) => format!(
            "{} GPUs, the layers placed across them ({})",
            ordinals.len(),
            match std::env::var("MAGNETAR_LAYER_PLACEMENT").as_deref() {
                Ok("balanced") => "balanced",
                _ => "filling the first",
            }
        ),
        None => "one GPU".to_string(),
    };
    if !available {
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
    let precisions: Vec<KvPrecisions> = env_or("MAGNETAR_KV_GAINS_PRECISIONS", "f32,tq4,tq3")
        .split(',')
        .map(parse_precisions)
        .collect();
    assert_eq!(
        precisions.first(),
        Some(&KvPrecisions::default()),
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
        // `MAGNETAR_CUDA_DEVICES=0,1`: one Provider spanning those GPUs, the
        // Runtime placing the layers across them, for a checkpoint larger
        // than one GPU (the free VRAM is then not reported).
        let single = devices
            .is_none()
            .then(|| Arc::new(magnetar_provider_cuda::CudaProvider::new()));
        let multi = devices.as_ref().map(|ordinals| {
            Arc::new(
                magnetar_provider_cuda::multi::CudaMultiProvider::for_devices(
                    ordinals,
                    "magnetar:provider/cuda",
                ),
            )
        });
        let provider: Arc<dyn Provider> = match (&single, &multi) {
            (Some(single), _) => single.clone(),
            (None, Some(multi)) => multi.clone(),
            (None, None) => unreachable!("a single Provider without devices"),
        };
        // Each GPU's free memory (diagnostic): after the load, and after a
        // context that failed.
        let free_memory = |when: &str| {
            if let Some(multi) = &multi {
                let free: Vec<String> = multi
                    .free_memory()
                    .iter()
                    .enumerate()
                    .map(|(gpu, free)| match free {
                        Some(bytes) => format!("GPU {gpu} {} MiB", bytes >> 20),
                        None => format!("GPU {gpu} unknown"),
                    })
                    .collect();
                eprintln!(
                    "[kv gains] {label} {}: free memory {when}: {}",
                    precision_label(&precision),
                    free.join(", ")
                );
            }
        };
        let mut model = ProductionLoadedModel::load_with_component(
            fixture,
            ingested.payload_source.as_ref(),
            ModelTrustStore::default().trust_digest(manifest.id.digest.value.clone()),
            provider,
            Some(component),
        )
        .expect("the checkpoint loads");
        model.request_kv_precisions(precision);
        // Diagnostics: decode steps dispatched node by node, and each
        // measured request's host time attributed to its phases.
        if std::env::var("MAGNETAR_KV_GAINS_STEP_REPLAY").as_deref() == Ok("0") {
            model.set_step_replay(false);
        }
        let phases = std::env::var("MAGNETAR_KV_GAINS_PHASES").as_deref() == Ok("1");
        free_memory("after the load");
        let free_bytes_after_load = single.as_ref().map(|single| {
            single
                .context()
                .and_then(|context| context.mem_get_info().ok())
                .map_or(0, |(free, _)| free as u64)
        });
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
            let context_started = Instant::now();
            let warm = if context <= MAX_WARMED_CONTEXT {
                model.generate(request(2), None).map(|_| ())
            } else {
                Ok(())
            };
            let mut token_times = Vec::new();
            let mut first_logits: Vec<f32> = Vec::new();
            if phases {
                model.enable_phase_attribution();
            }
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
            if phases {
                print_phases(
                    &format!("{label} {} context {context}", precision_label(&precision)),
                    &model.take_phase_attribution(),
                );
                model.disable_phase_attribution();
            }
            match &outcome {
                Ok(_) => eprintln!(
                    "[kv gains] {label} {} context {context}: ran, TTFT {:.1} ms, {} tokens, {:.0} s",
                    precision_label(&precision),
                    token_times.first().copied().unwrap_or(0.0),
                    token_times.len(),
                    context_started.elapsed().as_secs_f64()
                ),
                Err(error) => {
                    eprintln!(
                        "[kv gains] {label} {} context {context}: failed after {:.0} s: {error}",
                        precision_label(&precision),
                        context_started.elapsed().as_secs_f64()
                    );
                    free_memory(&format!("after context {context} failed"));
                }
            }
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
            precision,
            "{label}: {} stores were asked for: {:?}",
            precision_label(&precision),
            resolution.fallbacks
        );
        let kv_bytes_per_token = model.kv_bytes_per_token();
        let prefill_chunk = model.prefill_chunk();
        let _ = model.unload();
        runs.push(PrecisionRuns {
            precision,
            kv_bytes_per_token,
            free_bytes_after_load,
            prefill_chunk,
            measures,
        });
        // The report so far, so a run stopped by its time limit keeps it.
        if let Ok(path) = std::env::var("MAGNETAR_KV_GAINS_REPORT") {
            std::fs::write(&path, gains_report(&label, &gpus, new_tokens, &runs))
                .expect("the report is writable");
        }
    }

    let report = gains_report(&label, &gpus, new_tokens, &runs);
    eprintln!("{report}");
    let baseline = &runs[0];
    assert!(
        baseline.measures.first().is_some_and(|m| m.error.is_none()),
        "{label}: f32 must run at the shortest context: {:?}",
        baseline.measures.first().and_then(|m| m.error.clone())
    );
}

/// The gains of every precision against f32, as a Markdown report.
fn gains_report(label: &str, gpus: &str, new_tokens: usize, runs: &[PrecisionRuns]) -> String {
    let baseline = &runs[0];
    let prefill = match baseline.prefill_chunk {
        0 => "every prompt prefilled in one pass".to_string(),
        chunk => format!("longer prompts prefilled in passes of {chunk} rows"),
    };
    let mut report = format!(
        "### TurboQuant KV gains: {label}\n\n\
         Greedy decoding of {new_tokens} tokens after a prompt of each context length, {gpus}; \
         TTFT warm up to {MAX_WARMED_CONTEXT} tokens; {prefill}. An exact head keeps a store's \
         first rows at f32 (`keep-turboquant-kv-head-exact`): the bytes per token and the tokens \
         the free VRAM holds count every row at its TurboQuant size.\n\n\
         | KV precision | KV bytes/token | vs f32 | Tokens of KV in the free VRAM after load |\n\
         |---|---|---|---|\n"
    );
    for run in runs {
        let ratio = baseline.kv_bytes_per_token as f64 / run.kv_bytes_per_token.max(1) as f64;
        report.push_str(&format!(
            "| {} | {} | {ratio:.2}x smaller | {} |\n",
            precision_label(&run.precision),
            run.kv_bytes_per_token,
            run.free_bytes_after_load.map_or("-".to_string(), |free| {
                (free / run.kv_bytes_per_token.max(1)).to_string()
            })
        ));
    }
    report.push_str(
        "\n| KV precision | Context | Ran | TTFT (ms) | TPOT (ms) | TPOT vs f32 | \
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
                    precision_label(&run.precision),
                    measure.context,
                    measure.ttft_ms,
                    measure.tpot_ms,
                ));
                continue;
            };
            let reason: String = error.chars().take(80).collect();
            report.push_str(&format!(
                "| {} | {} | no: {} | - | - | - | - | - | - |\n",
                precision_label(&run.precision),
                measure.context,
                reason.replace('|', "/")
            ));
        }
    }
    report
}

/// Prints a measured request's heaviest phases per prefill and per decode
/// step (`MAGNETAR_KV_GAINS_PHASES=1`): host time, which covers the Device's
/// with `MAGNETAR_CUDA_SYNC_NODES=1`.
fn print_phases(label: &str, report: &magnetar_runtime::phase_attribution::PhaseAttributionReport) {
    use magnetar_runtime::phase_attribution::StepKind;
    for kind in [StepKind::Prefill, StepKind::Decode] {
        let Some(summary) = report.summary(kind) else {
            continue;
        };
        let steps = summary.steps.max(1) as f64;
        let mut phases: Vec<_> = summary.phases.iter().collect();
        phases.sort_by_key(|(_, total)| std::cmp::Reverse(total.host_micros));
        let top: Vec<String> = phases
            .iter()
            .take(12)
            .map(|(key, total)| {
                format!(
                    "{:?}{} {:.2} ms",
                    key.phase,
                    key.operator
                        .as_deref()
                        .map(|operator| format!("[{operator}]"))
                        .unwrap_or_default(),
                    total.host_micros as f64 / steps / 1_000.0
                )
            })
            .collect();
        eprintln!(
            "[kv gains] {label}: {kind:?}, {} steps of {:.2} ms ({:.2} unattributed): {}",
            summary.steps,
            summary.wall_micros as f64 / steps / 1_000.0,
            summary.unattributed_micros as f64 / steps / 1_000.0,
            top.join(", ")
        );
    }
}
