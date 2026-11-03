//! `qualify-real-checkpoint-performance` A01/A03/A04: the real-checkpoint
//! performance qualification harness. Drives the real production pipeline
//! (real ingestor, real compiled Qwen Component, real Provider dispatch)
//! through `ProductionLoadedModel` -- loaded once, then a cold request, warmup
//! requests and measured requests on the same resident instance -- and records
//! every token's arrival at the public streaming boundary into a
//! `magnetar-inference-qualification` campaign report with an immutable run
//! manifest, raw samples, statistics, resource usage and a hardware capability
//! matrix.
//!
//! Two lanes, kept separate on purpose:
//! - `fixture_only_campaign_*` runs per-PR on the tiny synthetic bundle. It
//!   proves the harness end to end and that such a run is labeled
//!   fixture-only and can never qualify production performance.
//! - `real_checkpoint_qualification_campaign` is `#[ignore]`d (a ~1GB
//!   checkpoint, minutes of Reference CPU compute) and meant for manual or
//!   nightly runs. Set up the checkpoint exactly as
//!   `tests_real_checkpoint_smoke.rs` describes, then:
//!   `MAGNETAR_QWEN_REAL_CHECKPOINT_DIR=<dir> \
//!    MAGNETAR_QUALIFICATION_REPORT_DIR=<out> \
//!    MAGNETAR_QUALIFICATION_CODE_REVISION=$(git rev-parse HEAD) \
//!    cargo test --release --manifest-path integration-tests/production-loading/Cargo.toml \
//!      real_checkpoint_qualification_campaign -- --include-ignored --nocapture`
//!   Optional: `MAGNETAR_QUALIFICATION_MAX_NEW_TOKENS` (default 16),
//!   `MAGNETAR_QUALIFICATION_MEASURED_RUNS` (default 5),
//!   `MAGNETAR_QUALIFICATION_PROVIDERS` (comma-separated subset of
//!   `reference-cpu,cuda`, default both), `MAGNETAR_QUALIFICATION_REQUIRE_CUDA=1`
//!   (fail instead of skipping when no CUDA Device exists -- set by the GPU
//!   CI job, which must never report a skipped CUDA campaign as a pass),
//!   `MAGNETAR_QUALIFICATION_DRIVER` (driver identity recorded in the
//!   manifest) and `MAGNETAR_QUALIFICATION_SUMMARY_PATH` (a Markdown summary
//!   is appended there, e.g. `$GITHUB_STEP_SUMMARY`).
//!
//! What the report does not claim: A02 logit/lifecycle evidence (produced
//! separately by `tests_real_checkpoint_numerics.rs`, not attached here), A07
//! budgeted memory campaigns, A08 downstream quality and device-memory
//! telemetry are recorded as pending gates / not measured, so the verdict
//! stays unqualified until those land.

use crate::test_support::{VOCAB_SIZE, register_real_qwen_component, write_tiny_production_bundle};
use crate::tests_real_checkpoint_smoke::{
    CHECKPOINT_REVISION, CHECKPOINT_WEIGHTS_SHA256, require_checkpoint_dir,
    verified_checkpoint_source,
};
use magnetar_inference_qualification::campaign::{
    Campaign, CampaignException, CampaignProtocol, CampaignReport, CellStatus, CheckpointIdentity,
    EvidenceClass, LoadSample, ProfileCell, ProviderIdentity, REPORT_SCHEMA_VERSION, RunManifest,
    RunPhase, standard_profile_grid,
};
use magnetar_inference_qualification::hardware::{
    CapabilityEvidence, HardwareCapability, HardwareCapabilityMatrix, HostIdentity, StageOutcome,
    measure_host_memory_bandwidth, probe_host_cpu_simd,
};
use magnetar_inference_qualification::resources::{IntervalProbe, ResourceUsage};
use magnetar_inference_qualification::trace::{GenerationTrace, TraceRecorder};
use magnetar_loader_huggingface::{
    HuggingFaceIngestor, HuggingFaceTokenizer, parse_tokenizer_config,
};
use magnetar_runtime::model::{ModelGenerationDefaults, ModelTrustStore};
use magnetar_runtime::phase_attribution::{PhaseAttributionReport, StepKind};
use magnetar_runtime::production_model_ingestion::{
    ProductionArtifactPayloadSource, ProductionModelArtifactIngestor, ProductionModelSource,
};
use magnetar_runtime::tokenizer::Tokenizer;
use magnetar_runtime::{
    E2eFixture, GenerationParameters, GenerationStreamEvent, ModelArtifactSource,
    ProductionGenerationRequest, ProductionLoadedModel, PromptInput, Provider,
    ReferenceCpuProvider, StopConditions, production_model_fixture,
};
use std::{collections::BTreeMap, fs, ops::ControlFlow, path::PathBuf, sync::Arc, time::Instant};

const REAL_VOCAB_SIZE: u64 = 151_936;
const REAL_PROMPT: &str = "The capital of France is";
const HARNESS: &str = "integration-tests/production-loading::tests_real_checkpoint_qualification";

/// One executed request before it is attached to a campaign.
struct RawRun {
    phase: RunPhase,
    trace: GenerationTrace,
    resources: ResourceUsage,
    token_ids: Vec<u32>,
    finish_reason: String,
}

struct CampaignTarget<'a> {
    evidence_class: EvidenceClass,
    checkpoint: CheckpointIdentity,
    fixture: E2eFixture,
    payload_source: &'a dyn ProductionArtifactPayloadSource,
    trust_store: ModelTrustStore,
    provider: Arc<dyn Provider>,
    prompt: &'static str,
    max_new_tokens: usize,
    protocol: CampaignProtocol,
}

fn sequential_protocol(warmup_runs: usize, measured_runs: usize) -> CampaignProtocol {
    CampaignProtocol {
        cold_run: true,
        warmup_runs,
        measured_runs,
        ordering: "sequential on one resident ProductionLoadedModel: load, cold, warmup, measured"
            .into(),
        interference_controls: vec![
            "single process, one request in flight".into(),
            "fresh inference session per request; model weights stay resident".into(),
            "peak RSS reset before each interval when /proc/self/clear_refs allows".into(),
        ],
    }
}

/// Loads `target` once and executes the protocol's cold, warmup and
/// measured requests, timing each token as it is delivered.
fn run_campaign(target: CampaignTarget<'_>, hardware: HardwareCapabilityMatrix) -> CampaignReport {
    let provider_metadata = target.provider.metadata();
    let load_probe = IntervalProbe::begin();
    let load_started = Instant::now();
    let mut model = ProductionLoadedModel::load(
        target.fixture,
        target.payload_source,
        target.trust_store,
        target.provider,
    )
    .expect("the campaign's model loads on its Provider");
    let load = LoadSample {
        duration_ms: load_started.elapsed().as_secs_f64() * 1_000.0,
        resources: load_probe.end(),
    };

    let mut phases = Vec::new();
    if target.protocol.cold_run {
        phases.push(RunPhase::Cold);
    }
    phases.extend(std::iter::repeat_n(
        RunPhase::Warmup,
        target.protocol.warmup_runs,
    ));
    phases.extend(std::iter::repeat_n(
        RunPhase::Measured,
        target.protocol.measured_runs,
    ));

    let provider_name = provider_metadata.name.clone();
    let mut phase_ordinals: BTreeMap<String, usize> = BTreeMap::new();
    let runs: Vec<RawRun> = phases
        .into_iter()
        .map(|phase| {
            let phase_label = format!("{phase:?}").to_lowercase();
            let ordinal = *phase_ordinals
                .entry(phase_label.clone())
                .and_modify(|count| *count += 1)
                .or_insert(0);
            let probe = IntervalProbe::begin();
            let mut recorder = TraceRecorder::start();
            let mut token_ids = Vec::new();
            let outcome = model
                .generate_streaming(
                    ProductionGenerationRequest {
                        prompt: PromptInput::PlainText(target.prompt.into()),
                        parameters: GenerationParameters::greedy(),
                        stop_conditions: StopConditions::default(),
                        max_new_tokens: Some(target.max_new_tokens),
                        max_generation_millis: None,
                    },
                    None,
                    &mut |event| {
                        if let GenerationStreamEvent::Token { token_id, .. } = event {
                            recorder.token();
                            token_ids.push(token_id);
                            if token_ids.len().is_multiple_of(PROGRESS_TOKEN_INTERVAL) {
                                eprintln!(
                                    "[{provider_name}] {phase_label} #{ordinal}: {} / {} tokens",
                                    token_ids.len(),
                                    target.max_new_tokens
                                );
                            }
                        }
                        ControlFlow::Continue(())
                    },
                )
                .expect("the campaign's request generates");
            let usage = &outcome.result.output.usage;
            let run = RawRun {
                phase,
                trace: recorder.finish(usage.prompt_tokens),
                resources: probe.end(),
                token_ids,
                finish_reason: format!("{:?}", outcome.result.output.finish_reason),
            };
            report_progress(&provider_name, &phase_label, ordinal, &run, usage);
            run
        })
        .collect();
    model
        .unload()
        .expect("the campaign's model unloads cleanly");

    let prompt_tokens = runs.first().map_or(0, |run| run.trace.prompt_tokens);
    let mut flags = BTreeMap::new();
    flags.insert("prompt".into(), target.prompt.into());
    let mut campaign = Campaign::new(RunManifest {
        schema_version: REPORT_SCHEMA_VERSION.into(),
        harness: HARNESS.into(),
        evidence_class: target.evidence_class,
        checkpoint: target.checkpoint,
        code_revision: std::env::var("MAGNETAR_QUALIFICATION_CODE_REVISION").ok(),
        provider: ProviderIdentity {
            name: provider_metadata.name,
            version: Some(provider_metadata.version),
            driver: std::env::var("MAGNETAR_QUALIFICATION_DRIVER").ok(),
        },
        dtype: "f32".into(),
        prompt_tokens,
        max_new_tokens: target.max_new_tokens,
        batch_size: 1,
        concurrency: 1,
        sampling: "greedy".into(),
        build_profile: if cfg!(debug_assertions) {
            "debug"
        } else {
            "release"
        }
        .into(),
        host_os: hardware.host.os.clone(),
        host_arch: hardware.host.arch.clone(),
        flags,
        protocol: target.protocol,
    });
    campaign.record_load(load);
    for run in runs {
        campaign.record(
            run.phase,
            run.trace,
            run.resources,
            run.token_ids,
            run.finish_reason,
        );
    }

    let profile_matrix = profile_matrix(prompt_tokens, target.max_new_tokens);
    campaign.finish(hardware, profile_matrix, pending_gates())
}

/// The actually-run cell plus every standard cell, each unrun one with
/// the reason it was not run.
fn profile_matrix(prompt_tokens: usize, max_new_tokens: usize) -> Vec<ProfileCell> {
    let mut cells = vec![ProfileCell {
        concurrency: 1,
        context_tokens: prompt_tokens,
        generated_tokens: max_new_tokens,
        status: CellStatus::Run,
    }];
    cells.extend(standard_profile_grid().into_iter().map(
        |(concurrency, context_tokens, generated_tokens)| ProfileCell {
            concurrency,
            context_tokens,
            generated_tokens,
            status: CellStatus::Skipped {
                reason: if concurrency > 1 {
                    "ProductionLoadedModel serves one request at a time; concurrent \
                     serving is connect-production-cli-and-service (A05) scope"
                        .into()
                } else {
                    "not part of this bounded single-request campaign profile".into()
                },
            },
        },
    ));
    cells
}

fn pending_gates() -> Vec<(String, String)> {
    [
        (
            "A02",
            "logit-reference comparison is a separate artifact \
             (tests_real_checkpoint_numerics) not attached to this report",
        ),
        (
            "A07",
            "budgeted memory-ladder / fixed-budget split campaigns not implemented yet",
        ),
        (
            "A08",
            "downstream quality datasets and thresholds not selected yet",
        ),
    ]
    .into_iter()
    .map(|(gate, reason)| (gate.to_string(), reason.to_string()))
    .collect()
}

/// A03: host discovery plus the accelerator targets this harness can see.
/// `generated_on` names the Provider/Device whose real generation ran;
/// only that entry's compute stage may pass.
fn hardware_matrix(generated_on: Option<(&str, bool)>) -> HardwareCapabilityMatrix {
    let host = HostIdentity::discover();
    let mut entries = probe_host_cpu_simd(&host);
    entries.push(measure_host_memory_bandwidth(&host, 64 << 20, 5));

    let cuda = magnetar_provider_cuda::CudaProvider::new();
    if cuda.is_available() {
        let device = cuda
            .devices()
            .first()
            .map(|device| device.metadata().name.clone())
            .unwrap_or_else(|| "cuda".into());
        for capability in [
            HardwareCapability::CudaComputeCapability,
            HardwareCapability::CudaGraphCapture,
            HardwareCapability::Fp16Compute,
            HardwareCapability::Bf16Compute,
        ] {
            entries.push(CapabilityEvidence {
                capability,
                device: Some(device.clone()),
                discovery: StageOutcome::passed("CUDA Provider reports an available Device"),
                compute: StageOutcome::skipped(
                    "no capability-specific compute probe in this harness yet",
                ),
                production_validation: StageOutcome::skipped("not qualified"),
            });
        }
    } else {
        for capability in [
            HardwareCapability::CudaComputeCapability,
            HardwareCapability::CudaGraphCapture,
            HardwareCapability::Fp16Compute,
            HardwareCapability::Bf16Compute,
        ] {
            entries.push(CapabilityEvidence::absent(
                capability,
                "no CUDA-capable Device available to the CUDA Provider",
            ));
        }
    }
    entries.push(CapabilityEvidence::absent(
        HardwareCapability::Hip,
        "not probed: the ROCm Provider has no qualified compute path on this harness",
    ));
    entries.push(CapabilityEvidence::absent(
        HardwareCapability::Wgpu,
        "not probed by this harness profile",
    ));

    if let Some((device, completed)) = generated_on {
        entries.push(CapabilityEvidence {
            capability: HardwareCapability::F32Generation,
            device: Some(device.into()),
            discovery: StageOutcome::passed("Provider registered and Device available"),
            compute: if completed {
                StageOutcome::passed("multi-token generation completed on this Device")
            } else {
                StageOutcome::failed("generation did not complete")
            },
            production_validation: StageOutcome::skipped(
                "A02/A07/A08 qualification gates are pending",
            ),
        });
    }
    HardwareCapabilityMatrix::new(host, entries)
}

/// A Provider name (e.g. `magnetar:provider/cuda`) as a single file-name
/// component: anything but ASCII alphanumerics, `-` and `_` becomes `-`.
fn file_label(name: &str) -> String {
    name.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '-'
            }
        })
        .collect()
}

/// Tokens between two in-request progress lines.
const PROGRESS_TOKEN_INTERVAL: usize = 16;

/// One line per finished request on stderr, and the same record appended
/// to `<report dir>/progress-<provider label>.jsonl` (see [`file_label`]), so a campaign cut off by a
/// timeout still leaves its completed requests behind. `step_ms` is the
/// time the Runtime measured inside its own prefill/decode steps; the rest
/// of the end-to-end latency is per-request work outside them.
fn report_progress(
    provider: &str,
    phase: &str,
    ordinal: usize,
    run: &RawRun,
    usage: &magnetar_runtime::GenerationUsage,
) {
    let metrics = run.trace.metrics();
    let step_ms = usage
        .prefill_duration_millis
        .unwrap_or(0)
        .saturating_add(usage.decode_duration_millis.unwrap_or(0));
    let record = serde_json::json!({
        "provider": provider,
        "phase": phase,
        "ordinal": ordinal,
        "prompt_tokens": metrics.prompt_tokens,
        "generated_tokens": metrics.generated_tokens,
        "time_to_first_token_ms": metrics.time_to_first_token_ms,
        "time_per_output_token_ms": metrics.time_per_output_token_ms,
        "inter_token_latency_slope_ms_per_token": metrics.inter_token_latency_slope_ms_per_token,
        "end_to_end_latency_ms": metrics.end_to_end_latency_ms,
        "runtime_step_ms": step_ms,
        "outside_step_ms": metrics.end_to_end_latency_ms - step_ms as f64,
        "finish_reason": run.finish_reason,
    });
    eprintln!(
        "[{provider}] {phase} #{ordinal}: {} tokens, TTFT {:.1} ms, TPOT {:.1} ms, \
         ITL growth {:.2} ms/token, end-to-end {:.0} ms (runtime steps {step_ms} ms)",
        metrics.generated_tokens,
        metrics.time_to_first_token_ms.unwrap_or(f64::NAN),
        metrics.time_per_output_token_ms.unwrap_or(f64::NAN),
        metrics
            .inter_token_latency_slope_ms_per_token
            .unwrap_or(f64::NAN),
        metrics.end_to_end_latency_ms,
    );
    if let Some(dir) = std::env::var_os("MAGNETAR_QUALIFICATION_REPORT_DIR").map(PathBuf::from) {
        fs::create_dir_all(&dir).expect("report directory is creatable");
        let mut file = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(dir.join(format!("progress-{}.jsonl", file_label(provider))))
            .expect("the progress file opens for appending");
        std::io::Write::write_all(&mut file, format!("{record}\n").as_bytes())
            .expect("the progress record is written");
    }
}

fn write_report_if_requested(report: &CampaignReport) {
    if let Some(dir) = std::env::var_os("MAGNETAR_QUALIFICATION_REPORT_DIR").map(PathBuf::from) {
        let path = report
            .write_immutable(&dir)
            .expect("the campaign report is written to a fresh, immutable file");
        eprintln!("qualification report written to {}", path.display());
    }
    if let Some(path) = std::env::var_os("MAGNETAR_QUALIFICATION_SUMMARY_PATH") {
        let mut file = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .expect("the qualification summary file opens for appending");
        std::io::Write::write_all(
            &mut file,
            format!("{}\n", report.markdown_summary()).as_bytes(),
        )
        .expect("the qualification summary is written");
    }
}

/// Which Providers the real campaign runs, from
/// `MAGNETAR_QUALIFICATION_PROVIDERS` (default: all known).
fn selected_providers() -> Vec<String> {
    let selection = std::env::var("MAGNETAR_QUALIFICATION_PROVIDERS")
        .unwrap_or_else(|_| "reference-cpu,cuda".into());
    let providers: Vec<String> = selection
        .split(',')
        .map(str::trim)
        .filter(|name| !name.is_empty())
        .map(str::to_string)
        .collect();
    for name in &providers {
        assert!(
            matches!(name.as_str(), "reference-cpu" | "cuda"),
            "MAGNETAR_QUALIFICATION_PROVIDERS names unknown Provider {name:?}"
        );
    }
    assert!(
        !providers.is_empty(),
        "MAGNETAR_QUALIFICATION_PROVIDERS selects no Provider"
    );
    providers
}

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .map(|value| {
            value
                .parse()
                .unwrap_or_else(|_| panic!("{key} must be an integer"))
        })
        .unwrap_or(default)
}

fn assert_complete_trace(report: &CampaignReport, expected_tokens: usize) {
    for sample in &report.samples {
        assert_eq!(
            sample.metrics.generated_tokens, expected_tokens,
            "every run delivers every requested token: {sample:?}"
        );
        assert_eq!(sample.generated_token_ids.len(), expected_tokens);
        assert!(sample.metrics.time_to_first_token_ms.is_some());
        assert!(sample.metrics.time_per_output_token_ms.is_some());
        assert!(sample.metrics.prompt_tokens > 0);
    }
    assert!(
        report
            .load
            .as_ref()
            .is_some_and(|load| load.duration_ms > 0.0)
    );
    assert!(report.manifest_digest_matches());
}

/// Task 2.3 (and the fast lane of 2.2/5.2): the full harness runs on the
/// tiny synthetic bundle, produces every timing, sample and statistic --
/// and is labeled fixture-only, so it cannot qualify production
/// performance no matter how complete its numbers are.
#[test]
fn fixture_only_campaign_is_measured_but_cannot_qualify_production_performance() {
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
        "qualification-fixture-tokenizer",
        Some(VOCAB_SIZE),
    )
    .unwrap();
    let tokenizer_metadata = tokenizer.metadata().clone();
    let tokenizer: Arc<dyn Tokenizer + Send + Sync> = Arc::new(tokenizer);
    let fixture =
        production_model_fixture(ingested.manifest.clone(), tokenizer_metadata, tokenizer).unwrap();
    let trust_store =
        ModelTrustStore::default().trust_digest(ingested.manifest.id.digest.value.clone());

    let report = run_campaign(
        CampaignTarget {
            evidence_class: EvidenceClass::FixtureOnly,
            checkpoint: CheckpointIdentity {
                name: "tiny-synthetic-production-bundle".into(),
                revision: None,
                weights_digest: Some(ingested.manifest.id.digest.value.clone()),
                digest_verified: false,
            },
            fixture,
            payload_source: ingested.payload_source.as_ref(),
            trust_store,
            provider: Arc::new(ReferenceCpuProvider::new()),
            prompt: "hi",
            max_new_tokens: 4,
            protocol: sequential_protocol(1, 5),
        },
        hardware_matrix(Some(("reference-cpu", true))),
    );

    assert_complete_trace(&report, 4);
    assert_eq!(report.samples.len(), 1 + 1 + 5);
    assert_eq!(
        report
            .warm_statistics
            .time_to_first_token_ms
            .as_ref()
            .unwrap()
            .count,
        5
    );
    assert!(
        report
            .exceptions
            .contains(&CampaignException::FixtureOnlyEvidence)
    );
    assert!(!report.verdict.qualifies_production_performance);
    assert!(
        report.hardware.production_validated().is_empty(),
        "a fixture run claims no production hardware capability"
    );
    assert!(report.profile_matrix.iter().any(|cell| matches!(
        &cell.status,
        CellStatus::Skipped { reason } if cell.concurrency == 32 && !reason.is_empty()
    )));
    let json = serde_json::to_value(&report).unwrap();
    assert_eq!(json["manifest"]["evidence_class"], "fixture_only");
}

#[test]
fn provider_names_become_single_file_name_components() {
    assert_eq!(
        file_label("magnetar:provider/cuda"),
        "magnetar-provider-cuda"
    );
    assert_eq!(file_label("reference-cpu"), "reference-cpu");
    let dir = tempfile::tempdir().unwrap();
    fs::write(
        dir.path()
            .join(format!("progress-{}.jsonl", file_label("a/b:c"))),
        b"{}",
    )
    .unwrap();
}

/// Task 4.3: with no CUDA Device, every CUDA capability is an explicit
/// skip with a reason and nothing is claimed for it; with one, discovery
/// passes but compute and production stages are not claimed by discovery
/// alone.
#[test]
fn hardware_matrix_never_turns_discovery_or_absence_into_a_capability_claim() {
    let matrix = hardware_matrix(None);
    let cuda = matrix.entries_for(HardwareCapability::CudaComputeCapability);
    assert_eq!(cuda.len(), 1);
    match &cuda[0].discovery {
        StageOutcome::Skipped { reason } => {
            assert!(!reason.is_empty());
            assert_eq!(cuda[0].device, None);
        }
        StageOutcome::Passed { .. } => assert!(cuda[0].device.is_some()),
        StageOutcome::Failed { reason } => panic!("discovery failed: {reason}"),
    }
    for entry in &matrix.entries {
        assert!(!entry.is_production_validated(), "{entry:?}");
    }
    assert!(
        matrix
            .entries_for(HardwareCapability::MemoryBandwidth)
            .iter()
            .all(|entry| entry.compute.is_passed()),
        "the bandwidth probe actually executes"
    );
}

/// The selected Providers that can run here, honoring
/// `MAGNETAR_QUALIFICATION_REQUIRE_CUDA=1`.
fn selected_real_providers() -> Vec<(&'static str, Arc<dyn Provider>)> {
    let selected = selected_providers();
    let require_cuda = std::env::var("MAGNETAR_QUALIFICATION_REQUIRE_CUDA").as_deref() == Ok("1");
    let mut providers: Vec<(&'static str, Arc<dyn Provider>)> = Vec::new();
    if selected.iter().any(|name| name == "reference-cpu") {
        providers.push(("reference-cpu", Arc::new(ReferenceCpuProvider::new())));
    }
    if selected.iter().any(|name| name == "cuda") {
        let cuda = magnetar_provider_cuda::CudaProvider::new();
        if cuda.is_available() {
            providers.push(("cuda", Arc::new(cuda) as Arc<dyn Provider>));
        } else {
            assert!(
                !require_cuda,
                "MAGNETAR_QUALIFICATION_REQUIRE_CUDA=1 but no CUDA-capable Device is available"
            );
            eprintln!("CUDA campaign skipped: no CUDA-capable Device available");
        }
    }
    providers
}

/// The pinned real checkpoint, digest-verified, ingested and trusted.
struct RealCheckpoint {
    ingested: magnetar_runtime::production_model_ingestion::ProductionIngestionResult,
    tokenizer_metadata: magnetar_runtime::tokenizer::TokenizerMetadata,
    tokenizer: Arc<dyn Tokenizer + Send + Sync>,
    trust_store: ModelTrustStore,
    checkpoint: CheckpointIdentity,
}

fn load_real_checkpoint(max_new_tokens: usize) -> RealCheckpoint {
    let dir = require_checkpoint_dir();
    let source = verified_checkpoint_source(&dir);
    register_real_qwen_component();
    let mut ingested = HuggingFaceIngestor::new()
        .ingest(&source)
        .expect("real Qwen2.5-0.5B-Instruct bundle ingests");
    ingested.manifest.generation = Some(ModelGenerationDefaults {
        max_tokens: Some(u32::try_from(max_new_tokens).expect("token budget fits u32")),
        ..Default::default()
    });
    let tokenizer_json = fs::read(dir.join("tokenizer.json")).unwrap();
    let tokenizer_config =
        parse_tokenizer_config(&fs::read(dir.join("tokenizer_config.json")).unwrap())
            .expect("real tokenizer_config.json parses");
    let tokenizer = HuggingFaceTokenizer::from_bytes(
        &tokenizer_json,
        Some(&tokenizer_config),
        "qwen2.5-0.5b-instruct",
        Some(REAL_VOCAB_SIZE),
    )
    .expect("real tokenizer.json loads");
    let tokenizer_metadata = tokenizer.metadata().clone();
    let trust_store =
        ModelTrustStore::default().trust_digest(ingested.manifest.id.digest.value.clone());
    RealCheckpoint {
        ingested,
        tokenizer_metadata,
        tokenizer: Arc::new(tokenizer),
        trust_store,
        checkpoint: CheckpointIdentity {
            name: "Qwen/Qwen2.5-0.5B-Instruct".into(),
            revision: Some(CHECKPOINT_REVISION.into()),
            weights_digest: Some(CHECKPOINT_WEIGHTS_SHA256.into()),
            // `verified_checkpoint_source` recomputed this digest from the
            // downloaded bytes (it panics otherwise).
            digest_verified: true,
        },
    }
}

/// Tasks 2.1/2.2, 4.2 and 5.1/5.2 on real data: the pinned real
/// Qwen2.5-0.5B-Instruct checkpoint, loaded once, one cold request, one
/// warmup and (by default) five measured requests on Reference CPU -- and
/// the same campaign on CUDA when a CUDA Device is available.
#[test]
#[ignore = "downloads/holds a ~1GB real checkpoint; manual/nightly qualification profile only"]
fn real_checkpoint_qualification_campaign() {
    // Provider selection first: a GPU job that requires CUDA fails before
    // spending minutes hashing and ingesting the checkpoint.
    let providers = selected_real_providers();
    let max_new_tokens = env_usize("MAGNETAR_QUALIFICATION_MAX_NEW_TOKENS", 16);
    let measured_runs = env_usize("MAGNETAR_QUALIFICATION_MEASURED_RUNS", 5);
    let RealCheckpoint {
        ingested,
        tokenizer_metadata,
        tokenizer,
        trust_store,
        checkpoint,
    } = load_real_checkpoint(max_new_tokens);

    for (device, provider) in providers {
        let fixture = production_model_fixture(
            ingested.manifest.clone(),
            tokenizer_metadata.clone(),
            Arc::clone(&tokenizer),
        )
        .expect("production fixture builds from the real ingested checkpoint");
        let report = run_campaign(
            CampaignTarget {
                evidence_class: EvidenceClass::RealCheckpoint,
                checkpoint: checkpoint.clone(),
                fixture,
                payload_source: ingested.payload_source.as_ref(),
                trust_store: trust_store.clone(),
                provider,
                prompt: REAL_PROMPT,
                max_new_tokens,
                protocol: sequential_protocol(1, measured_runs),
            },
            hardware_matrix(Some((device, true))),
        );

        assert_complete_trace(&report, max_new_tokens);
        assert!(
            !report
                .exceptions
                .contains(&CampaignException::FixtureOnlyEvidence)
        );
        assert!(
            !report.verdict.qualifies_production_performance,
            "A02/A07/A08 are still pending gates, so no production qualification is claimed"
        );
        let measured_token_ids: Vec<_> = report
            .samples
            .iter()
            .map(|sample| &sample.generated_token_ids)
            .collect();
        assert!(
            measured_token_ids.windows(2).all(|pair| pair[0] == pair[1]),
            "greedy generation on a resident instance is deterministic across runs"
        );
        let stats = &report.warm_statistics;
        eprintln!(
            "[{device}] load {:.0} ms; warm TTFT median {:.1} ms, TPOT median {:.1} ms, \
             decode {:.2} tok/s; peak RSS {:?}; exceptions {:?}",
            report.load.as_ref().unwrap().duration_ms,
            stats
                .time_to_first_token_ms
                .as_ref()
                .map_or(f64::NAN, |s| s.median),
            stats
                .time_per_output_token_ms
                .as_ref()
                .map_or(f64::NAN, |s| s.median),
            stats
                .decode_throughput_tokens_per_second
                .as_ref()
                .map_or(f64::NAN, |s| s.median),
            report.load.as_ref().unwrap().resources.peak_resident_bytes,
            report.exceptions,
        );
        write_report_if_requested(&report);
    }
}

/// Positions per bucket in the decode cost profile.
const PROFILE_BUCKET_TOKENS: usize = 8;
/// `magnetar_runtime`'s crate-private `INFERENCE_API_OBSERVATION_BUFFER_CAPACITY`:
/// the observer keeps only this many most recent events, so per-kind event
/// counts from a longer request are a lower bound, flagged as truncated.
const RUNTIME_OBSERVATION_BUFFER_CAPACITY: usize = 4096;

/// Diagnostic, not a qualification gate: where a decode step's cost comes
/// from. One warmed-up greedy request of `MAGNETAR_QUALIFICATION_PROFILE_TOKENS`
/// (default 64) tokens per selected Provider, reporting inter-token latency
/// by position bucket and its least-squares growth per token, the share of
/// time the Runtime spent inside its own prefill/decode steps, and how many
/// Runtime events (Kernel dispatches, Provider executions...) each token
/// costs. A growing latency means a step's work depends on history length;
/// a large outside-step share means per-request work around the steps. The
/// profile is written to `decode-cost-profile-<provider>.json` in
/// `MAGNETAR_QUALIFICATION_REPORT_DIR` and summarized to
/// `MAGNETAR_QUALIFICATION_SUMMARY_PATH`.
#[test]
#[ignore = "downloads/holds a ~1GB real checkpoint; manual/nightly diagnostic only"]
fn real_checkpoint_decode_cost_profile() {
    let providers = selected_real_providers();
    let tokens = env_usize("MAGNETAR_QUALIFICATION_PROFILE_TOKENS", 64);
    let real = load_real_checkpoint(tokens);
    for (device, provider) in providers {
        let fixture = production_model_fixture(
            real.ingested.manifest.clone(),
            real.tokenizer_metadata.clone(),
            Arc::clone(&real.tokenizer),
        )
        .unwrap();
        let mut model = ProductionLoadedModel::load(
            fixture,
            real.ingested.payload_source.as_ref(),
            real.trust_store.clone(),
            provider,
        )
        .expect("the real checkpoint loads");
        let request = |max_new_tokens| ProductionGenerationRequest {
            prompt: PromptInput::PlainText(REAL_PROMPT.into()),
            parameters: GenerationParameters::greedy(),
            stop_conditions: StopConditions::default(),
            max_new_tokens: Some(max_new_tokens),
            max_generation_millis: None,
        };
        // Warm the Component compilation and Kernel preparation caches.
        model
            .generate(request(2), None)
            .expect("warmup request generates");

        model.enable_phase_attribution();
        let mut recorder = TraceRecorder::start();
        let outcome = model
            .generate_streaming(request(tokens), None, &mut |event| {
                if let GenerationStreamEvent::Token { .. } = event {
                    recorder.token();
                }
                ControlFlow::Continue(())
            })
            .expect("profile request generates");
        let usage = outcome.result.output.usage.clone();
        let metrics = recorder.finish(usage.prompt_tokens).metrics();
        let attribution = model.take_phase_attribution();
        model.unload().expect("the real checkpoint unloads");
        let decode_phases = phase_rows(&attribution, StepKind::Decode);
        let preparation_phases = phase_rows(&attribution, StepKind::RequestPreparation);

        let buckets: Vec<serde_json::Value> = metrics
            .inter_token_latencies_ms
            .chunks(PROFILE_BUCKET_TOKENS)
            .enumerate()
            .map(|(bucket, latencies)| {
                let first_position = usage.prompt_tokens + 1 + bucket * PROFILE_BUCKET_TOKENS;
                serde_json::json!({
                    "first_position": first_position,
                    "tokens": latencies.len(),
                    "mean_inter_token_latency_ms":
                        latencies.iter().sum::<f64>() / latencies.len() as f64,
                })
            })
            .collect();
        let observations = outcome.observer.observations();
        let truncated = observations.len() >= RUNTIME_OBSERVATION_BUFFER_CAPACITY;
        let mut events_by_kind: BTreeMap<String, usize> = BTreeMap::new();
        for observation in observations {
            *events_by_kind
                .entry(format!("{:?}", observation.kind))
                .or_insert(0) += 1;
        }
        let step_ms = usage
            .prefill_duration_millis
            .unwrap_or(0)
            .saturating_add(usage.decode_duration_millis.unwrap_or(0));
        let profile = serde_json::json!({
            "provider": device,
            "prompt_tokens": usage.prompt_tokens,
            "generated_tokens": metrics.generated_tokens,
            "time_to_first_token_ms": metrics.time_to_first_token_ms,
            "time_per_output_token_ms": metrics.time_per_output_token_ms,
            "inter_token_latency_slope_ms_per_token": metrics.inter_token_latency_slope_ms_per_token,
            "inter_token_latency_buckets": buckets,
            "end_to_end_latency_ms": metrics.end_to_end_latency_ms,
            "runtime_prefill_ms": usage.prefill_duration_millis,
            "runtime_decode_ms": usage.decode_duration_millis,
            "outside_runtime_steps_ms": metrics.end_to_end_latency_ms - step_ms as f64,
            "runtime_events_by_kind": events_by_kind,
            "runtime_events_truncated": truncated,
            "decode_phase_attribution": decode_phases,
            "request_preparation_phase_attribution": preparation_phases,
            "phase_attribution": attribution,
        });
        eprintln!("[{device}] decode cost profile: {profile}");
        assert_eq!(
            metrics.generated_tokens, tokens,
            "every requested token is generated"
        );

        if let Some(dir) = std::env::var_os("MAGNETAR_QUALIFICATION_REPORT_DIR").map(PathBuf::from)
        {
            fs::create_dir_all(&dir).unwrap();
            fs::write(
                dir.join(format!("decode-cost-profile-{device}.json")),
                serde_json::to_vec_pretty(&profile).unwrap(),
            )
            .unwrap();
        }
        if let Some(path) = std::env::var_os("MAGNETAR_QUALIFICATION_SUMMARY_PATH") {
            let mut summary = format!(
                "### Decode cost profile: {device}\n\n{} prompt tokens, {} generated. \
                 TTFT {:.1} ms, TPOT {:.1} ms, ITL growth {:.3} ms/token; \
                 {:.0} of {:.0} ms inside Runtime steps.\n\n\
                 | First position | Tokens | Mean ITL (ms) |\n|---|---|---|\n",
                usage.prompt_tokens,
                metrics.generated_tokens,
                metrics.time_to_first_token_ms.unwrap_or(f64::NAN),
                metrics.time_per_output_token_ms.unwrap_or(f64::NAN),
                metrics
                    .inter_token_latency_slope_ms_per_token
                    .unwrap_or(f64::NAN),
                step_ms as f64,
                metrics.end_to_end_latency_ms,
            );
            for bucket in profile["inter_token_latency_buckets"].as_array().unwrap() {
                summary.push_str(&format!(
                    "| {} | {} | {:.1} |\n",
                    bucket["first_position"],
                    bucket["tokens"],
                    bucket["mean_inter_token_latency_ms"]
                        .as_f64()
                        .unwrap_or(f64::NAN)
                ));
            }
            for (title, rows) in [
                ("decode step", &decode_phases),
                ("request preparation", &preparation_phases),
            ] {
                summary.push_str(&format!(
                    "\nHost time per {title}, by phase (device time unavailable):\n\n\
                     | Phase | Operator | Calls | ms | Share |\n|---|---|---|---|---|\n",
                ));
                for row in rows {
                    summary.push_str(&format!(
                        "| {} | {} | {:.1} | {:.2} | {:.1}% |\n",
                        row["phase"].as_str().unwrap_or(""),
                        row["operator"].as_str().unwrap_or(""),
                        row["calls_per_step"].as_f64().unwrap_or(f64::NAN),
                        row["ms_per_step"].as_f64().unwrap_or(f64::NAN),
                        row["share"].as_f64().unwrap_or(f64::NAN) * 100.0,
                    ));
                }
            }
            let mut file = fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(path)
                .unwrap();
            std::io::Write::write_all(&mut file, format!("{summary}\n").as_bytes()).unwrap();
        }
    }
}

/// Mean host time per step of `kind` for every phase, largest first, with
/// the step time no phase covers as a final `unattributed` row.
fn phase_rows(report: &PhaseAttributionReport, kind: StepKind) -> Vec<serde_json::Value> {
    let Some(summary) = report.summary(kind) else {
        return Vec::new();
    };
    let steps = summary.steps.max(1) as f64;
    let wall = summary.wall_micros.max(1) as f64;
    let row = |phase: String, operator: Option<&str>, count: u64, micros: u64| {
        serde_json::json!({
            "phase": phase,
            "operator": operator,
            "calls_per_step": count as f64 / steps,
            "ms_per_step": micros as f64 / steps / 1_000.0,
            "share": micros as f64 / wall,
        })
    };
    let mut phases: Vec<_> = summary.phases.iter().collect();
    phases.sort_by_key(|(_, total)| std::cmp::Reverse(total.host_micros));
    let mut rows: Vec<_> = phases
        .into_iter()
        .map(|(key, total)| {
            row(
                serde_json::to_value(key.phase)
                    .ok()
                    .and_then(|value| value.as_str().map(str::to_string))
                    .unwrap_or_default(),
                key.operator.as_deref(),
                total.count,
                total.host_micros,
            )
        })
        .collect();
    rows.push(row(
        "unattributed".into(),
        None,
        0,
        summary.unattributed_micros,
    ));
    rows
}
