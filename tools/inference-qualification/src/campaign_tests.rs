use super::*;
use crate::{
    hardware::{CapabilityEvidence, HardwareCapability, HostIdentity},
    resources::IntervalProbe,
};

fn manifest(evidence_class: EvidenceClass, measured_runs: usize) -> RunManifest {
    RunManifest {
        schema_version: REPORT_SCHEMA_VERSION.into(),
        harness: "unit-test".into(),
        evidence_class,
        checkpoint: CheckpointIdentity {
            name: "example".into(),
            revision: Some("abc".into()),
            weights_digest: Some("sha256:00".into()),
            digest_verified: evidence_class == EvidenceClass::RealCheckpoint,
        },
        code_revision: None,
        provider: ProviderIdentity {
            name: "reference-cpu".into(),
            version: None,
            driver: None,
        },
        dtype: "f32".into(),
        prompt_tokens: 3,
        max_new_tokens: 4,
        batch_size: 1,
        concurrency: 1,
        sampling: "greedy".into(),
        build_profile: "test".into(),
        host_os: "linux".into(),
        host_arch: "x86_64".into(),
        flags: BTreeMap::new(),
        protocol: CampaignProtocol {
            cold_run: true,
            warmup_runs: 1,
            measured_runs,
            ordering: "sequential".into(),
            interference_controls: vec![],
        },
    }
}

fn trace(first_token_micros: u64) -> GenerationTrace {
    GenerationTrace {
        prompt_tokens: 3,
        token_offsets_micros: vec![
            first_token_micros,
            first_token_micros + 10_000,
            first_token_micros + 20_000,
        ],
        completed_micros: first_token_micros + 25_000,
    }
}

fn hardware() -> HardwareCapabilityMatrix {
    HardwareCapabilityMatrix::new(
        HostIdentity::discover(),
        vec![CapabilityEvidence::absent(
            HardwareCapability::CudaComputeCapability,
            "no CUDA Device",
        )],
    )
}

fn run_cell() -> Vec<ProfileCell> {
    vec![ProfileCell {
        concurrency: 1,
        context_tokens: 3,
        generated_tokens: 3,
        status: CellStatus::Run,
    }]
}

fn campaign(evidence_class: EvidenceClass, measured_runs: usize) -> Campaign {
    let mut campaign = Campaign::new(manifest(evidence_class, measured_runs));
    campaign.record_load(LoadSample {
        duration_ms: 12.0,
        resources: IntervalProbe::begin().end(),
    });
    campaign.record(
        RunPhase::Cold,
        trace(90_000),
        IntervalProbe::begin().end(),
        vec![1, 2, 3],
        "length",
    );
    campaign.record(
        RunPhase::Warmup,
        trace(50_000),
        IntervalProbe::begin().end(),
        vec![1, 2, 3],
        "length",
    );
    for _ in 0..measured_runs {
        campaign.record(
            RunPhase::Measured,
            trace(40_000),
            IntervalProbe::begin().end(),
            vec![1, 2, 3],
            "length",
        );
    }
    campaign
}

#[test]
fn complete_real_campaign_retains_raw_samples_and_statistics() {
    let report = campaign(EvidenceClass::RealCheckpoint, 5).finish(hardware(), run_cell(), vec![]);
    assert_eq!(
        report.samples.len(),
        7,
        "cold + warmup + 5 measured, all retained"
    );
    assert_eq!(report.samples[0].phase, RunPhase::Cold);
    assert_eq!(report.samples[6].ordinal, 4);
    let ttft = report
        .warm_statistics
        .time_to_first_token_ms
        .as_ref()
        .unwrap();
    assert_eq!(ttft.count, 5, "statistics cover measured runs only");
    assert_eq!(ttft.median, 40.0);
    assert_eq!(
        report
            .warm_statistics
            .inter_token_latency_ms
            .as_ref()
            .unwrap()
            .count,
        10
    );
    assert!(report.exceptions.is_empty(), "{:?}", report.exceptions);
    assert!(report.verdict.qualifies_production_performance);
    assert!(report.manifest_digest_matches());
}

#[test]
fn fixture_only_campaign_is_labeled_and_never_qualifies() {
    let report = campaign(EvidenceClass::FixtureOnly, 5).finish(hardware(), run_cell(), vec![]);
    assert!(
        report
            .exceptions
            .contains(&CampaignException::FixtureOnlyEvidence)
    );
    assert!(
        report
            .exceptions
            .contains(&CampaignException::UnverifiedCheckpointDigest)
    );
    assert!(!report.verdict.qualifies_production_performance);
}

#[test]
fn too_few_repetitions_unrun_cells_and_pending_gates_are_recorded() {
    let mut cells = run_cell();
    cells.push(ProfileCell {
        concurrency: 32,
        context_tokens: 16384,
        generated_tokens: 512,
        status: CellStatus::Skipped {
            reason: "no batching path".into(),
        },
    });
    let report = campaign(EvidenceClass::RealCheckpoint, 2).finish(
        hardware(),
        cells,
        vec![("A02".into(), "logit references not implemented".into())],
    );
    assert!(report.exceptions.iter().any(|exception| matches!(
        exception,
        CampaignException::InsufficientRepetitions {
            measured: 2,
            minimum: MINIMUM_WARM_REPETITIONS,
            ..
        }
    )));
    assert!(
        report
            .exceptions
            .contains(&CampaignException::UnrunProfileCells { count: 1 })
    );
    assert!(report.exceptions.iter().any(
        |exception| matches!(exception, CampaignException::PendingGate { gate, .. } if gate == "A02")
    ));
    assert!(!report.verdict.qualifies_production_performance);
    assert_eq!(report.verdict.blocking_exceptions, report.exceptions.len());
}

#[test]
fn noisy_or_missing_metrics_block_qualification() {
    let mut noisy = Campaign::new(manifest(EvidenceClass::RealCheckpoint, 5));
    for first_token in [10_000, 40_000, 80_000, 120_000, 200_000] {
        noisy.record(
            RunPhase::Measured,
            trace(first_token),
            IntervalProbe::begin().end(),
            vec![],
            "length",
        );
    }
    let report = noisy.finish(hardware(), run_cell(), vec![]);
    assert!(report.exceptions.iter().any(|exception| matches!(
        exception,
        CampaignException::HighDispersion { metric, .. } if metric == "time_to_first_token_ms"
    )));

    let empty = Campaign::new(manifest(EvidenceClass::RealCheckpoint, 0)).finish(
        hardware(),
        run_cell(),
        vec![],
    );
    assert!(
        empty
            .exceptions
            .contains(&CampaignException::MissingMetric {
                metric: "time_to_first_token_ms".into()
            })
    );
    assert!(!empty.verdict.qualifies_production_performance);
}

#[test]
fn manifest_digest_is_deterministic_and_sensitive_to_identity() {
    let a = manifest(EvidenceClass::RealCheckpoint, 5);
    assert_eq!(a.digest(), a.clone().digest());
    assert!(a.digest().starts_with("sha256:"));
    let mut b = a.clone();
    b.provider.name = "cuda".into();
    assert_ne!(a.digest(), b.digest());

    let mut report =
        campaign(EvidenceClass::RealCheckpoint, 5).finish(hardware(), run_cell(), vec![]);
    report.manifest.max_new_tokens = 999;
    assert!(!report.manifest_digest_matches(), "tampering is detectable");
}

#[test]
fn reports_are_written_once_and_never_overwritten() {
    let dir = std::env::temp_dir().join(format!(
        "magnetar-inference-qualification-{}",
        std::process::id()
    ));
    let report = campaign(EvidenceClass::FixtureOnly, 1).finish(hardware(), run_cell(), vec![]);
    let path = report.write_immutable(&dir).unwrap();
    let round_trip: CampaignReport = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
    assert_eq!(round_trip, report);
    assert!(
        report.write_immutable(&dir).is_err(),
        "an existing report is immutable"
    );
    fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn standard_grid_covers_every_declared_combination() {
    let grid = standard_profile_grid();
    assert_eq!(grid.len(), 4 * 3 * 2);
    assert!(grid.contains(&(1, 512, 128)));
    assert!(grid.contains(&(32, 16384, 512)));
}

#[test]
fn markdown_summary_highlights_medians_and_the_verdict() {
    let report = campaign(EvidenceClass::FixtureOnly, 5).finish(hardware(), run_cell(), vec![]);
    let summary = report.markdown_summary();
    assert!(summary.contains("example on reference-cpu"), "{summary}");
    assert!(summary.contains("@abc"));
    assert!(
        summary.contains("| TTFT | 40.00 ms | 40.00 ms | 5 |"),
        "{summary}"
    );
    assert!(summary.contains("| Decode throughput |"));
    assert!(summary.contains("Qualifies production performance: **no**"));
    assert!(summary.contains("fixture_only_evidence"));
    assert!(summary.contains(&report.manifest_digest));

    let empty = Campaign::new(manifest(EvidenceClass::RealCheckpoint, 0)).finish(
        hardware(),
        run_cell(),
        vec![],
    );
    assert!(
        empty
            .markdown_summary()
            .contains("| TTFT | not measured | | |")
    );
}
