//! Repeatable cold/warm campaigns (A01/A04): an immutable run manifest,
//! every raw sample, derived statistics, recorded exceptions and a
//! qualification verdict that fixture-only, under-repeated, noisy or
//! partially-gated evidence can never pass.

use std::{
    collections::BTreeMap,
    fs,
    io::Write,
    path::{Path, PathBuf},
};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::{
    hardware::HardwareCapabilityMatrix,
    resources::ResourceUsage,
    statistics::{SampleSummary, summarize},
    trace::{GenerationMetrics, GenerationTrace},
};

pub const REPORT_SCHEMA_VERSION: &str = "magnetar.inference-qualification.report/v1";

/// The fewest post-warmup repetitions a campaign needs before its
/// statistics may support any claim.
pub const MINIMUM_WARM_REPETITIONS: usize = 5;

/// Relative dispersion (MAD / median) above which a warm metric is
/// recorded as too noisy to qualify.
pub const MAXIMUM_RELATIVE_DISPERSION: f64 = 0.10;

/// What kind of model the campaign actually executed.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceClass {
    /// A synthetic or tiny fixture bundle. Useful for fast harness
    /// checks; never qualifies production performance.
    FixtureOnly,
    /// A real, digest-verified checkpoint.
    RealCheckpoint,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CheckpointIdentity {
    pub name: String,
    pub revision: Option<String>,
    pub weights_digest: Option<String>,
    /// Whether the harness recomputed `weights_digest` from the bytes it
    /// executed, rather than trusting a declared value.
    pub digest_verified: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ProviderIdentity {
    pub name: String,
    pub version: Option<String>,
    pub driver: Option<String>,
}

/// The warmup/ordering/isolation protocol a campaign followed.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CampaignProtocol {
    /// Whether the first request after load is recorded separately as the
    /// cold run.
    pub cold_run: bool,
    pub warmup_runs: usize,
    pub measured_runs: usize,
    /// How runs were ordered (e.g. sequential, interleaved).
    pub ordering: String,
    /// Steps taken to limit interference between runs.
    pub interference_controls: Vec<String>,
}

/// Everything that identifies a campaign. Its SHA-256 over the canonical
/// JSON encoding is the report's immutable identity.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RunManifest {
    pub schema_version: String,
    pub harness: String,
    pub evidence_class: EvidenceClass,
    pub checkpoint: CheckpointIdentity,
    pub code_revision: Option<String>,
    pub provider: ProviderIdentity,
    pub dtype: String,
    pub prompt_tokens: usize,
    pub max_new_tokens: usize,
    pub batch_size: usize,
    pub concurrency: usize,
    pub sampling: String,
    pub build_profile: String,
    pub host_os: String,
    pub host_arch: String,
    pub flags: BTreeMap<String, String>,
    pub protocol: CampaignProtocol,
}

impl RunManifest {
    /// `sha256:<hex>` over this manifest's JSON encoding. Field order is
    /// the struct's declaration order and `flags` is a sorted map, so the
    /// encoding is deterministic.
    pub fn digest(&self) -> String {
        let encoded = serde_json::to_vec(self).expect("a run manifest always serializes");
        let mut out = String::from("sha256:");
        for byte in Sha256::digest(&encoded) {
            out.push_str(&format!("{byte:02x}"));
        }
        out
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RunPhase {
    Cold,
    Warmup,
    Measured,
}

/// One executed request, with its raw trace and the metrics derived from
/// it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RunSample {
    pub phase: RunPhase,
    pub ordinal: usize,
    pub trace: GenerationTrace,
    pub metrics: GenerationMetrics,
    pub resources: ResourceUsage,
    pub generated_token_ids: Vec<u32>,
    pub finish_reason: String,
}

/// The one-time model load preceding the cold run.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LoadSample {
    pub duration_ms: f64,
    pub resources: ResourceUsage,
}

/// Summaries over the measured (post-warmup) samples only.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct WarmStatistics {
    pub time_to_first_token_ms: Option<SampleSummary>,
    pub time_per_output_token_ms: Option<SampleSummary>,
    pub inter_token_latency_ms: Option<SampleSummary>,
    pub end_to_end_latency_ms: Option<SampleSummary>,
    pub request_throughput_tokens_per_second: Option<SampleSummary>,
    pub decode_throughput_tokens_per_second: Option<SampleSummary>,
    pub inter_token_latency_slope_ms_per_token: Option<SampleSummary>,
}

/// One cell of the concurrency x context x generation profile grid.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ProfileCell {
    pub concurrency: usize,
    pub context_tokens: usize,
    pub generated_tokens: usize,
    pub status: CellStatus,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum CellStatus {
    Run,
    Unsupported { reason: String },
    OutOfMemory { reason: String },
    Skipped { reason: String },
}

/// The design's standard grid: concurrency 1/4/16/32, context
/// 512/4096/16384 and generation 128/512 tokens.
pub fn standard_profile_grid() -> Vec<(usize, usize, usize)> {
    let mut grid = Vec::new();
    for concurrency in [1, 4, 16, 32] {
        for context_tokens in [512, 4096, 16384] {
            for generated_tokens in [128, 512] {
                grid.push((concurrency, context_tokens, generated_tokens));
            }
        }
    }
    grid
}

/// Something that limits what a campaign may claim.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum CampaignException {
    FixtureOnlyEvidence,
    UnverifiedCheckpointDigest,
    InsufficientRepetitions {
        measured: usize,
        minimum: usize,
        reason: String,
    },
    HighDispersion {
        metric: String,
        relative_dispersion: f64,
        threshold: f64,
    },
    MissingMetric {
        metric: String,
    },
    UnrunProfileCells {
        count: usize,
    },
    PendingGate {
        gate: String,
        reason: String,
    },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct QualificationVerdict {
    pub qualifies_production_performance: bool,
    pub blocking_exceptions: usize,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CampaignReport {
    pub manifest: RunManifest,
    pub manifest_digest: String,
    pub load: Option<LoadSample>,
    pub samples: Vec<RunSample>,
    pub warm_statistics: WarmStatistics,
    pub hardware: HardwareCapabilityMatrix,
    pub profile_matrix: Vec<ProfileCell>,
    pub exceptions: Vec<CampaignException>,
    pub verdict: QualificationVerdict,
}

impl CampaignReport {
    /// Whether the stored digest still matches the stored manifest.
    pub fn manifest_digest_matches(&self) -> bool {
        self.manifest.digest() == self.manifest_digest
    }

    /// Writes this report as `<dir>/<manifest digest hex>.json`. Refuses to
    /// overwrite: a manifest identifies exactly one immutable result.
    pub fn write_immutable(&self, dir: &Path) -> std::io::Result<PathBuf> {
        let file_name = format!(
            "{}.json",
            self.manifest_digest.trim_start_matches("sha256:")
        );
        let path = dir.join(file_name);
        fs::create_dir_all(dir)?;
        let mut file = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)?;
        file.write_all(&serde_json::to_vec_pretty(self).map_err(std::io::Error::other)?)?;
        Ok(path)
    }

    /// A short Markdown summary (e.g. for a CI job summary). The JSON
    /// report stays the authoritative record; this only highlights the
    /// warm medians/p95 and the verdict.
    pub fn markdown_summary(&self) -> String {
        let manifest = &self.manifest;
        let summary_row = |label: &str, summary: &Option<SampleSummary>, unit: &str| {
            summary.as_ref().map_or_else(
                || format!("| {label} | not measured | | |\n"),
                |s| {
                    format!(
                        "| {label} | {:.2} {unit} | {:.2} {unit} | {} |\n",
                        s.median, s.p95, s.count
                    )
                },
            )
        };
        let stats = &self.warm_statistics;
        let mut out = format!(
            "### Qualification campaign: {} on {}\n\n\
             Checkpoint `{}`{}, {} prompt tokens, {} new tokens, batch {}, {} build. \
             Report `{}`.\n\n",
            manifest.checkpoint.name,
            manifest.provider.name,
            manifest.checkpoint.name,
            manifest
                .checkpoint
                .revision
                .as_deref()
                .map(|revision| format!("@{revision}"))
                .unwrap_or_default(),
            manifest.prompt_tokens,
            manifest.max_new_tokens,
            manifest.batch_size,
            manifest.build_profile,
            self.manifest_digest,
        );
        if let Some(load) = &self.load {
            out.push_str(&format!("Model load: {:.0} ms.\n\n", load.duration_ms));
        }
        out.push_str("| Warm metric | Median | p95 | Samples |\n|---|---|---|---|\n");
        out.push_str(&summary_row("TTFT", &stats.time_to_first_token_ms, "ms"));
        out.push_str(&summary_row("TPOT", &stats.time_per_output_token_ms, "ms"));
        out.push_str(&summary_row(
            "Inter-token latency",
            &stats.inter_token_latency_ms,
            "ms",
        ));
        out.push_str(&summary_row(
            "End-to-end",
            &stats.end_to_end_latency_ms,
            "ms",
        ));
        out.push_str(&summary_row(
            "Decode throughput",
            &stats.decode_throughput_tokens_per_second,
            "tok/s",
        ));
        out.push_str(&summary_row(
            "ITL growth per token",
            &stats.inter_token_latency_slope_ms_per_token,
            "ms",
        ));
        out.push_str(&format!(
            "\nQualifies production performance: **{}** ({} blocking exception(s)).\n",
            if self.verdict.qualifies_production_performance {
                "yes"
            } else {
                "no"
            },
            self.verdict.blocking_exceptions
        ));
        for exception in &self.exceptions {
            out.push_str(&format!(
                "- `{}`\n",
                serde_json::to_string(exception).unwrap_or_default()
            ));
        }
        out
    }
}

/// Accumulates one campaign's samples in execution order.
#[derive(Debug)]
pub struct Campaign {
    manifest: RunManifest,
    load: Option<LoadSample>,
    samples: Vec<RunSample>,
}

impl Campaign {
    pub fn new(manifest: RunManifest) -> Self {
        Self {
            manifest,
            load: None,
            samples: Vec::new(),
        }
    }

    pub fn manifest(&self) -> &RunManifest {
        &self.manifest
    }

    pub fn record_load(&mut self, load: LoadSample) {
        self.load = Some(load);
    }

    pub fn record(
        &mut self,
        phase: RunPhase,
        trace: GenerationTrace,
        resources: ResourceUsage,
        generated_token_ids: Vec<u32>,
        finish_reason: impl Into<String>,
    ) {
        let ordinal = self.samples.iter().filter(|s| s.phase == phase).count();
        self.samples.push(RunSample {
            phase,
            ordinal,
            metrics: trace.metrics(),
            trace,
            resources,
            generated_token_ids,
            finish_reason: finish_reason.into(),
        });
    }

    /// Finalizes the report. `pending_gates` names the qualification
    /// gates (`(gate, reason)`) this campaign does not cover; any pending
    /// gate blocks production qualification.
    pub fn finish(
        self,
        hardware: HardwareCapabilityMatrix,
        profile_matrix: Vec<ProfileCell>,
        pending_gates: Vec<(String, String)>,
    ) -> CampaignReport {
        let measured: Vec<&GenerationMetrics> = self
            .samples
            .iter()
            .filter(|sample| sample.phase == RunPhase::Measured)
            .map(|sample| &sample.metrics)
            .collect();
        let collect = |select: fn(&GenerationMetrics) -> Option<f64>| {
            summarize(
                &measured
                    .iter()
                    .filter_map(|m| select(m))
                    .collect::<Vec<_>>(),
            )
        };
        let inter_token: Vec<f64> = measured
            .iter()
            .flat_map(|m| m.inter_token_latencies_ms.iter().copied())
            .collect();
        let warm_statistics = WarmStatistics {
            time_to_first_token_ms: collect(|m| m.time_to_first_token_ms),
            time_per_output_token_ms: collect(|m| m.time_per_output_token_ms),
            inter_token_latency_ms: summarize(&inter_token),
            end_to_end_latency_ms: collect(|m| Some(m.end_to_end_latency_ms)),
            request_throughput_tokens_per_second: collect(|m| {
                m.request_throughput_tokens_per_second
            }),
            decode_throughput_tokens_per_second: collect(|m| m.decode_throughput_tokens_per_second),
            inter_token_latency_slope_ms_per_token: collect(|m| {
                m.inter_token_latency_slope_ms_per_token
            }),
        };

        let mut exceptions = Vec::new();
        if self.manifest.evidence_class == EvidenceClass::FixtureOnly {
            exceptions.push(CampaignException::FixtureOnlyEvidence);
        }
        if !self.manifest.checkpoint.digest_verified {
            exceptions.push(CampaignException::UnverifiedCheckpointDigest);
        }
        if measured.len() < MINIMUM_WARM_REPETITIONS {
            exceptions.push(CampaignException::InsufficientRepetitions {
                measured: measured.len(),
                minimum: MINIMUM_WARM_REPETITIONS,
                reason: format!(
                    "protocol requested {} measured runs",
                    self.manifest.protocol.measured_runs
                ),
            });
        }
        for (metric, summary) in [
            (
                "time_to_first_token_ms",
                &warm_statistics.time_to_first_token_ms,
            ),
            (
                "time_per_output_token_ms",
                &warm_statistics.time_per_output_token_ms,
            ),
            (
                "end_to_end_latency_ms",
                &warm_statistics.end_to_end_latency_ms,
            ),
        ] {
            match summary {
                None => exceptions.push(CampaignException::MissingMetric {
                    metric: metric.into(),
                }),
                Some(summary) => {
                    if let Some(relative) = summary.relative_dispersion
                        && relative > MAXIMUM_RELATIVE_DISPERSION
                    {
                        exceptions.push(CampaignException::HighDispersion {
                            metric: metric.into(),
                            relative_dispersion: relative,
                            threshold: MAXIMUM_RELATIVE_DISPERSION,
                        });
                    }
                }
            }
        }
        let unrun = profile_matrix
            .iter()
            .filter(|cell| cell.status != CellStatus::Run)
            .count();
        if unrun > 0 {
            exceptions.push(CampaignException::UnrunProfileCells { count: unrun });
        }
        exceptions.extend(
            pending_gates
                .into_iter()
                .map(|(gate, reason)| CampaignException::PendingGate { gate, reason }),
        );

        CampaignReport {
            manifest_digest: self.manifest.digest(),
            manifest: self.manifest,
            load: self.load,
            samples: self.samples,
            warm_statistics,
            hardware,
            profile_matrix,
            verdict: QualificationVerdict {
                qualifies_production_performance: exceptions.is_empty(),
                blocking_exceptions: exceptions.len(),
            },
            exceptions,
        }
    }
}

#[cfg(test)]
#[path = "campaign_tests.rs"]
mod tests;
