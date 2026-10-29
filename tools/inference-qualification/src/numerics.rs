//! Checkpoint-bound logit references and cross-Provider numerical
//! comparison (A02). A reference keeps, per generation step, the sampled
//! token, the top logits by value and a digest of the full vector, so a
//! later run on any Provider can be compared under a declared numerical
//! profile -- and a run whose decoded text matches but whose logits drift
//! beyond tolerance still fails.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::campaign::CheckpointIdentity;

pub const LOGIT_REFERENCE_SCHEMA_VERSION: &str =
    "magnetar.inference-qualification.logit-reference/v1";

/// How many top logits each snapshot stores. Comparisons use a smaller
/// `compared_top_k`, so an id that slips a few ranks on the candidate is
/// still found rather than reported missing.
pub const SNAPSHOT_TOP_K: usize = 32;

/// A declared numerical acceptance profile.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct NumericalProfile {
    pub name: String,
    pub dtype: String,
    /// Largest accepted absolute difference of any compared logit.
    pub max_abs_logit_difference: f32,
    /// Whether every step must sample the same token.
    pub require_identical_tokens: bool,
    /// How many of the reference's top logits are compared per step.
    pub compared_top_k: usize,
}

impl NumericalProfile {
    /// Initial F32 cross-Provider profile: identical greedy tokens and
    /// every compared logit within 5e-2 of the reference. CPU/GPU F32
    /// accumulation-order differences on Qwen2.5-0.5B logits (magnitudes
    /// of order 10) are expected well below this; anything larger is a
    /// numerical defect, not rounding.
    pub fn f32_cross_provider() -> Self {
        Self {
            name: "f32-cross-provider-v1".into(),
            dtype: "f32".into(),
            max_abs_logit_difference: 5e-2,
            require_identical_tokens: true,
            compared_top_k: 8,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct TopLogit {
    pub token_id: u32,
    pub logit: f32,
}

/// One generation step's logits, reduced to what a reference needs.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LogitStepSnapshot {
    pub step: usize,
    pub sampled_token_id: u32,
    pub vocab_size: usize,
    /// Highest logits first; ties broken by lower token id.
    pub top: Vec<TopLogit>,
    /// `sha256:<hex>` over the full vector's little-endian f32 bytes.
    pub full_digest: String,
    pub non_finite_count: usize,
}

impl LogitStepSnapshot {
    pub fn capture(step: usize, logits: &[f32], sampled_token_id: u32) -> Self {
        let mut ranked: Vec<(usize, f32)> = logits
            .iter()
            .copied()
            .enumerate()
            .filter(|(_, logit)| logit.is_finite())
            .collect();
        let keep = SNAPSHOT_TOP_K.min(ranked.len());
        if keep > 0 && keep < ranked.len() {
            ranked.select_nth_unstable_by(keep - 1, rank_order);
            ranked.truncate(keep);
        }
        ranked.sort_by(rank_order);
        let mut hasher = Sha256::new();
        for logit in logits {
            hasher.update(logit.to_le_bytes());
        }
        let mut full_digest = String::from("sha256:");
        for byte in hasher.finalize() {
            full_digest.push_str(&format!("{byte:02x}"));
        }
        Self {
            step,
            sampled_token_id,
            vocab_size: logits.len(),
            top: ranked
                .into_iter()
                .map(|(token_id, logit)| TopLogit {
                    token_id: u32::try_from(token_id).unwrap_or(u32::MAX),
                    logit,
                })
                .collect(),
            full_digest,
            non_finite_count: logits.iter().filter(|logit| !logit.is_finite()).count(),
        }
    }

    fn logit_of(&self, token_id: u32) -> Option<f32> {
        self.top
            .iter()
            .find(|entry| entry.token_id == token_id)
            .map(|entry| entry.logit)
    }
}

fn rank_order(a: &(usize, f32), b: &(usize, f32)) -> std::cmp::Ordering {
    b.1.total_cmp(&a.1).then(a.0.cmp(&b.0))
}

/// Logits of one complete greedy generation, bound to the checkpoint,
/// Provider and exact prompt token ids that produced them.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LogitReference {
    pub schema_version: String,
    pub checkpoint: CheckpointIdentity,
    pub provider: String,
    pub dtype: String,
    pub prompt_token_ids: Vec<u32>,
    pub generated_token_ids: Vec<u32>,
    pub steps: Vec<LogitStepSnapshot>,
}

/// Records snapshots step by step as logits are observed.
#[derive(Debug, Default)]
pub struct LogitRecorder {
    pending: Vec<Vec<f32>>,
}

impl LogitRecorder {
    pub fn observe(&mut self, logits: &[f32]) {
        self.pending.push(logits.to_vec());
    }

    /// Pairs each observed step with the token it sampled. Extra observed
    /// steps without a delivered token (none in a normal run) are kept
    /// with `u32::MAX` as their sampled id so a mismatch is visible.
    pub fn finish(
        self,
        checkpoint: CheckpointIdentity,
        provider: impl Into<String>,
        dtype: impl Into<String>,
        prompt_token_ids: Vec<u32>,
        generated_token_ids: Vec<u32>,
    ) -> LogitReference {
        let steps = self
            .pending
            .iter()
            .enumerate()
            .map(|(step, logits)| {
                LogitStepSnapshot::capture(
                    step,
                    logits,
                    generated_token_ids.get(step).copied().unwrap_or(u32::MAX),
                )
            })
            .collect();
        LogitReference {
            schema_version: LOGIT_REFERENCE_SCHEMA_VERSION.into(),
            checkpoint,
            provider: provider.into(),
            dtype: dtype.into(),
            prompt_token_ids,
            generated_token_ids,
            steps,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StepComparison {
    pub step: usize,
    pub reference_token_id: u32,
    pub candidate_token_id: u32,
    pub bit_identical: bool,
    pub compared_logits: usize,
    pub max_abs_difference: f32,
    pub worst_token_id: Option<u32>,
    /// Reference top ids absent from the candidate's stored top logits.
    pub missing_token_ids: Vec<u32>,
}

/// The full differential evidence of one comparison. Kept whole on
/// failure: every step's differences, not only the first violation.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LogitComparison {
    pub profile: NumericalProfile,
    pub reference_provider: String,
    pub candidate_provider: String,
    pub steps: Vec<StepComparison>,
    pub max_abs_difference: f32,
    pub failures: Vec<String>,
}

impl LogitComparison {
    pub fn passed(&self) -> bool {
        self.failures.is_empty()
    }
}

/// Compares `candidate` against `reference` under `profile`.
pub fn compare_logit_references(
    reference: &LogitReference,
    candidate: &LogitReference,
    profile: &NumericalProfile,
) -> LogitComparison {
    let mut failures = Vec::new();
    if reference.checkpoint.weights_digest != candidate.checkpoint.weights_digest {
        failures.push("reference and candidate are bound to different checkpoint digests".into());
    }
    if reference.prompt_token_ids != candidate.prompt_token_ids {
        failures.push("reference and candidate used different prompt token ids".into());
    }
    if reference.steps.len() != candidate.steps.len() {
        failures.push(format!(
            "step count differs: reference {}, candidate {}",
            reference.steps.len(),
            candidate.steps.len()
        ));
    }
    let mut steps = Vec::new();
    for (expected, actual) in reference.steps.iter().zip(&candidate.steps) {
        let mut max_abs_difference = 0.0_f32;
        let mut worst_token_id = None;
        let mut missing_token_ids = Vec::new();
        let compared: Vec<&TopLogit> = expected.top.iter().take(profile.compared_top_k).collect();
        for entry in &compared {
            match actual.logit_of(entry.token_id) {
                Some(logit) => {
                    let difference = (logit - entry.logit).abs();
                    if difference > max_abs_difference || worst_token_id.is_none() {
                        max_abs_difference = max_abs_difference.max(difference);
                        worst_token_id = Some(entry.token_id);
                    }
                }
                None => missing_token_ids.push(entry.token_id),
            }
        }
        if expected.vocab_size != actual.vocab_size {
            failures.push(format!(
                "step {}: vocabulary size {} != {}",
                expected.step, actual.vocab_size, expected.vocab_size
            ));
        }
        if actual.non_finite_count > 0 {
            failures.push(format!(
                "step {}: candidate produced {} non-finite logits",
                expected.step, actual.non_finite_count
            ));
        }
        if profile.require_identical_tokens && expected.sampled_token_id != actual.sampled_token_id
        {
            failures.push(format!(
                "step {}: sampled token {} != reference {}",
                expected.step, actual.sampled_token_id, expected.sampled_token_id
            ));
        }
        if !missing_token_ids.is_empty() {
            failures.push(format!(
                "step {}: reference top tokens {:?} missing from the candidate's top {}",
                expected.step, missing_token_ids, SNAPSHOT_TOP_K
            ));
        }
        if max_abs_difference > profile.max_abs_logit_difference {
            failures.push(format!(
                "step {}: logit difference {max_abs_difference} exceeds {} (token {:?})",
                expected.step, profile.max_abs_logit_difference, worst_token_id
            ));
        }
        steps.push(StepComparison {
            step: expected.step,
            reference_token_id: expected.sampled_token_id,
            candidate_token_id: actual.sampled_token_id,
            bit_identical: expected.full_digest == actual.full_digest,
            compared_logits: compared.len() - missing_token_ids.len(),
            max_abs_difference,
            worst_token_id,
            missing_token_ids,
        });
    }
    LogitComparison {
        profile: profile.clone(),
        reference_provider: reference.provider.clone(),
        candidate_provider: candidate.provider.clone(),
        max_abs_difference: steps
            .iter()
            .map(|step| step.max_abs_difference)
            .fold(0.0, f32::max),
        steps,
        failures,
    }
}

#[cfg(test)]
#[path = "numerics_tests.rs"]
mod tests;
