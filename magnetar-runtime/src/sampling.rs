//! Sampling and logits processing contract.
//!
//! Sampling is the Runtime-owned boundary that consumes token logits or
//! equivalent next-token scores and returns a token ID. It is token based: it
//! does not decode text, advance KV cache, select Providers or Devices, or
//! grant processors filesystem, network, Git, secrets, workspace, or process
//! authority.

use crate::{
    CorrelationId, DeviceBinding, GenerationParameters, HostStagingPolicy,
    LogitsProcessorReference, MemoryAllocationClass, MemoryAllocationOwner,
    MemoryAllocationRequest, MemoryManager, MemoryPlacement, ProviderBinding, ResourceAffinity,
    SpecialTokenKind, TokenId, TokenizerMetadata,
};
use std::{
    collections::{BTreeMap, BTreeSet},
    error::Error,
    fmt,
};

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct SamplingRequestId(String);

impl SamplingRequestId {
    pub fn new(value: impl Into<String>) -> Result<Self, SamplingError> {
        let value = value.into();
        validate_identity(&value, "sampling request id")?;
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for SamplingRequestId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct SamplingRequest {
    pub request_id: SamplingRequestId,
    pub logits: Option<LogitsReference>,
    pub vocabulary_size: u32,
    pub step_index: usize,
    pub token_history: Vec<TokenId>,
    pub tokenizer: Option<TokenizerMetadata>,
    pub parameters: GenerationParameters,
    pub processors: Vec<LogitsProcessorConfig>,
    pub rng_seed: Option<u64>,
    pub rng_state: Option<SamplingRngState>,
    pub deterministic: bool,
    pub allowed_token_mask: Option<Vec<bool>>,
    pub banned_token_ids: Vec<TokenId>,
    pub stop: SamplingStopMetadata,
    pub policy: SamplingPolicy,
    pub correlation_id: Option<CorrelationId>,
}

impl SamplingRequest {
    pub fn host_scores(
        request_id: SamplingRequestId,
        scores: Vec<f32>,
        tokenizer: TokenizerMetadata,
    ) -> Self {
        Self {
            request_id,
            vocabulary_size: tokenizer.vocabulary_size,
            logits: Some(LogitsReference::HostScores(scores)),
            step_index: 0,
            token_history: Vec::new(),
            tokenizer: Some(tokenizer),
            parameters: GenerationParameters::default(),
            processors: Vec::new(),
            rng_seed: None,
            rng_state: None,
            deterministic: false,
            allowed_token_mask: None,
            banned_token_ids: Vec::new(),
            stop: SamplingStopMetadata::default(),
            policy: SamplingPolicy::default(),
            correlation_id: None,
        }
    }

    pub fn validate(&self) -> Result<(), SamplingError> {
        if self.vocabulary_size == 0 {
            return Err(SamplingError::LogitsInvalid {
                message: "vocabulary size must be greater than zero".into(),
            });
        }
        if let Some(tokenizer) = &self.tokenizer {
            tokenizer
                .validate()
                .map_err(|error| SamplingError::TokenizerMetadataMissing {
                    message: error.to_string(),
                })?;
            if tokenizer.vocabulary_size != self.vocabulary_size {
                return Err(SamplingError::VocabularyMismatch {
                    expected: tokenizer.vocabulary_size,
                    actual: self.vocabulary_size,
                });
            }
        } else {
            return Err(SamplingError::TokenizerMetadataMissing {
                message: "sampling requires tokenizer metadata".into(),
            });
        }
        validate_logits_shape(self.logits.as_ref(), self.vocabulary_size)?;
        self.parameters.validate().map_err(SamplingError::from)?;
        validate_optional_probability(
            "top-p",
            self.parameters.top_p,
            SamplingErrorKind::TopPInvalid,
        )?;
        validate_optional_probability(
            "min-p",
            self.parameters.min_p,
            SamplingErrorKind::MinPUnsupported,
        )?;
        validate_optional_probability(
            "typical-p",
            self.parameters.typical_p,
            SamplingErrorKind::TypicalPUnsupported,
        )?;
        if self.parameters.top_k == Some(0) {
            return Err(SamplingError::TopKInvalid {
                message: "top-k must be greater than zero".into(),
            });
        }
        if self.parameters.min_p.is_some() {
            return Err(SamplingError::MinPUnsupported);
        }
        if self.parameters.typical_p.is_some() {
            return Err(SamplingError::TypicalPUnsupported);
        }
        for token_id in self
            .banned_token_ids
            .iter()
            .chain(self.parameters.banned_token_ids.iter())
            .chain(self.parameters.allowed_token_ids.iter().flatten())
            .chain(self.stop.stop_token_ids.iter())
        {
            validate_token_id(*token_id, self)?;
        }
        if let Some(mask) = &self.allowed_token_mask
            && mask.len() != self.vocabulary_size as usize
        {
            return Err(SamplingError::AllowedTokenInvalid {
                token_id: self.vocabulary_size,
            });
        }
        if self.deterministic
            && self.rng_seed.is_none()
            && self.rng_state.is_none()
            && !self.parameters.greedy
            && self.parameters.sampling_enabled
        {
            return Err(SamplingError::DeterministicModeUnsupported {
                message: "deterministic stochastic sampling requires a seed or runtime RNG state"
                    .into(),
            });
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum LogitsReference {
    RuntimeTensor {
        id: String,
    },
    ProviderOwnedTensor {
        id: String,
        provider: ProviderBinding,
        affinity: Option<ResourceAffinity>,
    },
    DeviceResidentTensor {
        id: String,
        device: DeviceBinding,
        affinity: Option<ResourceAffinity>,
    },
    HostScores(Vec<f32>),
    TestFixture(Vec<f32>),
}

impl LogitsReference {
    pub fn host_scores(&self, policy: &SamplingPolicy) -> Result<&[f32], SamplingError> {
        match self {
            Self::HostScores(scores) | Self::TestFixture(scores) => Ok(scores),
            Self::RuntimeTensor { .. }
            | Self::ProviderOwnedTensor { .. }
            | Self::DeviceResidentTensor { .. } => {
                if policy.allow_logits_materialization {
                    Err(SamplingError::LogitsMaterializationFailed {
                        message: "no materializer is attached to this sampling contract fixture"
                            .into(),
                    })
                } else {
                    Err(SamplingError::LogitsMaterializationDenied)
                }
            }
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LogitsProcessorConfig {
    pub id: String,
    pub kind: LogitsProcessorKind,
    pub authority: LogitsProcessorAuthority,
}

impl LogitsProcessorConfig {
    pub fn runtime(kind: LogitsProcessorKind) -> Self {
        Self {
            id: format!("{kind:?}"),
            kind,
            authority: LogitsProcessorAuthority::inference_scoped(),
        }
    }

    pub fn from_generation(reference: &LogitsProcessorReference) -> Self {
        Self {
            id: reference.id.clone(),
            kind: LogitsProcessorKind::Custom,
            authority: LogitsProcessorAuthority::inference_scoped(),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum LogitsProcessorKind {
    InvalidTokenMask,
    VocabularyRangeMask,
    SpecialTokenMask,
    BannedTokenMask,
    AllowedTokenMask,
    RepetitionPenalty,
    FrequencyPenalty,
    PresencePenalty,
    Temperature,
    TopK,
    TopP,
    MinP,
    TypicalP,
    StopTokenMask,
    PolicyFilter,
    Custom,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct LogitsProcessorAuthority {
    pub filesystem: bool,
    pub network: bool,
    pub git: bool,
    pub secrets: bool,
    pub workspace: bool,
    pub process: bool,
}

impl LogitsProcessorAuthority {
    pub fn inference_scoped() -> Self {
        Self::default()
    }

    pub const fn has_external_authority(&self) -> bool {
        self.filesystem
            || self.network
            || self.git
            || self.secrets
            || self.workspace
            || self.process
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TemperatureZeroPolicy {
    Greedy,
    Invalid,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SamplingPolicy {
    pub allow_probability_metadata: bool,
    pub allow_logits_materialization: bool,
    pub host_staging: HostStagingPolicy,
    pub temperature_zero: TemperatureZeroPolicy,
    pub allow_provider_assisted: bool,
    pub allow_eos_before_min_length: bool,
    pub suppress_pad: bool,
    pub suppress_bos_after_beginning: bool,
    pub suppress_unknown: bool,
    pub suppress_additional_special_tokens: bool,
}

impl Default for SamplingPolicy {
    fn default() -> Self {
        Self {
            allow_probability_metadata: false,
            allow_logits_materialization: false,
            host_staging: HostStagingPolicy::Forbid,
            temperature_zero: TemperatureZeroPolicy::Greedy,
            allow_provider_assisted: false,
            allow_eos_before_min_length: false,
            suppress_pad: true,
            suppress_bos_after_beginning: true,
            suppress_unknown: true,
            suppress_additional_special_tokens: true,
        }
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct SamplingStopMetadata {
    pub eos_token_ids: Vec<TokenId>,
    pub stop_token_ids: Vec<TokenId>,
    pub minimum_generated_tokens: Option<usize>,
    pub mask_stop_tokens_before_minimum: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SamplingRngState {
    bytes: Vec<u8>,
    inspectable: bool,
}

impl SamplingRngState {
    pub fn opaque(bytes: Vec<u8>) -> Self {
        Self {
            bytes,
            inspectable: false,
        }
    }

    pub fn inspectable(bytes: Vec<u8>) -> Self {
        Self {
            bytes,
            inspectable: true,
        }
    }

    pub fn expose(&self) -> Option<&[u8]> {
        self.inspectable.then_some(&self.bytes)
    }

    /// Encodes a stream position produced by this module.
    fn from_stream(state: u64) -> Self {
        Self::opaque(state.to_le_bytes().to_vec())
    }

    /// Recovers the stream position.
    ///
    /// States this module produced are exactly eight little-endian bytes and
    /// round-trip losslessly, which is what lets a caller resume a stream
    /// rather than restart it. A state of any other length came from
    /// somewhere else, so it is folded down to a starting position instead.
    fn stream_position(&self) -> u64 {
        match <[u8; 8]>::try_from(self.bytes.as_slice()) {
            Ok(bytes) => u64::from_le_bytes(bytes),
            Err(_) => self.bytes.iter().fold(0xcbf29ce484222325, |hash, byte| {
                (hash ^ u64::from(*byte)).wrapping_mul(0x100000001b3)
            }),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct SamplingResult {
    pub request_id: SamplingRequestId,
    pub selected_token_id: TokenId,
    pub selection_mode: SamplingSelectionMode,
    pub token_rank: Option<usize>,
    pub token_probability: Option<f32>,
    pub token_log_probability: Option<f32>,
    pub finish_hint: Option<SamplingFinishHint>,
    pub diagnostics: Vec<SamplingDiagnostic>,
    pub updated_rng_state: Option<SamplingRngState>,
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum SamplingSelectionMode {
    Greedy,
    Stochastic,
    ProviderAssisted,
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum SamplingFinishHint {
    EosCandidate,
    StopTokenCandidate,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SamplingDiagnostic {
    pub kind: SamplingDiagnosticKind,
    pub message: String,
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum SamplingDiagnosticKind {
    ProcessorApplied,
    DeterministicSeedUsed,
    ProbabilityMetadataRequested,
    LogitsMaterializationRequested,
    ProviderAssistedSamplingUsed,
    Nondeterminism,
    RedactedLogits,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SamplingObservation {
    pub request_id: SamplingRequestId,
    pub kind: SamplingObservationKind,
    pub step_index: usize,
    pub selected_token_id: Option<TokenId>,
    pub error: Option<SamplingErrorKind>,
    pub correlation_id: Option<CorrelationId>,
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum SamplingObservationKind {
    SamplingRequested,
    ProcessorChainBuilt,
    ProcessorApplied,
    TokenSelected,
    SamplingFailed,
    NoEligibleToken,
    DeterministicSeedUsed,
    ProbabilityMetadataRequested,
    LogitsMaterializationRequested,
    LogitsMaterializationDenied,
    ProviderAssistedSamplingUsed,
    MemoryAllocationFailed,
    PolicyDenied,
}

pub fn sampling_observation(
    request: &SamplingRequest,
    kind: SamplingObservationKind,
) -> SamplingObservation {
    SamplingObservation {
        request_id: request.request_id.clone(),
        kind,
        step_index: request.step_index,
        selected_token_id: None,
        error: None,
        correlation_id: request.correlation_id.clone(),
    }
}

pub fn select_next_token(request: &SamplingRequest) -> Result<SamplingResult, SamplingError> {
    request.validate()?;
    for processor in &request.processors {
        if processor.authority.has_external_authority() {
            return Err(SamplingError::ProcessorUnsupported {
                message: format!("processor '{}' requests external authority", processor.id),
            });
        }
    }

    let scores = request
        .logits
        .as_ref()
        .ok_or(SamplingError::LogitsUnavailable)?
        .host_scores(&request.policy)?;
    let tokenizer =
        request
            .tokenizer
            .as_ref()
            .ok_or_else(|| SamplingError::TokenizerMetadataMissing {
                message: "sampling requires tokenizer metadata".into(),
            })?;
    let mut candidates = scores
        .iter()
        .enumerate()
        .map(|(index, score)| {
            let token_id = tokenizer.token_id_range.start.saturating_add(index as u32);
            Candidate {
                token_id,
                score: *score,
                eligible: score.is_finite(),
            }
        })
        .collect::<Vec<_>>();

    apply_token_constraints(request, &mut candidates)?;
    apply_penalties(request, &mut candidates)?;
    apply_temperature(request, &mut candidates)?;
    apply_top_k(request, &mut candidates);
    apply_top_p(request, &mut candidates)?;

    let eligible = candidates
        .iter()
        .filter(|candidate| candidate.eligible)
        .collect::<Vec<_>>();
    if eligible.is_empty() {
        return Err(SamplingError::NoEligibleToken);
    }

    let probabilities = softmax(&eligible);
    let mut stream_state = 0_u64;
    let selection_mode = if request.parameters.greedy
        || !request.parameters.sampling_enabled
        || request.parameters.temperature == 0.0
    {
        SamplingSelectionMode::Greedy
    } else {
        SamplingSelectionMode::Stochastic
    };
    let selected_index = match selection_mode {
        SamplingSelectionMode::Greedy | SamplingSelectionMode::ProviderAssisted => probabilities
            .iter()
            .enumerate()
            .max_by(|(_, left), (_, right)| compare_f32(left.score, right.score))
            .map(|(index, _)| index)
            .expect("eligible set is non-empty"),
        SamplingSelectionMode::Stochastic => {
            stream_state = resolve_stream_state(request, &probabilities);
            sample_index(&probabilities, &mut stream_state)
        }
    };
    let selected = &probabilities[selected_index];
    let rank = rank_for(selected.token_id, &probabilities);
    let probability_allowed = request.policy.allow_probability_metadata;
    let updated_rng_state = matches!(selection_mode, SamplingSelectionMode::Stochastic)
        .then(|| SamplingRngState::from_stream(stream_state));

    Ok(SamplingResult {
        request_id: request.request_id.clone(),
        selected_token_id: selected.token_id,
        selection_mode,
        token_rank: Some(rank),
        token_probability: probability_allowed.then_some(selected.probability),
        token_log_probability: probability_allowed.then_some(selected.probability.ln()),
        finish_hint: finish_hint_for(request, selected.token_id),
        diagnostics: diagnostics_for(request, selection_mode),
        updated_rng_state,
    })
}

pub fn processor_order(parameters: &GenerationParameters) -> Vec<LogitsProcessorKind> {
    let mut order = vec![
        LogitsProcessorKind::InvalidTokenMask,
        LogitsProcessorKind::VocabularyRangeMask,
        LogitsProcessorKind::SpecialTokenMask,
        LogitsProcessorKind::BannedTokenMask,
        LogitsProcessorKind::AllowedTokenMask,
    ];
    if parameters.repetition_penalty.is_some() {
        order.push(LogitsProcessorKind::RepetitionPenalty);
    }
    if parameters.frequency_penalty.is_some() {
        order.push(LogitsProcessorKind::FrequencyPenalty);
    }
    if parameters.presence_penalty.is_some() {
        order.push(LogitsProcessorKind::PresencePenalty);
    }
    order.push(LogitsProcessorKind::Temperature);
    if parameters.top_k.is_some() {
        order.push(LogitsProcessorKind::TopK);
    }
    if parameters.top_p.is_some() {
        order.push(LogitsProcessorKind::TopP);
    }
    if parameters.min_p.is_some() {
        order.push(LogitsProcessorKind::MinP);
    }
    if parameters.typical_p.is_some() {
        order.push(LogitsProcessorKind::TypicalP);
    }
    order.push(LogitsProcessorKind::StopTokenMask);
    order.push(LogitsProcessorKind::PolicyFilter);
    order
}

pub fn sampling_workspace_requests(
    request: &SamplingRequest,
    manager: &MemoryManager,
) -> Result<Vec<MemoryAllocationRequest>, SamplingError> {
    let vocab = u64::from(request.vocabulary_size);
    let mut requests = Vec::new();
    for (label, bytes) in [
        ("logits", vocab.checked_mul(4)),
        ("probabilities", vocab.checked_mul(4)),
        ("mask", Some(vocab)),
        ("sorted-tokens", vocab.checked_mul(8)),
        ("top-k", request.parameters.top_k.map(|k| u64::from(k) * 8)),
        (
            "top-p",
            request.parameters.top_p.map(|_| vocab.saturating_mul(8)),
        ),
        (
            "rng-state",
            (request.rng_seed.is_some() || request.rng_state.is_some()).then_some(32),
        ),
        (
            "history",
            (!request.token_history.is_empty()).then_some(request.token_history.len() as u64 * 4),
        ),
        (
            "penalty",
            penalties_enabled(&request.parameters).then_some(vocab.saturating_mul(4)),
        ),
    ] {
        if let Some(bytes) = bytes {
            let allocation = MemoryAllocationRequest::new(
                MemoryAllocationClass::TemporaryWorkspace,
                bytes,
                MemoryPlacement::HostOrdinary,
                MemoryAllocationOwner::Runtime,
            );
            let feasibility = manager.feasibility(&allocation);
            if !feasibility.feasible {
                return Err(SamplingError::MemoryAllocationFailed {
                    message: format!("sampling {label} buffer was denied by memory policy"),
                });
            }
            requests.push(allocation);
        }
    }
    Ok(requests)
}

#[derive(Clone, Copy, Debug)]
struct Candidate {
    token_id: TokenId,
    score: f32,
    eligible: bool,
}

#[derive(Clone, Copy, Debug)]
struct ProbabilityCandidate {
    token_id: TokenId,
    score: f32,
    probability: f32,
}

fn validate_logits_shape(
    logits: Option<&LogitsReference>,
    vocabulary_size: u32,
) -> Result<(), SamplingError> {
    let logits = logits.ok_or(SamplingError::LogitsUnavailable)?;
    match logits {
        LogitsReference::HostScores(scores) | LogitsReference::TestFixture(scores) => {
            if scores.len() != vocabulary_size as usize {
                return Err(SamplingError::VocabularyMismatch {
                    expected: vocabulary_size,
                    actual: scores.len() as u32,
                });
            }
            if scores.iter().all(|score| !score.is_finite()) {
                return Err(SamplingError::LogitsInvalid {
                    message: "at least one logit must be finite".into(),
                });
            }
        }
        LogitsReference::RuntimeTensor { id }
        | LogitsReference::ProviderOwnedTensor { id, .. }
        | LogitsReference::DeviceResidentTensor { id, .. } => {
            validate_identity(id, "logits reference")?;
        }
    }
    Ok(())
}

fn validate_optional_probability(
    name: &'static str,
    value: Option<f32>,
    unsupported_kind: SamplingErrorKind,
) -> Result<(), SamplingError> {
    if let Some(value) = value
        && (!value.is_finite() || value <= 0.0 || value > 1.0)
    {
        return match unsupported_kind {
            SamplingErrorKind::TopPInvalid => Err(SamplingError::TopPInvalid {
                message: format!("{name} must be within (0, 1]"),
            }),
            SamplingErrorKind::MinPUnsupported => Err(SamplingError::MinPUnsupported),
            SamplingErrorKind::TypicalPUnsupported => Err(SamplingError::TypicalPUnsupported),
            _ => Err(SamplingError::InvalidSamplingParameter {
                parameter: name,
                message: "probability parameter is invalid".into(),
            }),
        };
    }
    Ok(())
}

fn validate_token_id(token_id: TokenId, request: &SamplingRequest) -> Result<(), SamplingError> {
    let Some(tokenizer) = &request.tokenizer else {
        return Err(SamplingError::TokenizerMetadataMissing {
            message: "sampling requires tokenizer metadata".into(),
        });
    };
    let end = tokenizer
        .token_id_range
        .start
        .saturating_add(request.vocabulary_size.saturating_sub(1));
    if token_id < tokenizer.token_id_range.start || token_id > end {
        return Err(SamplingError::InvalidTokenId { token_id });
    }
    Ok(())
}

fn apply_token_constraints(
    request: &SamplingRequest,
    candidates: &mut [Candidate],
) -> Result<(), SamplingError> {
    let tokenizer =
        request
            .tokenizer
            .as_ref()
            .ok_or_else(|| SamplingError::TokenizerMetadataMissing {
                message: "sampling requires tokenizer metadata".into(),
            })?;
    let special = tokenizer
        .special_tokens
        .iter()
        .chain(tokenizer.additional_special_tokens.iter())
        .map(|token| (token.id, token.kind))
        .collect::<BTreeMap<_, _>>();
    let banned = request
        .banned_token_ids
        .iter()
        .chain(request.parameters.banned_token_ids.iter())
        .copied()
        .collect::<BTreeSet<_>>();
    let allowed = request
        .parameters
        .allowed_token_ids
        .as_ref()
        .map(|ids| ids.iter().copied().collect::<BTreeSet<_>>());
    let minimum_active = request
        .stop
        .minimum_generated_tokens
        .is_some_and(|minimum| request.step_index < minimum);

    for (index, candidate) in candidates.iter_mut().enumerate() {
        if let Some(kind) = special.get(&candidate.token_id).copied() {
            match kind {
                SpecialTokenKind::Pad if request.policy.suppress_pad => candidate.eligible = false,
                SpecialTokenKind::Unknown if request.policy.suppress_unknown => {
                    candidate.eligible = false
                }
                SpecialTokenKind::Bos
                    if request.policy.suppress_bos_after_beginning && request.step_index > 0 =>
                {
                    candidate.eligible = false
                }
                SpecialTokenKind::Additional
                    if request.policy.suppress_additional_special_tokens =>
                {
                    candidate.eligible = false
                }
                SpecialTokenKind::Eos
                    if minimum_active && !request.policy.allow_eos_before_min_length =>
                {
                    candidate.eligible = false
                }
                _ => {}
            }
        }
        if request.stop.mask_stop_tokens_before_minimum
            && minimum_active
            && request.stop.stop_token_ids.contains(&candidate.token_id)
        {
            candidate.eligible = false;
        }
        if let Some(mask) = &request.allowed_token_mask
            && !mask[index]
        {
            candidate.eligible = false;
        }
        if let Some(allowed) = &allowed
            && !allowed.contains(&candidate.token_id)
        {
            candidate.eligible = false;
        }
        if banned.contains(&candidate.token_id) {
            candidate.eligible = false;
        }
    }
    Ok(())
}

fn apply_penalties(
    request: &SamplingRequest,
    candidates: &mut [Candidate],
) -> Result<(), SamplingError> {
    if !penalties_enabled(&request.parameters) {
        return Ok(());
    }
    let counts = request.token_history.iter().fold(
        BTreeMap::<TokenId, u32>::new(),
        |mut counts, token_id| {
            *counts.entry(*token_id).or_default() += 1;
            counts
        },
    );
    for candidate in candidates {
        let count = counts.get(&candidate.token_id).copied().unwrap_or_default() as f32;
        if count == 0.0 {
            continue;
        }
        if let Some(penalty) = request.parameters.repetition_penalty {
            if penalty < 0.0 || !penalty.is_finite() {
                return Err(SamplingError::RepetitionPenaltyInvalid);
            }
            if penalty > 0.0 {
                candidate.score -= penalty;
            }
        }
        if let Some(penalty) = request.parameters.frequency_penalty {
            if penalty < 0.0 || !penalty.is_finite() {
                return Err(SamplingError::FrequencyPenaltyInvalid);
            }
            candidate.score -= penalty * count;
        }
        if let Some(penalty) = request.parameters.presence_penalty {
            if penalty < 0.0 || !penalty.is_finite() {
                return Err(SamplingError::PresencePenaltyInvalid);
            }
            candidate.score -= penalty;
        }
    }
    Ok(())
}

fn apply_temperature(
    request: &SamplingRequest,
    candidates: &mut [Candidate],
) -> Result<(), SamplingError> {
    let temperature = request.parameters.temperature;
    if !temperature.is_finite() || temperature < 0.0 {
        return Err(SamplingError::TemperatureInvalid {
            message: "temperature must be finite and non-negative".into(),
        });
    }
    if temperature == 0.0 {
        if request.policy.temperature_zero == TemperatureZeroPolicy::Invalid {
            return Err(SamplingError::TemperatureInvalid {
                message: "temperature zero is invalid by policy".into(),
            });
        }
        return Ok(());
    }
    for candidate in candidates.iter_mut().filter(|candidate| candidate.eligible) {
        candidate.score /= temperature;
    }
    Ok(())
}

fn apply_top_k(request: &SamplingRequest, candidates: &mut [Candidate]) {
    let Some(k) = request.parameters.top_k else {
        return;
    };
    let mut eligible = candidates
        .iter()
        .filter(|candidate| candidate.eligible)
        .copied()
        .collect::<Vec<_>>();
    if k as usize >= eligible.len() {
        return;
    }
    eligible.sort_by(|left, right| {
        compare_f32(right.score, left.score).then(left.token_id.cmp(&right.token_id))
    });
    let keep = eligible
        .into_iter()
        .take(k as usize)
        .map(|candidate| candidate.token_id)
        .collect::<BTreeSet<_>>();
    for candidate in candidates {
        if candidate.eligible && !keep.contains(&candidate.token_id) {
            candidate.eligible = false;
        }
    }
}

fn apply_top_p(
    request: &SamplingRequest,
    candidates: &mut [Candidate],
) -> Result<(), SamplingError> {
    let Some(top_p) = request.parameters.top_p else {
        return Ok(());
    };
    if !top_p.is_finite() || top_p <= 0.0 || top_p > 1.0 {
        return Err(SamplingError::TopPInvalid {
            message: "top-p must be within (0, 1]".into(),
        });
    }
    let mut eligible = candidates
        .iter()
        .filter(|candidate| candidate.eligible)
        .copied()
        .collect::<Vec<_>>();
    eligible.sort_by(|left, right| {
        compare_f32(right.score, left.score).then(left.token_id.cmp(&right.token_id))
    });
    let probabilities = softmax(&eligible.iter().collect::<Vec<_>>());
    let mut cumulative = 0.0;
    let mut keep = BTreeSet::new();
    for candidate in probabilities {
        cumulative += candidate.probability;
        keep.insert(candidate.token_id);
        if cumulative >= top_p {
            break;
        }
    }
    for candidate in candidates {
        if candidate.eligible && !keep.contains(&candidate.token_id) {
            candidate.eligible = false;
        }
    }
    Ok(())
}

fn softmax(candidates: &[&Candidate]) -> Vec<ProbabilityCandidate> {
    let max = candidates
        .iter()
        .map(|candidate| candidate.score)
        .fold(f32::NEG_INFINITY, f32::max);
    let denom = candidates
        .iter()
        .map(|candidate| (candidate.score - max).exp())
        .sum::<f32>();
    candidates
        .iter()
        .map(|candidate| ProbabilityCandidate {
            token_id: candidate.token_id,
            score: candidate.score,
            probability: (candidate.score - max).exp() / denom,
        })
        .collect()
}

fn rank_for(token_id: TokenId, probabilities: &[ProbabilityCandidate]) -> usize {
    let mut ranked = probabilities.to_vec();
    ranked.sort_by(|left, right| {
        compare_f32(right.score, left.score).then(left.token_id.cmp(&right.token_id))
    });
    ranked
        .iter()
        .position(|candidate| candidate.token_id == token_id)
        .map(|index| index + 1)
        .unwrap_or(1)
}

/// Chooses the stream position this step draws from.
///
/// A threaded [`SamplingRngState`] resumes exactly where the previous step
/// left off. Callers that only set a fixed `rng_seed` and never thread the
/// state back get a position derived from that seed *and* the step index, so
/// the stream still advances across a generation instead of redrawing the same
/// number every step.
fn resolve_stream_state(request: &SamplingRequest, probabilities: &[ProbabilityCandidate]) -> u64 {
    if let Some(state) = request.rng_state.as_ref() {
        return state.stream_position();
    }
    let origin = request
        .rng_seed
        .unwrap_or_else(|| deterministic_seed_from_scores(probabilities));
    // Distinct steps must start at distinct positions; mixing the counter
    // through the finalizer keeps consecutive steps uncorrelated.
    let mut state = origin ^ (request.step_index as u64).wrapping_mul(0x9e3779b97f4a7c15);
    splitmix64(&mut state);
    state
}

fn sample_index(candidates: &[ProbabilityCandidate], state: &mut u64) -> usize {
    let threshold = unit_interval(splitmix64(state));
    let mut cumulative = 0.0;
    for (index, candidate) in candidates.iter().enumerate() {
        cumulative += candidate.probability;
        if threshold < cumulative {
            return index;
        }
    }
    candidates.len().saturating_sub(1)
}

fn deterministic_seed_from_scores(candidates: &[ProbabilityCandidate]) -> u64 {
    candidates
        .iter()
        .fold(0x9e3779b97f4a7c15, |seed, candidate| {
            seed ^ (u64::from(candidate.token_id) << 32) ^ u64::from(candidate.score.to_bits())
        })
}

/// SplitMix64: advances `state` by the golden-gamma increment and returns the
/// finalized output. Chosen over the previous single-round xorshift because a
/// counter plus a strong finalizer decorrelates adjacent seeds, which is
/// exactly the property a per-step sampling stream needs.
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e3779b97f4a7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    z ^ (z >> 31)
}

/// Maps a draw onto `[0, 1)` using the top 53 bits, the exactly
/// representable range of an `f64` mantissa.
fn unit_interval(draw: u64) -> f32 {
    ((draw >> 11) as f64 / (1_u64 << 53) as f64) as f32
}

fn finish_hint_for(request: &SamplingRequest, token_id: TokenId) -> Option<SamplingFinishHint> {
    if request.stop.eos_token_ids.contains(&token_id) {
        Some(SamplingFinishHint::EosCandidate)
    } else if request.stop.stop_token_ids.contains(&token_id) {
        Some(SamplingFinishHint::StopTokenCandidate)
    } else {
        None
    }
}

fn diagnostics_for(
    request: &SamplingRequest,
    selection_mode: SamplingSelectionMode,
) -> Vec<SamplingDiagnostic> {
    let mut diagnostics = vec![SamplingDiagnostic {
        kind: SamplingDiagnosticKind::RedactedLogits,
        message: "raw logits are not logged by default".into(),
    }];
    if request.policy.allow_probability_metadata {
        diagnostics.push(SamplingDiagnostic {
            kind: SamplingDiagnosticKind::ProbabilityMetadataRequested,
            message: "probability metadata requested and allowed by policy".into(),
        });
    }
    if request.rng_seed.is_some() {
        diagnostics.push(SamplingDiagnostic {
            kind: SamplingDiagnosticKind::DeterministicSeedUsed,
            message: "runtime-owned deterministic seed used".into(),
        });
    }
    if selection_mode == SamplingSelectionMode::Stochastic && request.rng_seed.is_none() {
        diagnostics.push(SamplingDiagnostic {
            kind: SamplingDiagnosticKind::Nondeterminism,
            message: "stochastic sampling used runtime-derived seed".into(),
        });
    }
    diagnostics
}

fn compare_f32(left: f32, right: f32) -> std::cmp::Ordering {
    left.partial_cmp(&right).unwrap_or(std::cmp::Ordering::Less)
}

fn penalties_enabled(parameters: &GenerationParameters) -> bool {
    parameters.repetition_penalty.is_some()
        || parameters.frequency_penalty.is_some()
        || parameters.presence_penalty.is_some()
}

fn validate_identity(value: &str, label: &'static str) -> Result<(), SamplingError> {
    if value.trim().is_empty()
        || value.contains('/')
        || value.contains('\\')
        || value.contains(':')
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'-' | b'_'))
    {
        return Err(SamplingError::InvalidSamplingParameter {
            parameter: label,
            message: "value must use a portable identifier".into(),
        });
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum SamplingErrorKind {
    LogitsUnavailable,
    LogitsInvalid,
    VocabularyMismatch,
    InvalidTokenId,
    InvalidSamplingParameter,
    TemperatureInvalid,
    TopKInvalid,
    TopPInvalid,
    MinPUnsupported,
    TypicalPUnsupported,
    RepetitionPenaltyInvalid,
    FrequencyPenaltyInvalid,
    PresencePenaltyInvalid,
    BannedTokenInvalid,
    AllowedTokenInvalid,
    NoEligibleToken,
    DeterministicModeUnsupported,
    RngUnavailable,
    ProbabilityMetadataUnsupported,
    LogitsMaterializationDenied,
    LogitsMaterializationFailed,
    MemoryAllocationFailed,
    ProviderAssistedSamplingUnavailable,
    ProviderExecutionFailed,
    ResourceAffinityConflict,
    TokenizerMetadataMissing,
    ProcessorUnsupported,
    ProcessorFailed,
    PolicyDenied,
    BrowserFeatureUnsupported,
    InternalSampling,
}

#[derive(Clone, Debug, PartialEq)]
pub enum SamplingError {
    LogitsUnavailable,
    LogitsInvalid {
        message: String,
    },
    VocabularyMismatch {
        expected: u32,
        actual: u32,
    },
    InvalidTokenId {
        token_id: TokenId,
    },
    InvalidSamplingParameter {
        parameter: &'static str,
        message: String,
    },
    TemperatureInvalid {
        message: String,
    },
    TopKInvalid {
        message: String,
    },
    TopPInvalid {
        message: String,
    },
    MinPUnsupported,
    TypicalPUnsupported,
    RepetitionPenaltyInvalid,
    FrequencyPenaltyInvalid,
    PresencePenaltyInvalid,
    BannedTokenInvalid {
        token_id: TokenId,
    },
    AllowedTokenInvalid {
        token_id: TokenId,
    },
    NoEligibleToken,
    DeterministicModeUnsupported {
        message: String,
    },
    RngUnavailable,
    ProbabilityMetadataUnsupported,
    LogitsMaterializationDenied,
    LogitsMaterializationFailed {
        message: String,
    },
    MemoryAllocationFailed {
        message: String,
    },
    ProviderAssistedSamplingUnavailable,
    ProviderExecutionFailed {
        message: String,
    },
    ResourceAffinityConflict {
        message: String,
    },
    TokenizerMetadataMissing {
        message: String,
    },
    ProcessorUnsupported {
        message: String,
    },
    ProcessorFailed {
        message: String,
    },
    PolicyDenied {
        message: String,
    },
    BrowserFeatureUnsupported {
        message: String,
    },
    InternalSampling {
        message: String,
    },
}

impl SamplingError {
    pub const fn kind(&self) -> SamplingErrorKind {
        match self {
            Self::LogitsUnavailable => SamplingErrorKind::LogitsUnavailable,
            Self::LogitsInvalid { .. } => SamplingErrorKind::LogitsInvalid,
            Self::VocabularyMismatch { .. } => SamplingErrorKind::VocabularyMismatch,
            Self::InvalidTokenId { .. } => SamplingErrorKind::InvalidTokenId,
            Self::InvalidSamplingParameter { .. } => SamplingErrorKind::InvalidSamplingParameter,
            Self::TemperatureInvalid { .. } => SamplingErrorKind::TemperatureInvalid,
            Self::TopKInvalid { .. } => SamplingErrorKind::TopKInvalid,
            Self::TopPInvalid { .. } => SamplingErrorKind::TopPInvalid,
            Self::MinPUnsupported => SamplingErrorKind::MinPUnsupported,
            Self::TypicalPUnsupported => SamplingErrorKind::TypicalPUnsupported,
            Self::RepetitionPenaltyInvalid => SamplingErrorKind::RepetitionPenaltyInvalid,
            Self::FrequencyPenaltyInvalid => SamplingErrorKind::FrequencyPenaltyInvalid,
            Self::PresencePenaltyInvalid => SamplingErrorKind::PresencePenaltyInvalid,
            Self::BannedTokenInvalid { .. } => SamplingErrorKind::BannedTokenInvalid,
            Self::AllowedTokenInvalid { .. } => SamplingErrorKind::AllowedTokenInvalid,
            Self::NoEligibleToken => SamplingErrorKind::NoEligibleToken,
            Self::DeterministicModeUnsupported { .. } => {
                SamplingErrorKind::DeterministicModeUnsupported
            }
            Self::RngUnavailable => SamplingErrorKind::RngUnavailable,
            Self::ProbabilityMetadataUnsupported => {
                SamplingErrorKind::ProbabilityMetadataUnsupported
            }
            Self::LogitsMaterializationDenied => SamplingErrorKind::LogitsMaterializationDenied,
            Self::LogitsMaterializationFailed { .. } => {
                SamplingErrorKind::LogitsMaterializationFailed
            }
            Self::MemoryAllocationFailed { .. } => SamplingErrorKind::MemoryAllocationFailed,
            Self::ProviderAssistedSamplingUnavailable => {
                SamplingErrorKind::ProviderAssistedSamplingUnavailable
            }
            Self::ProviderExecutionFailed { .. } => SamplingErrorKind::ProviderExecutionFailed,
            Self::ResourceAffinityConflict { .. } => SamplingErrorKind::ResourceAffinityConflict,
            Self::TokenizerMetadataMissing { .. } => SamplingErrorKind::TokenizerMetadataMissing,
            Self::ProcessorUnsupported { .. } => SamplingErrorKind::ProcessorUnsupported,
            Self::ProcessorFailed { .. } => SamplingErrorKind::ProcessorFailed,
            Self::PolicyDenied { .. } => SamplingErrorKind::PolicyDenied,
            Self::BrowserFeatureUnsupported { .. } => SamplingErrorKind::BrowserFeatureUnsupported,
            Self::InternalSampling { .. } => SamplingErrorKind::InternalSampling,
        }
    }
}

impl fmt::Display for SamplingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LogitsUnavailable => f.write_str("logits unavailable"),
            Self::LogitsInvalid { message } => write!(f, "logits invalid: {message}"),
            Self::VocabularyMismatch { expected, actual } => {
                write!(
                    f,
                    "vocabulary mismatch: expected {expected}, actual {actual}"
                )
            }
            Self::InvalidTokenId { token_id } => write!(f, "invalid token id {token_id}"),
            Self::InvalidSamplingParameter { parameter, message } => {
                write!(f, "sampling parameter '{parameter}' invalid: {message}")
            }
            Self::TemperatureInvalid { message } => write!(f, "temperature invalid: {message}"),
            Self::TopKInvalid { message } => write!(f, "top-k invalid: {message}"),
            Self::TopPInvalid { message } => write!(f, "top-p invalid: {message}"),
            Self::MinPUnsupported => f.write_str("min-p unsupported"),
            Self::TypicalPUnsupported => f.write_str("typical-p unsupported"),
            Self::RepetitionPenaltyInvalid => f.write_str("repetition penalty invalid"),
            Self::FrequencyPenaltyInvalid => f.write_str("frequency penalty invalid"),
            Self::PresencePenaltyInvalid => f.write_str("presence penalty invalid"),
            Self::BannedTokenInvalid { token_id } => write!(f, "banned token invalid: {token_id}"),
            Self::AllowedTokenInvalid { token_id } => {
                write!(f, "allowed token invalid: {token_id}")
            }
            Self::NoEligibleToken => f.write_str("no eligible token"),
            Self::DeterministicModeUnsupported { message } => {
                write!(f, "deterministic mode unsupported: {message}")
            }
            Self::RngUnavailable => f.write_str("rng unavailable"),
            Self::ProbabilityMetadataUnsupported => f.write_str("probability metadata unsupported"),
            Self::LogitsMaterializationDenied => f.write_str("logits materialization denied"),
            Self::LogitsMaterializationFailed { message } => {
                write!(f, "logits materialization failed: {message}")
            }
            Self::MemoryAllocationFailed { message } => {
                write!(f, "memory allocation failed: {message}")
            }
            Self::ProviderAssistedSamplingUnavailable => {
                f.write_str("provider-assisted sampling unavailable")
            }
            Self::ProviderExecutionFailed { message } => {
                write!(f, "provider execution failed: {message}")
            }
            Self::ResourceAffinityConflict { message } => {
                write!(f, "resource affinity conflict: {message}")
            }
            Self::TokenizerMetadataMissing { message } => {
                write!(f, "tokenizer metadata missing: {message}")
            }
            Self::ProcessorUnsupported { message } => write!(f, "processor unsupported: {message}"),
            Self::ProcessorFailed { message } => write!(f, "processor failed: {message}"),
            Self::PolicyDenied { message } => write!(f, "policy denied: {message}"),
            Self::BrowserFeatureUnsupported { message } => {
                write!(f, "browser feature unsupported: {message}")
            }
            Self::InternalSampling { message } => write!(f, "internal sampling error: {message}"),
        }
    }
}

impl Error for SamplingError {}

impl From<crate::GenerationError> for SamplingError {
    fn from(error: crate::GenerationError) -> Self {
        use crate::GenerationError;
        match error {
            GenerationError::ParameterInvalid {
                parameter: "temperature",
                message,
            } => SamplingError::TemperatureInvalid { message },
            GenerationError::ParameterInvalid { parameter, message } => {
                SamplingError::InvalidSamplingParameter { parameter, message }
            }
            GenerationError::DeterministicModeUnsupported { message } => {
                SamplingError::DeterministicModeUnsupported { message }
            }
            GenerationError::SamplingModeUnsupported { message } => {
                SamplingError::ProcessorUnsupported { message }
            }
            other => SamplingError::InvalidSamplingParameter {
                parameter: "generation parameters",
                message: other.to_string(),
            },
        }
    }
}
