//! Token-based generation contract for inference Runtime.
//!
//! Generation consumes tokenizer-validated input token IDs and produces new
//! token IDs. It does not own tokenization, chat rendering, Provider or Device
//! selection, sampling implementation, or KV cache internals.

use crate::{
    CorrelationId, MemoryAdmissionDecision, MemoryAdmissionRequest, MemoryAllocationClass,
    MemoryAllocationOwner, MemoryAllocationRequest, MemoryManager, MemoryPlacement,
    ModelArtifactId, ProviderExecutionErrorCode, RuntimeTokenizer, StreamingDecodeState, TokenId,
    TokenStopPattern, Tokenizer, TokenizerError, TokenizerId, TokenizerMetadata, TraceId,
};
use std::{error::Error, fmt};

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct GenerationRequestId(String);

impl GenerationRequestId {
    pub fn new(value: impl Into<String>) -> Result<Self, GenerationError> {
        let value = value.into();
        validate_identity(&value, "generation request id")?;
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for GenerationRequestId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum GenerationModelReference {
    ModelArtifact(ModelArtifactId),
    LoadedModelContext(String),
    FutureModelInstance(String),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GenerationTokenizerReference {
    pub tokenizer_id: TokenizerId,
    pub metadata: TokenizerMetadata,
}

#[derive(Clone, Debug, PartialEq)]
pub struct GenerationRequest {
    pub request_id: GenerationRequestId,
    pub model: GenerationModelReference,
    pub tokenizer: GenerationTokenizerReference,
    pub input_token_ids: Vec<TokenId>,
    pub prompt_token_count: usize,
    pub max_new_tokens: usize,
    pub max_total_tokens: Option<usize>,
    pub model_context_length: Option<usize>,
    pub parameters: GenerationParameters,
    pub stop_conditions: StopConditions,
    pub streaming: StreamingMode,
    pub priority: GenerationPriority,
    pub cancellation: CancellationMetadata,
    pub memory: GenerationMemoryEstimate,
    pub correlation_id: Option<CorrelationId>,
    pub trace_id: Option<TraceId>,
}

impl GenerationRequest {
    pub fn validate(&self) -> Result<(), GenerationError> {
        if self.input_token_ids.is_empty() {
            return Err(GenerationError::InputTokensInvalid {
                message: "input token ids must not be empty".into(),
            });
        }
        if self.prompt_token_count != self.input_token_ids.len() {
            return Err(GenerationError::InputTokensInvalid {
                message: "prompt token count must match input token id length".into(),
            });
        }
        if self.max_new_tokens == 0 {
            return Err(GenerationError::MaxTokensInvalid {
                message: "max new tokens must be greater than zero".into(),
            });
        }
        for token_id in &self.input_token_ids {
            if !self.tokenizer.metadata.token_id_range.contains(*token_id) {
                return Err(GenerationError::InputTokensInvalid {
                    message: format!("token id {token_id} is outside tokenizer range"),
                });
            }
        }
        self.parameters.validate()?;
        self.stop_conditions.validate(&self.tokenizer.metadata)?;
        if self.parameters.deterministic && self.parameters.seed.is_none() {
            return Err(GenerationError::DeterministicModeUnsupported {
                message: "deterministic mode requires an explicit seed until an execution path declares stronger support".into(),
            });
        }
        let requested_total = self
            .prompt_token_count
            .checked_add(self.max_new_tokens)
            .ok_or_else(|| GenerationError::MaxTokensInvalid {
                message: "requested total token count overflowed".into(),
            })?;
        if let Some(max_total_tokens) = self.max_total_tokens
            && requested_total > max_total_tokens
        {
            return Err(GenerationError::PromptTooLong {
                prompt_tokens: self.prompt_token_count,
                requested_total_tokens: requested_total,
                limit: max_total_tokens,
            });
        }
        if let Some(context_length) = self.model_context_length
            && requested_total > context_length
        {
            return Err(GenerationError::PromptTooLong {
                prompt_tokens: self.prompt_token_count,
                requested_total_tokens: requested_total,
                limit: context_length,
            });
        }
        if let Some(tokenizer_length) = self.tokenizer.metadata.model_max_length
            && requested_total > tokenizer_length as usize
        {
            return Err(GenerationError::PromptTooLong {
                prompt_tokens: self.prompt_token_count,
                requested_total_tokens: requested_total,
                limit: tokenizer_length as usize,
            });
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct GenerationParameters {
    pub temperature: f32,
    pub top_p: Option<f32>,
    pub top_k: Option<u32>,
    pub min_p: Option<f32>,
    pub typical_p: Option<f32>,
    pub repetition_penalty: Option<f32>,
    pub frequency_penalty: Option<f32>,
    pub presence_penalty: Option<f32>,
    pub seed: Option<u64>,
    pub deterministic: bool,
    pub greedy: bool,
    pub sampling_enabled: bool,
    pub banned_token_ids: Vec<TokenId>,
    pub allowed_token_ids: Option<Vec<TokenId>>,
    pub logits_processors: Vec<LogitsProcessorReference>,
}

impl Default for GenerationParameters {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_p: None,
            top_k: None,
            min_p: None,
            typical_p: None,
            repetition_penalty: None,
            frequency_penalty: None,
            presence_penalty: None,
            seed: None,
            deterministic: false,
            greedy: false,
            sampling_enabled: true,
            banned_token_ids: Vec::new(),
            allowed_token_ids: None,
            logits_processors: Vec::new(),
        }
    }
}

impl GenerationParameters {
    pub fn greedy() -> Self {
        Self {
            temperature: 0.0,
            greedy: true,
            sampling_enabled: false,
            ..Self::default()
        }
    }

    pub fn validate(&self) -> Result<(), GenerationError> {
        validate_probability("temperature", self.temperature, true)?;
        for (name, value) in [
            ("top-p", self.top_p),
            ("min-p", self.min_p),
            ("typical-p", self.typical_p),
        ] {
            if let Some(value) = value {
                validate_probability(name, value, false)?;
            }
        }
        for (name, value) in [
            ("repetition penalty", self.repetition_penalty),
            ("frequency penalty", self.frequency_penalty),
            ("presence penalty", self.presence_penalty),
        ] {
            if let Some(value) = value
                && (!value.is_finite() || value < 0.0)
            {
                return Err(GenerationError::ParameterInvalid {
                    parameter: name,
                    message: "penalty must be finite and non-negative".into(),
                });
            }
        }
        if self.greedy && self.sampling_enabled {
            return Err(GenerationError::ParameterInvalid {
                parameter: "greedy",
                message: "greedy mode cannot also enable sampling".into(),
            });
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LogitsProcessorReference {
    pub id: String,
    pub policy_controlled: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StopConditions {
    pub eos: EosPolicy,
    pub stop_token_ids: Vec<TokenId>,
    pub stop_token_patterns: Vec<Vec<TokenId>>,
    pub stop_text_sequences: Vec<String>,
    pub prepared_stop_sequences: Vec<TokenStopPattern>,
    pub cancellation_enabled: bool,
    pub memory_policy_stop: bool,
    pub runtime_shutdown_stop: bool,
}

impl Default for StopConditions {
    fn default() -> Self {
        Self {
            eos: EosPolicy::default(),
            stop_token_ids: Vec::new(),
            stop_token_patterns: Vec::new(),
            stop_text_sequences: Vec::new(),
            prepared_stop_sequences: Vec::new(),
            cancellation_enabled: true,
            memory_policy_stop: true,
            runtime_shutdown_stop: true,
        }
    }
}

impl StopConditions {
    pub fn validate(&self, metadata: &TokenizerMetadata) -> Result<(), GenerationError> {
        for token_id in self
            .eos
            .eos_token_ids
            .iter()
            .chain(self.stop_token_ids.iter())
            .chain(self.stop_token_patterns.iter().flatten())
            .chain(
                self.prepared_stop_sequences
                    .iter()
                    .flat_map(|pattern| pattern.token_ids.iter()),
            )
        {
            if !metadata.token_id_range.contains(*token_id) {
                return Err(GenerationError::StopConditionInvalid {
                    message: format!("stop token id {token_id} is outside tokenizer range"),
                });
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EosPolicy {
    pub mode: EosMode,
    pub output: EosOutputPolicy,
    pub eos_token_ids: Vec<TokenId>,
}

impl Default for EosPolicy {
    fn default() -> Self {
        Self {
            mode: EosMode::Stop,
            output: EosOutputPolicy::Exclude,
            eos_token_ids: Vec::new(),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EosMode {
    Stop,
    Ignore,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EosOutputPolicy {
    Include,
    Exclude,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum StreamingMode {
    #[default]
    Disabled,
    TokenIds,
    TokenIdsWithTokenizerText,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct GenerationPriority {
    pub priority: u8,
    pub deadline_millis: Option<u64>,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct CancellationMetadata {
    pub cancellation_id: Option<String>,
    pub requested: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GenerationMemoryEstimate {
    pub input_token_buffer_bytes: u64,
    pub output_token_buffer_bytes: u64,
    pub logits_buffer_bytes: u64,
    pub sampling_workspace_bytes: u64,
    pub prefill_workspace_bytes: u64,
    pub decode_workspace_bytes: u64,
    pub kv_cache_placeholder_bytes: u64,
    pub prefix_cache_placeholder_bytes: u64,
    pub placement: MemoryPlacement,
    pub queue_allowed: bool,
}

impl Default for GenerationMemoryEstimate {
    fn default() -> Self {
        Self {
            input_token_buffer_bytes: 0,
            output_token_buffer_bytes: 0,
            logits_buffer_bytes: 0,
            sampling_workspace_bytes: 0,
            prefill_workspace_bytes: 0,
            decode_workspace_bytes: 0,
            kv_cache_placeholder_bytes: 0,
            prefix_cache_placeholder_bytes: 0,
            placement: MemoryPlacement::HostOrdinary,
            queue_allowed: false,
        }
    }
}

impl GenerationMemoryEstimate {
    pub fn total_bytes(&self) -> Result<u64, GenerationError> {
        [
            self.input_token_buffer_bytes,
            self.output_token_buffer_bytes,
            self.logits_buffer_bytes,
            self.sampling_workspace_bytes,
            self.prefill_workspace_bytes,
            self.decode_workspace_bytes,
            self.kv_cache_placeholder_bytes,
            self.prefix_cache_placeholder_bytes,
        ]
        .into_iter()
        .try_fold(0u64, |total, value| {
            total
                .checked_add(value)
                .ok_or(GenerationError::InternalGeneration {
                    message: "generation memory estimate overflowed".into(),
                })
        })
    }

    pub fn admission_request(
        &self,
        request_id: &GenerationRequestId,
    ) -> Result<MemoryAdmissionRequest, GenerationError> {
        Ok(MemoryAdmissionRequest {
            allocation: MemoryAllocationRequest::new(
                MemoryAllocationClass::TemporaryWorkspace,
                self.total_bytes()?,
                self.placement.clone(),
                MemoryAllocationOwner::Session(request_id.as_str().into()),
            ),
            pressure: Default::default(),
            queue_allowed: self.queue_allowed,
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GenerationOutput {
    pub request_id: GenerationRequestId,
    pub generated_token_ids: Vec<TokenId>,
    pub generated_token_count: usize,
    pub finish_reason: FinishReason,
    pub usage: GenerationUsage,
    pub diagnostics: Vec<GenerationDiagnostic>,
}

impl GenerationOutput {
    pub fn new(
        request: &GenerationRequest,
        generated_token_ids: Vec<TokenId>,
        finish_reason: FinishReason,
    ) -> Self {
        let generated_token_count = generated_token_ids.len();
        Self {
            request_id: request.request_id.clone(),
            generated_token_ids,
            generated_token_count,
            finish_reason,
            usage: GenerationUsage::new(
                request.prompt_token_count,
                generated_token_count,
                finish_reason,
            ),
            diagnostics: Vec::new(),
        }
    }

    pub fn validate(&self) -> Result<(), GenerationError> {
        if self.generated_token_count != self.generated_token_ids.len() {
            return Err(GenerationError::InternalGeneration {
                message: "generated token count must match generated token id length".into(),
            });
        }
        if self.generated_token_count != self.usage.generated_tokens {
            return Err(GenerationError::InternalGeneration {
                message: "usage generated token count must match output".into(),
            });
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum FinishReason {
    MaxNewTokens,
    MaxTotalTokens,
    EosToken,
    StopToken,
    StopSequence,
    Cancelled,
    Interrupted,
    LengthLimit,
    MemoryLimit,
    RuntimeShutdown,
    ProviderError,
    ModelError,
    PolicyDenied,
    Error,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GenerationUsage {
    pub prompt_tokens: usize,
    pub generated_tokens: usize,
    pub total_tokens: usize,
    pub prefill_duration_millis: Option<u64>,
    pub decode_duration_millis: Option<u64>,
    pub tokens_per_second: Option<u64>,
    pub finish_reason: FinishReason,
}

impl GenerationUsage {
    pub fn new(prompt_tokens: usize, generated_tokens: usize, finish_reason: FinishReason) -> Self {
        Self {
            prompt_tokens,
            generated_tokens,
            total_tokens: prompt_tokens.saturating_add(generated_tokens),
            prefill_duration_millis: None,
            decode_duration_millis: None,
            tokens_per_second: None,
            finish_reason,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GenerationDiagnostic {
    pub kind: GenerationDiagnosticKind,
    pub message: String,
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum GenerationDiagnosticKind {
    DeterminismUnsupported,
    UnsupportedOnBrowser,
    MemoryAdmission,
    ProviderExecution,
    TokenizerBoundary,
    RedactedPrompt,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PrefillState {
    pub request_id: GenerationRequestId,
    pub prompt_token_count: usize,
    pub model_execution_state: Option<String>,
    pub kv_cache_placeholder: Option<String>,
    pub observations: Vec<GenerationEvent>,
}

pub fn prefill(request: &GenerationRequest) -> Result<PrefillState, GenerationError> {
    request.validate()?;
    Ok(PrefillState {
        request_id: request.request_id.clone(),
        prompt_token_count: request.prompt_token_count,
        model_execution_state: Some("runtime-owned-model-execution-state".into()),
        kv_cache_placeholder: Some("future-kv-cache-state".into()),
        observations: vec![
            GenerationEvent::new(request, GenerationEventKind::PrefillStarted),
            GenerationEvent::new(request, GenerationEventKind::PrefillCompleted),
        ],
    })
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DecodeStepInput {
    pub request_id: GenerationRequestId,
    pub generated_so_far: Vec<TokenId>,
    pub next_token_logits_ref: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DecodeStepOutput {
    pub token_id: TokenId,
    pub token_index: usize,
    pub finish_reason: Option<FinishReason>,
    pub state_update: Option<String>,
}

pub fn decode_step(
    request: &GenerationRequest,
    generated_so_far: &[TokenId],
    next_token_id: TokenId,
) -> Result<DecodeStepOutput, GenerationError> {
    request.validate()?;
    if !request
        .tokenizer
        .metadata
        .token_id_range
        .contains(next_token_id)
    {
        return Err(GenerationError::InputTokensInvalid {
            message: format!("generated token id {next_token_id} is outside tokenizer range"),
        });
    }
    let mut candidate = generated_so_far.to_vec();
    candidate.push(next_token_id);
    let finish_reason = stop_reason_for(request, &candidate);
    Ok(DecodeStepOutput {
        token_id: next_token_id,
        token_index: generated_so_far.len(),
        finish_reason,
        state_update: Some("runtime-owned-decode-state".into()),
    })
}

pub fn stop_reason_for(request: &GenerationRequest, generated: &[TokenId]) -> Option<FinishReason> {
    if generated.len() >= request.max_new_tokens {
        return Some(FinishReason::MaxNewTokens);
    }
    if let Some(max_total) = request.max_total_tokens
        && request.prompt_token_count.saturating_add(generated.len()) >= max_total
    {
        return Some(FinishReason::MaxTotalTokens);
    }
    let last = generated.last().copied();
    if let Some(token_id) = last {
        if request.stop_conditions.stop_token_ids.contains(&token_id) {
            return Some(FinishReason::StopToken);
        }
        if request.stop_conditions.eos.mode == EosMode::Stop
            && request
                .stop_conditions
                .eos
                .eos_token_ids
                .contains(&token_id)
        {
            return Some(FinishReason::EosToken);
        }
    }
    if request
        .stop_conditions
        .stop_token_patterns
        .iter()
        .chain(
            request
                .stop_conditions
                .prepared_stop_sequences
                .iter()
                .map(|pattern| &pattern.token_ids),
        )
        .any(|pattern| generated.ends_with(pattern))
    {
        return Some(FinishReason::StopSequence);
    }
    if request.cancellation.requested && request.stop_conditions.cancellation_enabled {
        return Some(FinishReason::Cancelled);
    }
    None
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GenerationEvent {
    pub request_id: GenerationRequestId,
    pub kind: GenerationEventKind,
    pub token_id: Option<TokenId>,
    pub token_index: Option<usize>,
    pub token_probability: Option<Probability>,
    pub finish_reason: Option<FinishReason>,
    pub usage: Option<GenerationUsage>,
    pub correlation_id: Option<CorrelationId>,
}

impl GenerationEvent {
    pub fn new(request: &GenerationRequest, kind: GenerationEventKind) -> Self {
        Self {
            request_id: request.request_id.clone(),
            kind,
            token_id: None,
            token_index: None,
            token_probability: None,
            finish_reason: None,
            usage: None,
            correlation_id: request.correlation_id.clone(),
        }
    }

    pub fn token_generated(
        request: &GenerationRequest,
        token_id: TokenId,
        token_index: usize,
        token_probability: Option<Probability>,
    ) -> Self {
        Self {
            token_id: Some(token_id),
            token_index: Some(token_index),
            token_probability,
            ..Self::new(request, GenerationEventKind::TokenGenerated)
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum GenerationEventKind {
    GenerationRequested,
    GenerationAdmitted,
    GenerationRejected,
    GenerationStarted,
    PrefillStarted,
    PrefillCompleted,
    DecodeStarted,
    TokenGenerated,
    DecodeStepCompleted,
    StopConditionMet,
    GenerationCompleted,
    GenerationCancelled,
    GenerationFailed,
    MemoryAdmissionFailed,
    ProviderExecutionFailed,
    StreamingBackpressure,
    UsageUpdated,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Probability(f32);

impl Probability {
    pub fn new(value: f32) -> Result<Self, GenerationError> {
        validate_probability("probability", value, false)?;
        Ok(Self(value))
    }

    pub const fn value(self) -> f32 {
        self.0
    }
}

impl Eq for Probability {}

pub fn token_stream_events(
    request: &GenerationRequest,
    tokens: &[TokenId],
    probabilities: Option<&[Probability]>,
) -> Result<Vec<GenerationEvent>, GenerationError> {
    if let Some(probabilities) = probabilities
        && probabilities.len() != tokens.len()
    {
        return Err(GenerationError::StreamingConsumerFailed {
            message: "token probability count must match token count".into(),
        });
    }
    Ok(tokens
        .iter()
        .copied()
        .enumerate()
        .map(|(index, token_id)| {
            GenerationEvent::token_generated(
                request,
                token_id,
                index,
                probabilities.and_then(|values| values.get(index).copied()),
            )
        })
        .collect())
}

pub fn streaming_text_chunk<T: Tokenizer>(
    tokenizer: &RuntimeTokenizer<T>,
    state: StreamingDecodeState,
    token_ids: Vec<TokenId>,
    flush: bool,
) -> Result<crate::DecodeOutput, GenerationError> {
    tokenizer
        .decode(
            crate::DecodeInput {
                token_ids,
                skip_special_tokens: true,
                clean_up_tokenization_spaces: false,
                streaming_state: Some(state),
            },
            &mut crate::TokenizerObserver::default(),
        )
        .map_err(GenerationError::from)
        .map(|mut output| {
            if flush {
                output.pending_partial_state = None;
            }
            output
        })
}

pub fn prepare_stop_sequences<T: Tokenizer>(
    tokenizer: &RuntimeTokenizer<T>,
    stop_text_sequences: &[String],
) -> Result<Vec<TokenStopPattern>, GenerationError> {
    stop_text_sequences
        .iter()
        .map(|text| {
            tokenizer
                .implementation()
                .resolve_stop_sequence(text)
                .map_err(GenerationError::from)
        })
        .collect()
}

pub fn memory_admission(
    request: &GenerationRequest,
    manager: &MemoryManager,
) -> Result<MemoryAdmissionDecision, GenerationError> {
    let mut admission = request.memory.admission_request(&request.request_id)?;
    admission.pressure = manager.pressure_snapshot();
    Ok(manager.admit(admission))
}

pub fn finish_reason_from_provider_error(code: ProviderExecutionErrorCode) -> FinishReason {
    match code {
        ProviderExecutionErrorCode::ExecutionInterrupted => FinishReason::Interrupted,
        ProviderExecutionErrorCode::OutOfMemory | ProviderExecutionErrorCode::ResourceExhausted => {
            FinishReason::MemoryLimit
        }
        _ => FinishReason::ProviderError,
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum GenerationError {
    ModelNotLoaded,
    ModelArtifactInvalid {
        message: String,
    },
    ModelInstanceUnavailable {
        message: String,
    },
    TokenizerIncompatible {
        message: String,
    },
    InputTokensInvalid {
        message: String,
    },
    PromptTooLong {
        prompt_tokens: usize,
        requested_total_tokens: usize,
        limit: usize,
    },
    MaxTokensInvalid {
        message: String,
    },
    ParameterInvalid {
        parameter: &'static str,
        message: String,
    },
    StopConditionInvalid {
        message: String,
    },
    DeterministicModeUnsupported {
        message: String,
    },
    SamplingModeUnsupported {
        message: String,
    },
    LogitsProcessorUnsupported {
        message: String,
    },
    MemoryAdmissionFailed {
        message: String,
    },
    ProviderResolutionFailed {
        message: String,
    },
    ProviderExecutionFailed {
        message: String,
    },
    ProviderNotReady {
        message: String,
    },
    ProviderSaturated {
        message: String,
    },
    CancellationRequested,
    CancellationUnsupported {
        message: String,
    },
    StreamingConsumerFailed {
        message: String,
    },
    RuntimeShutdown,
    GenerationInterrupted {
        message: String,
    },
    InternalGeneration {
        message: String,
    },
}

impl fmt::Display for GenerationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ModelNotLoaded => f.write_str("model is not loaded"),
            Self::ModelArtifactInvalid { message } => {
                write!(f, "model artifact invalid: {message}")
            }
            Self::ModelInstanceUnavailable { message } => {
                write!(f, "model instance unavailable: {message}")
            }
            Self::TokenizerIncompatible { message } => {
                write!(f, "tokenizer incompatible: {message}")
            }
            Self::InputTokensInvalid { message } => write!(f, "input tokens invalid: {message}"),
            Self::PromptTooLong {
                prompt_tokens,
                requested_total_tokens,
                limit,
            } => write!(
                f,
                "prompt has {prompt_tokens} tokens and requested total is {requested_total_tokens}, limit is {limit}"
            ),
            Self::MaxTokensInvalid { message } => write!(f, "max tokens invalid: {message}"),
            Self::ParameterInvalid { parameter, message } => {
                write!(f, "generation parameter '{parameter}' invalid: {message}")
            }
            Self::StopConditionInvalid { message } => {
                write!(f, "stop condition invalid: {message}")
            }
            Self::DeterministicModeUnsupported { message } => {
                write!(f, "deterministic mode unsupported: {message}")
            }
            Self::SamplingModeUnsupported { message } => {
                write!(f, "sampling mode unsupported: {message}")
            }
            Self::LogitsProcessorUnsupported { message } => {
                write!(f, "logits processor unsupported: {message}")
            }
            Self::MemoryAdmissionFailed { message } => {
                write!(f, "generation memory admission failed: {message}")
            }
            Self::ProviderResolutionFailed { message } => {
                write!(f, "provider resolution failed: {message}")
            }
            Self::ProviderExecutionFailed { message } => {
                write!(f, "provider execution failed: {message}")
            }
            Self::ProviderNotReady { message } => write!(f, "provider not ready: {message}"),
            Self::ProviderSaturated { message } => write!(f, "provider saturated: {message}"),
            Self::CancellationRequested => f.write_str("generation cancellation requested"),
            Self::CancellationUnsupported { message } => {
                write!(f, "generation cancellation unsupported: {message}")
            }
            Self::StreamingConsumerFailed { message } => {
                write!(f, "streaming consumer failed: {message}")
            }
            Self::RuntimeShutdown => f.write_str("runtime shutdown during generation"),
            Self::GenerationInterrupted { message } => {
                write!(f, "generation interrupted: {message}")
            }
            Self::InternalGeneration { message } => {
                write!(f, "internal generation error: {message}")
            }
        }
    }
}

impl Error for GenerationError {}

impl From<TokenizerError> for GenerationError {
    fn from(error: TokenizerError) -> Self {
        Self::TokenizerIncompatible {
            message: error.to_string(),
        }
    }
}

fn validate_probability(
    parameter: &'static str,
    value: f32,
    allow_zero: bool,
) -> Result<(), GenerationError> {
    let lower_valid = if allow_zero {
        value >= 0.0
    } else {
        value > 0.0
    };
    if !value.is_finite() || !lower_valid || value > 1.0 && parameter != "temperature" {
        return Err(GenerationError::ParameterInvalid {
            parameter,
            message: if parameter == "temperature" {
                "temperature must be finite and non-negative".into()
            } else {
                "probability must be finite and within (0, 1]".into()
            },
        });
    }
    Ok(())
}

fn validate_identity(value: &str, label: &str) -> Result<(), GenerationError> {
    if value.trim().is_empty()
        || value.contains('/')
        || value.contains('\\')
        || value.contains(':')
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'-' | b'_'))
    {
        return Err(GenerationError::InputTokensInvalid {
            message: format!("{label} must use a portable identifier"),
        });
    }
    Ok(())
}
