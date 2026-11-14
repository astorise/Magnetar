//! Runtime-owned inference session contracts.
//!
//! A session is the bounded Runtime context used to bind inference resources
//! together. It is not a client conversation, model artifact, provider handle,
//! device handle, workspace, tool state, or raw prompt store.

pub use crate::model_identity::{InferenceSessionId, SessionError, SessionLifecycleState};
use crate::{
    CorrelationId, FallbackClass, GenerationModelReference, GenerationParameters,
    GenerationTokenizerReference, KvCacheId, KvCachePolicy, MemoryAdmissionDecision,
    MemoryAdmissionRequest, MemoryAllocationClass, MemoryAllocationId, MemoryAllocationOwner,
    MemoryAllocationRequest, MemoryManager, MemoryPlacement, ResourceAffinity,
    StreamingDecodeState,
};
use std::collections::BTreeSet;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum SessionConcurrencyPolicy {
    #[default]
    SingleActiveOperation,
    AllowParallelOperations,
    QueueOperations,
    RejectWhileActive,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SessionPolicy {
    pub max_prompt_tokens: Option<usize>,
    pub max_generated_tokens: Option<usize>,
    pub max_total_tokens: Option<usize>,
    pub allowed_generation_parameters: BTreeSet<SessionGenerationParameter>,
    pub sampling_modes_allowed: bool,
    pub streaming_allowed: bool,
    pub cancellation_allowed: bool,
    pub concurrency: SessionConcurrencyPolicy,
    pub memory_budget_bytes: Option<u64>,
    pub kv_cache_budget_bytes: Option<u64>,
    pub kv_cache_policy: KvCachePolicy,
    pub prefix_cache_allowed: bool,
    pub redaction: SessionRedactionPolicy,
    pub raw_prompt_logging_allowed: bool,
    pub idle_ttl_millis: Option<u64>,
    pub total_ttl_millis: Option<u64>,
}

impl Default for SessionPolicy {
    fn default() -> Self {
        Self {
            max_prompt_tokens: None,
            max_generated_tokens: None,
            max_total_tokens: None,
            allowed_generation_parameters: SessionGenerationParameter::all(),
            sampling_modes_allowed: true,
            streaming_allowed: true,
            cancellation_allowed: true,
            concurrency: SessionConcurrencyPolicy::default(),
            memory_budget_bytes: None,
            kv_cache_budget_bytes: None,
            kv_cache_policy: KvCachePolicy::default(),
            prefix_cache_allowed: false,
            redaction: SessionRedactionPolicy::RedactRawInputs,
            raw_prompt_logging_allowed: false,
            idle_ttl_millis: None,
            total_ttl_millis: None,
        }
    }
}

impl SessionPolicy {
    pub fn validate(&self) -> Result<(), SessionError> {
        if let (Some(prompt), Some(total)) = (self.max_prompt_tokens, self.max_total_tokens)
            && prompt > total
        {
            return Err(SessionError::SessionPolicyDenied {
                reason: "max prompt tokens exceeds max total tokens".into(),
            });
        }
        if let (Some(generated), Some(total)) = (self.max_generated_tokens, self.max_total_tokens)
            && generated > total
        {
            return Err(SessionError::SessionPolicyDenied {
                reason: "max generated tokens exceeds max total tokens".into(),
            });
        }
        if matches!(self.idle_ttl_millis, Some(0)) || matches!(self.total_ttl_millis, Some(0)) {
            return Err(SessionError::SessionPolicyDenied {
                reason: "session TTL must be greater than zero when set".into(),
            });
        }
        Ok(())
    }

    pub fn validate_generation(
        &self,
        prompt_tokens: usize,
        generated_tokens: usize,
    ) -> Result<(), SessionError> {
        let total = prompt_tokens.checked_add(generated_tokens).ok_or_else(|| {
            SessionError::SessionPolicyDenied {
                reason: "requested token count overflowed".into(),
            }
        })?;
        if let Some(limit) = self.max_prompt_tokens
            && prompt_tokens > limit
        {
            return Err(SessionError::SessionPolicyDenied {
                reason: "prompt token limit exceeded".into(),
            });
        }
        if let Some(limit) = self.max_generated_tokens
            && generated_tokens > limit
        {
            return Err(SessionError::SessionPolicyDenied {
                reason: "generated token limit exceeded".into(),
            });
        }
        if let Some(limit) = self.max_total_tokens
            && total > limit
        {
            return Err(SessionError::SessionPolicyDenied {
                reason: "total token limit exceeded".into(),
            });
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum SessionGenerationParameter {
    Temperature,
    TopP,
    TopK,
    MinP,
    TypicalP,
    RepetitionPenalty,
    FrequencyPenalty,
    PresencePenalty,
    Seed,
    Deterministic,
    Greedy,
    LogitsProcessors,
}

impl SessionGenerationParameter {
    pub fn all() -> BTreeSet<Self> {
        [
            Self::Temperature,
            Self::TopP,
            Self::TopK,
            Self::MinP,
            Self::TypicalP,
            Self::RepetitionPenalty,
            Self::FrequencyPenalty,
            Self::PresencePenalty,
            Self::Seed,
            Self::Deterministic,
            Self::Greedy,
            Self::LogitsProcessors,
        ]
        .into_iter()
        .collect()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SessionRedactionPolicy {
    RedactRawInputs,
    PolicyControlled,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SessionCreationRequest {
    pub model: GenerationModelReference,
    pub tokenizer: GenerationTokenizerReference,
    pub generation_defaults: GenerationParameters,
    pub policy: SessionPolicy,
    pub memory: SessionMemoryBudget,
    pub allowed_capabilities: BTreeSet<String>,
    pub correlation_id: Option<CorrelationId>,
    pub created_at_millis: u64,
}

impl SessionCreationRequest {
    pub fn validate(&self) -> Result<(), SessionError> {
        self.generation_defaults.validate().map_err(|error| {
            SessionError::SessionCreationFailed {
                reason: error.to_string(),
            }
        })?;
        self.policy.validate()?;
        if let Some(limit) = self.policy.memory_budget_bytes
            && self.memory.total_reserved_bytes()? > limit
        {
            return Err(SessionError::MemoryBudgetExceeded {
                reason: "requested session memory exceeds policy budget".into(),
            });
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SessionMemoryBudget {
    pub input_token_buffer_bytes: u64,
    pub output_token_buffer_bytes: u64,
    pub logits_buffer_bytes: u64,
    pub sampling_workspace_bytes: u64,
    pub tokenizer_streaming_state_bytes: u64,
    pub temporary_workspace_bytes: u64,
    pub kv_cache_placeholder_bytes: u64,
    pub prefix_cache_placeholder_bytes: u64,
    pub placement: MemoryPlacement,
}

impl Default for SessionMemoryBudget {
    fn default() -> Self {
        Self {
            input_token_buffer_bytes: 0,
            output_token_buffer_bytes: 0,
            logits_buffer_bytes: 0,
            sampling_workspace_bytes: 0,
            tokenizer_streaming_state_bytes: 0,
            temporary_workspace_bytes: 0,
            kv_cache_placeholder_bytes: 0,
            prefix_cache_placeholder_bytes: 0,
            placement: MemoryPlacement::HostOrdinary,
        }
    }
}

impl SessionMemoryBudget {
    pub fn total_reserved_bytes(&self) -> Result<u64, SessionError> {
        [
            self.input_token_buffer_bytes,
            self.output_token_buffer_bytes,
            self.logits_buffer_bytes,
            self.sampling_workspace_bytes,
            self.tokenizer_streaming_state_bytes,
            self.temporary_workspace_bytes,
            self.kv_cache_placeholder_bytes,
            self.prefix_cache_placeholder_bytes,
        ]
        .into_iter()
        .try_fold(0u64, |total, value| {
            total
                .checked_add(value)
                .ok_or_else(|| SessionError::MemoryBudgetExceeded {
                    reason: "session memory budget overflowed".into(),
                })
        })
    }

    pub fn admission_request(
        &self,
        session: &InferenceSessionId,
        queue_allowed: bool,
    ) -> Result<MemoryAdmissionRequest, SessionError> {
        Ok(MemoryAdmissionRequest {
            allocation: MemoryAllocationRequest::new(
                MemoryAllocationClass::TemporaryWorkspace,
                self.total_reserved_bytes()?,
                self.placement.clone(),
                MemoryAllocationOwner::Session(session.as_str().into()),
            ),
            pressure: Default::default(),
            queue_allowed,
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SessionResources {
    pub tokenizer_state: bool,
    pub streaming_decode_state: Option<StreamingDecodeState>,
    pub output_token_buffer_tokens: usize,
    pub temporary_generation_buffer_bytes: u64,
    pub memory_allocations: BTreeSet<MemoryAllocationId>,
    pub kv_cache: Option<KvCacheId>,
    pub prefix_cache_placeholder: Option<String>,
    pub model_residency_reference: Option<String>,
}

impl Default for SessionResources {
    fn default() -> Self {
        Self {
            tokenizer_state: true,
            streaming_decode_state: None,
            output_token_buffer_tokens: 0,
            temporary_generation_buffer_bytes: 0,
            memory_allocations: BTreeSet::new(),
            kv_cache: None,
            prefix_cache_placeholder: None,
            model_residency_reference: None,
        }
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct SessionOperationState {
    pub active_operations: usize,
    pub queued_operations: usize,
    pub cancellation_requested: bool,
    pub finish_reason: Option<String>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct InferenceSession {
    pub id: InferenceSessionId,
    pub lifecycle: SessionLifecycleState,
    pub model: GenerationModelReference,
    pub tokenizer: GenerationTokenizerReference,
    pub generation_defaults: GenerationParameters,
    pub policy: SessionPolicy,
    pub memory: SessionMemoryBudget,
    pub resources: SessionResources,
    pub operation: SessionOperationState,
    pub affinity: ResourceAffinity,
    pub correlation_id: Option<CorrelationId>,
    pub created_at_millis: u64,
    pub last_activity_millis: u64,
    pub expires_at_millis: Option<u64>,
    pub last_error: Option<SessionError>,
}

impl InferenceSession {
    pub fn create(
        id: InferenceSessionId,
        request: SessionCreationRequest,
        runtime_affinity: ResourceAffinity,
    ) -> Result<Self, SessionError> {
        request.validate()?;
        let expires_at_millis = request
            .policy
            .total_ttl_millis
            .map(|ttl| request.created_at_millis.saturating_add(ttl));
        Ok(Self {
            id,
            lifecycle: SessionLifecycleState::Ready,
            model: request.model,
            tokenizer: request.tokenizer,
            generation_defaults: request.generation_defaults,
            policy: request.policy,
            memory: request.memory,
            resources: SessionResources::default(),
            operation: SessionOperationState::default(),
            affinity: runtime_affinity,
            correlation_id: request.correlation_id,
            created_at_millis: request.created_at_millis,
            last_activity_millis: request.created_at_millis,
            expires_at_millis,
            last_error: None,
        })
    }

    pub fn transition_to(&mut self, next: SessionLifecycleState) -> Result<(), SessionError> {
        if self.lifecycle.allows_transition_to(next) {
            self.lifecycle = next;
            Ok(())
        } else {
            Err(SessionError::InvalidLifecycleTransition {
                from: self.lifecycle,
                to: next,
            })
        }
    }

    pub fn start_operation(&mut self) -> Result<SessionOperationAdmission, SessionError> {
        match self.lifecycle {
            SessionLifecycleState::Ready
            | SessionLifecycleState::Idle
            | SessionLifecycleState::Active => {}
            SessionLifecycleState::Draining => return Err(SessionError::SessionDraining),
            SessionLifecycleState::Closed => return Err(SessionError::SessionClosed),
            SessionLifecycleState::Expired => return Err(SessionError::SessionExpired),
            SessionLifecycleState::Cancelled => return Err(SessionError::SessionCancelled),
            _ => return Err(SessionError::SessionNotReady),
        }
        if self.operation.active_operations > 0 {
            match self.policy.concurrency {
                SessionConcurrencyPolicy::AllowParallelOperations => {}
                SessionConcurrencyPolicy::QueueOperations => {
                    self.operation.queued_operations =
                        self.operation.queued_operations.saturating_add(1);
                    return Ok(SessionOperationAdmission::Queued);
                }
                SessionConcurrencyPolicy::SingleActiveOperation
                | SessionConcurrencyPolicy::RejectWhileActive => {
                    return Err(SessionError::ConcurrencyViolation);
                }
            }
        }
        self.operation.active_operations = self.operation.active_operations.saturating_add(1);
        self.lifecycle = SessionLifecycleState::Active;
        Ok(SessionOperationAdmission::Started)
    }

    pub fn finish_operation(&mut self) -> Result<(), SessionError> {
        if self.operation.active_operations == 0 {
            return Err(SessionError::InternalSession {
                reason: "no active operation to finish".into(),
            });
        }
        self.operation.active_operations -= 1;
        if self.operation.active_operations == 0 && self.lifecycle == SessionLifecycleState::Active
        {
            self.lifecycle = SessionLifecycleState::Idle;
        }
        Ok(())
    }

    pub fn cancel(&mut self) -> Result<(), SessionError> {
        if !self.policy.cancellation_allowed {
            return Err(SessionError::SessionPolicyDenied {
                reason: "cancellation is disabled by session policy".into(),
            });
        }
        self.operation.cancellation_requested = true;
        if !self.lifecycle.is_terminal() {
            self.lifecycle = SessionLifecycleState::Cancelled;
        }
        Ok(())
    }

    pub fn drain(&mut self) -> Result<(), SessionError> {
        if matches!(
            self.lifecycle,
            SessionLifecycleState::Ready
                | SessionLifecycleState::Idle
                | SessionLifecycleState::Active
        ) {
            self.lifecycle = SessionLifecycleState::Draining;
            Ok(())
        } else {
            Err(SessionError::SessionNotReady)
        }
    }

    pub fn close(&mut self) -> Result<(), SessionError> {
        if self.lifecycle == SessionLifecycleState::Closed {
            return Ok(());
        }
        if matches!(self.lifecycle, SessionLifecycleState::Expired) {
            self.lifecycle = SessionLifecycleState::Closed;
        } else {
            self.transition_to(SessionLifecycleState::Closed)?;
        }
        self.resources.memory_allocations.clear();
        self.resources.streaming_decode_state = None;
        self.operation.active_operations = 0;
        self.operation.queued_operations = 0;
        Ok(())
    }

    pub fn expire_if_needed(&mut self, now_millis: u64) -> bool {
        let total_expired = self
            .expires_at_millis
            .is_some_and(|expires_at| now_millis >= expires_at);
        let idle_expired = self
            .policy
            .idle_ttl_millis
            .is_some_and(|ttl| now_millis >= self.last_activity_millis.saturating_add(ttl));
        if (total_expired || idle_expired)
            && matches!(
                self.lifecycle,
                SessionLifecycleState::Ready
                    | SessionLifecycleState::Idle
                    | SessionLifecycleState::Draining
            )
        {
            self.lifecycle = SessionLifecycleState::Expired;
            self.resources.memory_allocations.clear();
            self.resources.streaming_decode_state = None;
            return true;
        }
        false
    }

    pub fn memory_admission(
        &self,
        manager: &MemoryManager,
    ) -> Result<MemoryAdmissionDecision, SessionError> {
        let mut request = self.memory.admission_request(
            &self.id,
            self.policy.concurrency == SessionConcurrencyPolicy::QueueOperations,
        )?;
        request.pressure = manager.pressure_snapshot();
        Ok(manager.admit(request))
    }

    pub fn status(&self) -> SessionStatus {
        SessionStatus {
            id: self.id.clone(),
            lifecycle: self.lifecycle,
            model: self.model.clone(),
            tokenizer: self.tokenizer.tokenizer_id.clone(),
            active_operation_count: self.operation.active_operations,
            queued_operation_count: self.operation.queued_operations,
            memory_usage: SessionMemoryUsage {
                reserved_bytes: self.memory.total_reserved_bytes().unwrap_or(u64::MAX),
                allocation_count: self.resources.memory_allocations.len(),
                kv_cache_placeholder_bytes: self.memory.kv_cache_placeholder_bytes,
                prefix_cache_placeholder_bytes: self.memory.prefix_cache_placeholder_bytes,
            },
            streaming: SessionStreamingStatus {
                streaming_decode_active: self.resources.streaming_decode_state.is_some(),
                cancellation_requested: self.operation.cancellation_requested,
                finish_reason: self.operation.finish_reason.clone(),
            },
            last_error: self.last_error.as_ref().map(ToString::to_string),
            created_at_millis: self.created_at_millis,
            last_activity_millis: self.last_activity_millis,
            expires_at_millis: self.expires_at_millis,
            raw_prompt_available: false,
            raw_handles_available: false,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SessionOperationAdmission {
    Started,
    Queued,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SessionStatus {
    pub id: InferenceSessionId,
    pub lifecycle: SessionLifecycleState,
    pub model: GenerationModelReference,
    pub tokenizer: crate::TokenizerId,
    pub active_operation_count: usize,
    pub queued_operation_count: usize,
    pub memory_usage: SessionMemoryUsage,
    pub streaming: SessionStreamingStatus,
    pub last_error: Option<String>,
    pub created_at_millis: u64,
    pub last_activity_millis: u64,
    pub expires_at_millis: Option<u64>,
    pub raw_prompt_available: bool,
    pub raw_handles_available: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SessionMemoryUsage {
    pub reserved_bytes: u64,
    pub allocation_count: usize,
    pub kv_cache_placeholder_bytes: u64,
    pub prefix_cache_placeholder_bytes: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SessionStreamingStatus {
    pub streaming_decode_active: bool,
    pub cancellation_requested: bool,
    pub finish_reason: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SessionAccessPolicy {
    pub authorized_session_ids: BTreeSet<InferenceSessionId>,
    pub allow_raw_prompt: bool,
    pub allow_raw_handles: bool,
}

impl SessionAccessPolicy {
    pub fn authorize(session: InferenceSessionId) -> Self {
        Self {
            authorized_session_ids: [session].into_iter().collect(),
            allow_raw_prompt: false,
            allow_raw_handles: false,
        }
    }

    pub fn permits(&self, session: &InferenceSessionId) -> bool {
        self.authorized_session_ids.contains(session)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SessionObservationKind {
    CreateRequested,
    Created,
    CreationFailed,
    Ready,
    Active,
    Idle,
    Draining,
    Cancelled,
    Closed,
    Expired,
    OperationStarted,
    OperationCompleted,
    OperationFailed,
    MemoryPressure,
    Cleanup,
    PolicyRejection,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SessionObservation {
    pub kind: SessionObservationKind,
    pub session: Option<InferenceSessionId>,
    pub message: String,
    pub correlation_id: Option<CorrelationId>,
}

pub fn runtime_session_affinity(execution_context: crate::ExecutionContextId) -> ResourceAffinity {
    ResourceAffinity::new(FallbackClass::Transparent).with_execution_context(execution_context)
}

#[cfg(test)]
mod tests;
