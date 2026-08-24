//! Runtime/Scheduler-owned continuous batching contract.
//!
//! Continuous batching coordinates admitted generation operations over prefill
//! and decode steps. It owns batch identity, slot identity, lifecycle,
//! compatibility checks, policy decisions, redacted observations, and
//! Runtime-mediated memory admission requests. It does not own raw memory,
//! raw KV cache contents, Provider handles, Device handles, raw logits, or
//! token selection semantics.

use crate::{
    ComputeDType, DTypeDescriptor, DeviceBinding, GenerationModelReference, GenerationPriority,
    GenerationRequest, GenerationRequestId, InferenceSessionId, KvCacheId, KvCacheLayoutFormat,
    MemoryAdmissionRequest, MemoryAllocationClass, MemoryAllocationId, MemoryAllocationLifetime,
    MemoryAllocationOwner, MemoryAllocationRequest, MemoryPlacement, MemoryPressureLevel,
    PrefixCacheEntryId, ProviderBinding, ProviderPressureLevel, ResourceAffinity, SamplingPolicy,
    TokenizerId,
};
use std::{collections::BTreeMap, error::Error, fmt};

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct BatchId(String);

impl BatchId {
    pub fn new(value: impl Into<String>) -> Result<Self, BatchingError> {
        let value = value.into();
        validate_batch_identity(&value, "batch id")?;
        Ok(Self(value))
    }

    pub(crate) fn runtime_issued(sequence: u64) -> Self {
        Self(format!("batch-{sequence}"))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for BatchId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct BatchSlotId(String);

impl BatchSlotId {
    pub fn new(value: impl Into<String>) -> Result<Self, BatchingError> {
        let value = value.into();
        validate_batch_identity(&value, "batch slot id")?;
        Ok(Self(value))
    }

    pub(crate) fn runtime_issued(batch: &BatchId, sequence: u64) -> Self {
        Self(format!("{}.slot-{sequence}", batch.as_str()))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for BatchSlotId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum BatchedOperationState {
    Admitted,
    Queued,
    PrefillPending,
    Prefilling,
    DecodePending,
    Decoding,
    Streaming,
    Completed,
    Cancelled,
    Failed,
    Rejected,
    Evicted,
}

impl BatchedOperationState {
    pub const fn is_terminal(self) -> bool {
        matches!(
            self,
            Self::Completed | Self::Cancelled | Self::Failed | Self::Rejected | Self::Evicted
        )
    }

    pub const fn allows_transition_to(self, next: Self) -> bool {
        matches!(
            (self, next),
            (Self::Admitted, Self::Queued)
                | (Self::Admitted, Self::Rejected)
                | (Self::Admitted, Self::Cancelled)
                | (Self::Queued, Self::PrefillPending)
                | (Self::Queued, Self::DecodePending)
                | (Self::Queued, Self::Cancelled)
                | (Self::Queued, Self::Failed)
                | (Self::Queued, Self::Rejected)
                | (Self::Queued, Self::Evicted)
                | (Self::PrefillPending, Self::Prefilling)
                | (Self::PrefillPending, Self::Cancelled)
                | (Self::PrefillPending, Self::Failed)
                | (Self::Prefilling, Self::DecodePending)
                | (Self::Prefilling, Self::Streaming)
                | (Self::Prefilling, Self::Cancelled)
                | (Self::Prefilling, Self::Failed)
                | (Self::DecodePending, Self::Decoding)
                | (Self::DecodePending, Self::Completed)
                | (Self::DecodePending, Self::Cancelled)
                | (Self::DecodePending, Self::Failed)
                | (Self::Decoding, Self::DecodePending)
                | (Self::Decoding, Self::Streaming)
                | (Self::Decoding, Self::Completed)
                | (Self::Decoding, Self::Cancelled)
                | (Self::Decoding, Self::Failed)
                | (Self::Streaming, Self::DecodePending)
                | (Self::Streaming, Self::Completed)
                | (Self::Streaming, Self::Cancelled)
                | (Self::Streaming, Self::Failed)
        )
    }

    pub const fn is_prefill_ready(self) -> bool {
        matches!(self, Self::Queued | Self::PrefillPending)
    }

    pub const fn is_decode_ready(self) -> bool {
        matches!(self, Self::DecodePending)
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum BatchPhase {
    #[default]
    Mixed,
    Prefill,
    Decode,
}

#[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum BatchSchedulingMode {
    #[default]
    Fifo,
    Priority,
    Deadline,
    Fairness,
    LatencyTarget,
    ThroughputTarget,
    DecodePriority,
    PrefillPriority,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BatchingPolicy {
    pub mode: BatchSchedulingMode,
    pub max_queue_time_millis: Option<u64>,
    pub max_active_operations: usize,
    pub max_batch_tokens: Option<usize>,
    pub max_batch_sequences: usize,
    pub allow_prefill: bool,
    pub allow_decode: bool,
    pub allow_queueing: bool,
    pub prevent_starvation: bool,
    pub weighted_fairness_enabled: bool,
    pub browser_feature_required: bool,
}

impl Default for BatchingPolicy {
    fn default() -> Self {
        Self {
            mode: BatchSchedulingMode::Fifo,
            max_queue_time_millis: None,
            max_active_operations: 1024,
            max_batch_tokens: None,
            max_batch_sequences: 32,
            allow_prefill: true,
            allow_decode: true,
            allow_queueing: true,
            prevent_starvation: true,
            weighted_fairness_enabled: false,
            browser_feature_required: false,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BatchCompatibility {
    pub model: GenerationModelReference,
    pub model_context: Option<String>,
    pub architecture: Option<String>,
    pub compute_dtype: Option<DTypeDescriptor>,
    pub tokenizer: TokenizerId,
    pub provider: Option<ProviderBinding>,
    pub device: Option<DeviceBinding>,
    pub affinity: Option<ResourceAffinity>,
    pub kv_cache_layout: Option<KvCacheLayoutFormat>,
    pub max_sequence_length: Option<usize>,
    pub sampling_policy: Option<SamplingPolicy>,
    pub memory_placement: Option<MemoryPlacement>,
    pub provider_assisted_sampling: bool,
}

impl BatchCompatibility {
    pub fn from_generation(request: &GenerationRequest) -> Self {
        Self {
            model: request.model.clone(),
            model_context: None,
            architecture: None,
            compute_dtype: None,
            tokenizer: request.tokenizer.tokenizer_id.clone(),
            provider: None,
            device: None,
            affinity: None,
            kv_cache_layout: None,
            max_sequence_length: request.model_context_length,
            sampling_policy: None,
            memory_placement: Some(request.memory.placement.clone()),
            provider_assisted_sampling: false,
        }
    }

    pub fn validate_with(&self, other: &Self) -> Result<(), BatchingError> {
        if self.model != other.model || self.model_context != other.model_context {
            return Err(BatchingError::ModelIncompatible);
        }
        if self.architecture != other.architecture {
            return Err(BatchingError::BatchCompatibilityFailed {
                reason: "model architecture differs".into(),
            });
        }
        if self.compute_dtype != other.compute_dtype {
            return Err(BatchingError::BatchCompatibilityFailed {
                reason: "compute dtype differs".into(),
            });
        }
        if self.tokenizer != other.tokenizer {
            return Err(BatchingError::TokenizerIncompatible);
        }
        if self.provider != other.provider {
            return Err(BatchingError::ProviderUnavailable);
        }
        if self.device != other.device {
            return Err(BatchingError::DeviceUnavailable);
        }
        if self.kv_cache_layout != other.kv_cache_layout {
            return Err(BatchingError::KvCacheIncompatible);
        }
        if self.provider_assisted_sampling && self.sampling_policy != other.sampling_policy {
            return Err(BatchingError::BatchCompatibilityFailed {
                reason: "provider-assisted sampling policy differs".into(),
            });
        }
        if let (Some(left), Some(right)) = (&self.affinity, &other.affinity) {
            left.validate_with(right)
                .map_err(|_| BatchingError::ResourceAffinityConflict)?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BatchMemoryEstimate {
    pub input_buffer_bytes: u64,
    pub output_buffer_bytes: u64,
    pub logits_buffer_bytes: u64,
    pub attention_mask_bytes: u64,
    pub position_buffer_bytes: u64,
    pub sampling_workspace_bytes: u64,
    pub kv_cache_block_bytes: u64,
    pub prefix_cache_lookup_bytes: u64,
    pub temporary_staging_bytes: u64,
    pub provider_workspace_bytes: u64,
    pub placement: MemoryPlacement,
    pub queue_allowed: bool,
}

impl Default for BatchMemoryEstimate {
    fn default() -> Self {
        Self {
            input_buffer_bytes: 0,
            output_buffer_bytes: 0,
            logits_buffer_bytes: 0,
            attention_mask_bytes: 0,
            position_buffer_bytes: 0,
            sampling_workspace_bytes: 0,
            kv_cache_block_bytes: 0,
            prefix_cache_lookup_bytes: 0,
            temporary_staging_bytes: 0,
            provider_workspace_bytes: 0,
            placement: MemoryPlacement::HostOrdinary,
            queue_allowed: false,
        }
    }
}

impl BatchMemoryEstimate {
    pub fn total_bytes(&self) -> Result<u64, BatchingError> {
        [
            self.input_buffer_bytes,
            self.output_buffer_bytes,
            self.logits_buffer_bytes,
            self.attention_mask_bytes,
            self.position_buffer_bytes,
            self.sampling_workspace_bytes,
            self.kv_cache_block_bytes,
            self.prefix_cache_lookup_bytes,
            self.temporary_staging_bytes,
            self.provider_workspace_bytes,
        ]
        .into_iter()
        .try_fold(0u64, |total, value| {
            total
                .checked_add(value)
                .ok_or(BatchingError::InternalBatching)
        })
    }

    pub fn admission_request(
        &self,
        batch: &BatchId,
    ) -> Result<MemoryAdmissionRequest, BatchingError> {
        let mut allocation = MemoryAllocationRequest::new(
            MemoryAllocationClass::TemporaryWorkspace,
            self.total_bytes()?,
            self.placement.clone(),
            MemoryAllocationOwner::Session(batch.as_str().into()),
        );
        allocation.lifetime = MemoryAllocationLifetime::Operation;
        Ok(MemoryAdmissionRequest {
            allocation,
            pressure: Default::default(),
            queue_allowed: self.queue_allowed,
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BatchSlot {
    pub id: BatchSlotId,
    pub batch: BatchId,
    pub operation: GenerationRequestId,
    pub session: Option<InferenceSessionId>,
    pub compatibility: BatchCompatibility,
    pub state: BatchedOperationState,
    pub current_sequence_length: usize,
    pub generated_token_count: usize,
    pub kv_cache: Option<KvCacheId>,
    pub prefix_cache_entry: Option<PrefixCacheEntryId>,
    pub prefix_reuse_boundary_tokens: Option<usize>,
    pub provider: Option<ProviderBinding>,
    pub device: Option<DeviceBinding>,
    pub memory_reservation: Option<MemoryAllocationId>,
    pub priority: GenerationPriority,
    pub deadline_millis: Option<u64>,
    pub cancellation_requested: bool,
    pub streaming_backpressure: bool,
}

impl BatchSlot {
    pub fn transition_to(&mut self, next: BatchedOperationState) -> Result<(), BatchingError> {
        if self.state.allows_transition_to(next) {
            self.state = next;
            Ok(())
        } else if self.state == next || self.state.is_terminal() && next == self.state {
            Ok(())
        } else {
            Err(BatchingError::InvalidOperationState {
                from: self.state,
                to: next,
            })
        }
    }

    pub fn record_streamed_token(&mut self, token_index: usize) -> Result<(), BatchingError> {
        if token_index != self.generated_token_count {
            return Err(BatchingError::StreamingBackpressure {
                reason: "streaming token index would break per-operation order".into(),
            });
        }
        self.generated_token_count = self.generated_token_count.saturating_add(1);
        self.current_sequence_length = self.current_sequence_length.saturating_add(1);
        Ok(())
    }

    pub fn is_compatible_with(&self, other: &Self) -> Result<(), BatchingError> {
        self.compatibility.validate_with(&other.compatibility)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BatchAdmission {
    pub request_id: GenerationRequestId,
    pub session: Option<InferenceSessionId>,
    pub compatibility: BatchCompatibility,
    pub prompt_token_count: usize,
    pub generated_token_count: usize,
    pub kv_cache: Option<KvCacheId>,
    pub prefix_cache_entry: Option<PrefixCacheEntryId>,
    pub provider: Option<ProviderBinding>,
    pub device: Option<DeviceBinding>,
    pub memory_reservation: Option<MemoryAllocationId>,
    pub priority: GenerationPriority,
    pub deadline_millis: Option<u64>,
    pub cancellation_requested: bool,
}

impl BatchAdmission {
    pub fn from_generation(request: &GenerationRequest) -> Self {
        Self {
            request_id: request.request_id.clone(),
            session: request.session.clone(),
            compatibility: BatchCompatibility::from_generation(request),
            prompt_token_count: request.prompt_token_count,
            generated_token_count: 0,
            kv_cache: None,
            prefix_cache_entry: None,
            provider: None,
            device: None,
            memory_reservation: None,
            priority: request.priority.clone(),
            deadline_millis: request.priority.deadline_millis,
            cancellation_requested: request.cancellation.requested,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BatchExecutionStep {
    pub batch: BatchId,
    pub phase: BatchPhase,
    pub slots: Vec<BatchSlotId>,
    pub total_sequence_tokens: usize,
    pub provider: Option<ProviderBinding>,
    pub device: Option<DeviceBinding>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BatchObservationKind {
    OperationAdmitted,
    OperationRejected,
    OperationQueued,
    BatchFormed,
    BatchResized,
    PrefillScheduled,
    DecodeScheduled,
    BatchSubmitted,
    BatchCompleted,
    OperationCompleted,
    OperationCancelled,
    OperationFailed,
    QueuePressure,
    MemoryPressure,
    ProviderPressure,
    DevicePressure,
    PrefixCacheHit,
    KvCacheAssigned,
    StreamingBackpressure,
    FairnessAdjustment,
    StarvationPrevented,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BatchObservation {
    pub kind: BatchObservationKind,
    pub batch: Option<BatchId>,
    pub slot: Option<BatchSlotId>,
    pub message: String,
    pub raw_prompt_available: bool,
    pub raw_logits_available: bool,
    pub raw_kv_cache_available: bool,
    pub raw_provider_handle_available: bool,
}

impl BatchObservation {
    pub fn redacted(
        kind: BatchObservationKind,
        batch: Option<BatchId>,
        slot: Option<BatchSlotId>,
        message: impl Into<String>,
    ) -> Self {
        Self {
            kind,
            batch,
            slot,
            message: message.into(),
            raw_prompt_available: false,
            raw_logits_available: false,
            raw_kv_cache_available: false,
            raw_provider_handle_available: false,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ContinuousBatch {
    pub id: BatchId,
    pub phase: BatchPhase,
    pub policy: BatchingPolicy,
    pub slots: BTreeMap<BatchSlotId, BatchSlot>,
    pub provider_pressure: Option<ProviderPressureLevel>,
    pub device_pressure: Option<MemoryPressureLevel>,
    pub memory_pressure: Option<MemoryPressureLevel>,
    pub shutdown: bool,
}

impl ContinuousBatch {
    pub fn active_slot_count(&self) -> usize {
        self.slots
            .values()
            .filter(|slot| !slot.state.is_terminal())
            .count()
    }

    pub fn compatible_anchor(&self) -> Option<&BatchCompatibility> {
        self.slots
            .values()
            .find(|slot| !slot.state.is_terminal())
            .map(|slot| &slot.compatibility)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ContinuousBatchingManager {
    next_batch_id: u64,
    next_slot_id: u64,
    batches: BTreeMap<BatchId, ContinuousBatch>,
    observations: Vec<BatchObservation>,
}

impl Default for ContinuousBatchingManager {
    fn default() -> Self {
        Self::new()
    }
}

impl ContinuousBatchingManager {
    pub fn new() -> Self {
        Self {
            next_batch_id: 1,
            next_slot_id: 1,
            batches: BTreeMap::new(),
            observations: Vec::new(),
        }
    }

    pub fn batches(&self) -> impl Iterator<Item = &ContinuousBatch> {
        self.batches.values()
    }

    pub fn observations(&self) -> &[BatchObservation] {
        &self.observations
    }

    pub fn create_batch(&mut self, policy: BatchingPolicy) -> BatchId {
        let id = BatchId::runtime_issued(self.next_batch_id);
        self.next_batch_id = self.next_batch_id.saturating_add(1);
        self.batches.insert(
            id.clone(),
            ContinuousBatch {
                id: id.clone(),
                phase: BatchPhase::Mixed,
                policy,
                slots: BTreeMap::new(),
                provider_pressure: None,
                device_pressure: None,
                memory_pressure: None,
                shutdown: false,
            },
        );
        self.observe(
            BatchObservationKind::BatchFormed,
            Some(id.clone()),
            None,
            "continuous batch created",
        );
        id
    }

    pub fn batch(&self, id: &BatchId) -> Result<&ContinuousBatch, BatchingError> {
        self.batches.get(id).ok_or(BatchingError::OperationNotFound)
    }

    pub fn batch_mut(&mut self, id: &BatchId) -> Result<&mut ContinuousBatch, BatchingError> {
        self.batches
            .get_mut(id)
            .ok_or(BatchingError::OperationNotFound)
    }

    pub fn admit_operation(
        &mut self,
        batch: &BatchId,
        admission: BatchAdmission,
    ) -> Result<BatchSlotId, BatchingError> {
        let id = BatchSlotId::runtime_issued(batch, self.next_slot_id);
        self.next_slot_id = self.next_slot_id.saturating_add(1);
        let queued = {
            let batch_state = self.batch(batch)?;
            if batch_state.shutdown {
                return Err(BatchingError::RuntimeShutdown);
            }
            if batch_state.active_slot_count() >= batch_state.policy.max_active_operations {
                return Err(BatchingError::SessionConcurrencyLimit);
            }
            if batch_state.slots.len() >= batch_state.policy.max_batch_sequences {
                return Err(BatchingError::QueueFull);
            }
            if let Some(anchor) = batch_state.compatible_anchor() {
                anchor.validate_with(&admission.compatibility)?;
            }
            batch_state.policy.allow_queueing
        };
        if admission.cancellation_requested {
            return Err(BatchingError::OperationCancelled);
        }
        let slot = BatchSlot {
            id: id.clone(),
            batch: batch.clone(),
            operation: admission.request_id,
            session: admission.session,
            compatibility: admission.compatibility,
            state: if queued {
                BatchedOperationState::Queued
            } else {
                BatchedOperationState::Admitted
            },
            current_sequence_length: admission.prompt_token_count,
            generated_token_count: admission.generated_token_count,
            kv_cache: admission.kv_cache,
            prefix_cache_entry: admission.prefix_cache_entry,
            prefix_reuse_boundary_tokens: None,
            provider: admission.provider,
            device: admission.device,
            memory_reservation: admission.memory_reservation,
            priority: admission.priority,
            deadline_millis: admission.deadline_millis,
            cancellation_requested: false,
            streaming_backpressure: false,
        };
        self.batch_mut(batch)?.slots.insert(id.clone(), slot);
        self.observe(
            BatchObservationKind::OperationQueued,
            Some(batch.clone()),
            Some(id.clone()),
            "operation admitted to continuous batch queue",
        );
        Ok(id)
    }

    pub fn slot(&self, id: &BatchSlotId) -> Result<&BatchSlot, BatchingError> {
        self.batches
            .values()
            .find_map(|batch| batch.slots.get(id))
            .ok_or(BatchingError::OperationNotFound)
    }

    pub fn slot_mut(&mut self, id: &BatchSlotId) -> Result<&mut BatchSlot, BatchingError> {
        self.batches
            .values_mut()
            .find_map(|batch| batch.slots.get_mut(id))
            .ok_or(BatchingError::OperationNotFound)
    }

    pub fn transition_slot(
        &mut self,
        id: &BatchSlotId,
        next: BatchedOperationState,
    ) -> Result<(), BatchingError> {
        let (batch, kind) = {
            let slot = self.slot_mut(id)?;
            slot.transition_to(next)?;
            (
                slot.batch.clone(),
                match next {
                    BatchedOperationState::Cancelled => BatchObservationKind::OperationCancelled,
                    BatchedOperationState::Completed => BatchObservationKind::OperationCompleted,
                    BatchedOperationState::Failed => BatchObservationKind::OperationFailed,
                    _ => BatchObservationKind::BatchResized,
                },
            )
        };
        self.observe(
            kind,
            Some(batch),
            Some(id.clone()),
            "batch slot state changed",
        );
        Ok(())
    }

    pub fn schedule_prefill(
        &mut self,
        batch: &BatchId,
        max_slots: usize,
    ) -> Result<BatchExecutionStep, BatchingError> {
        self.schedule(batch, BatchPhase::Prefill, max_slots)
    }

    pub fn schedule_decode(
        &mut self,
        batch: &BatchId,
        max_slots: usize,
    ) -> Result<BatchExecutionStep, BatchingError> {
        self.schedule(batch, BatchPhase::Decode, max_slots)
    }

    pub fn memory_admission_request(
        &self,
        batch: &BatchId,
        estimate: &BatchMemoryEstimate,
    ) -> Result<MemoryAdmissionRequest, BatchingError> {
        self.batch(batch)?;
        estimate.admission_request(batch)
    }

    pub fn assign_kv_cache(
        &mut self,
        slot: &BatchSlotId,
        cache: KvCacheId,
    ) -> Result<(), BatchingError> {
        let batch = {
            let slot_state = self.slot_mut(slot)?;
            slot_state.kv_cache = Some(cache);
            slot_state.batch.clone()
        };
        self.observe(
            BatchObservationKind::KvCacheAssigned,
            Some(batch),
            Some(slot.clone()),
            "kv cache assigned through Runtime-owned reference",
        );
        Ok(())
    }

    pub fn apply_prefix_cache_hit(
        &mut self,
        slot: &BatchSlotId,
        entry: PrefixCacheEntryId,
        boundary_tokens: usize,
    ) -> Result<(), BatchingError> {
        let batch = {
            let slot_state = self.slot_mut(slot)?;
            slot_state.prefix_cache_entry = Some(entry);
            slot_state.prefix_reuse_boundary_tokens = Some(boundary_tokens);
            slot_state.batch.clone()
        };
        self.observe(
            BatchObservationKind::PrefixCacheHit,
            Some(batch),
            Some(slot.clone()),
            "prefix cache hit applied to batch slot",
        );
        Ok(())
    }

    pub fn record_streamed_token(
        &mut self,
        slot: &BatchSlotId,
        token_index: usize,
    ) -> Result<(), BatchingError> {
        let batch = {
            let slot_state = self.slot_mut(slot)?;
            let result = slot_state.record_streamed_token(token_index);
            let batch = slot_state.batch.clone();
            if result.is_err() {
                slot_state.streaming_backpressure = true;
            }
            result?;
            batch
        };
        self.observe(
            BatchObservationKind::BatchResized,
            Some(batch),
            Some(slot.clone()),
            "streamed token order preserved",
        );
        Ok(())
    }

    pub fn cancel_slot(&mut self, slot: &BatchSlotId) -> Result<(), BatchingError> {
        self.transition_slot(slot, BatchedOperationState::Cancelled)
    }

    pub fn fail_slot(&mut self, slot: &BatchSlotId) -> Result<(), BatchingError> {
        self.transition_slot(slot, BatchedOperationState::Failed)
    }

    pub fn remove_terminal_slots(
        &mut self,
        batch: &BatchId,
    ) -> Result<Vec<BatchSlotId>, BatchingError> {
        let removed = {
            let batch_state = self.batch_mut(batch)?;
            let ids = batch_state
                .slots
                .iter()
                .filter_map(|(id, slot)| slot.state.is_terminal().then_some(id.clone()))
                .collect::<Vec<_>>();
            for id in &ids {
                batch_state.slots.remove(id);
            }
            ids
        };
        if !removed.is_empty() {
            self.observe(
                BatchObservationKind::BatchResized,
                Some(batch.clone()),
                None,
                "terminal operations removed from batch slots",
            );
        }
        Ok(removed)
    }

    fn schedule(
        &mut self,
        batch: &BatchId,
        phase: BatchPhase,
        max_slots: usize,
    ) -> Result<BatchExecutionStep, BatchingError> {
        if max_slots == 0 {
            return Err(BatchingError::BatchSizeUnsupported);
        }
        let (ids, total_sequence_tokens, provider, device) = {
            let batch_state = self.batch_mut(batch)?;
            if phase == BatchPhase::Prefill && !batch_state.policy.allow_prefill {
                return Err(BatchingError::SchedulingPolicyDenied);
            }
            if phase == BatchPhase::Decode && !batch_state.policy.allow_decode {
                return Err(BatchingError::SchedulingPolicyDenied);
            }
            if matches!(
                batch_state.provider_pressure,
                Some(ProviderPressureLevel::Saturated)
            ) {
                return Err(BatchingError::ProviderSaturated);
            }
            if matches!(
                batch_state.memory_pressure,
                Some(MemoryPressureLevel::Saturated)
            ) {
                return Err(BatchingError::MemoryAdmissionFailed);
            }
            let target = match phase {
                BatchPhase::Prefill => BatchedOperationState::Prefilling,
                BatchPhase::Decode => BatchedOperationState::Decoding,
                BatchPhase::Mixed => BatchedOperationState::Prefilling,
            };
            let mut ids = Vec::new();
            let mut total_sequence_tokens = 0usize;
            let mut provider = None;
            let mut device = None;
            for slot in batch_state.slots.values_mut() {
                let eligible = match phase {
                    BatchPhase::Prefill => slot.state.is_prefill_ready(),
                    BatchPhase::Decode => slot.state.is_decode_ready(),
                    BatchPhase::Mixed => {
                        slot.state.is_prefill_ready() || slot.state.is_decode_ready()
                    }
                };
                if !eligible || ids.len() >= max_slots {
                    continue;
                }
                total_sequence_tokens =
                    total_sequence_tokens.saturating_add(slot.current_sequence_length);
                if let Some(max_tokens) = batch_state.policy.max_batch_tokens
                    && total_sequence_tokens > max_tokens
                {
                    total_sequence_tokens =
                        total_sequence_tokens.saturating_sub(slot.current_sequence_length);
                    continue;
                }
                if phase == BatchPhase::Prefill && slot.state == BatchedOperationState::Queued {
                    slot.transition_to(BatchedOperationState::PrefillPending)?;
                }
                slot.transition_to(target)?;
                provider = provider.or_else(|| slot.provider.clone());
                device = device.or_else(|| slot.device.clone());
                ids.push(slot.id.clone());
            }
            if ids.is_empty() {
                return Err(BatchingError::QueueFull);
            }
            batch_state.phase = phase;
            (ids, total_sequence_tokens, provider, device)
        };
        self.observe(
            if phase == BatchPhase::Decode {
                BatchObservationKind::DecodeScheduled
            } else {
                BatchObservationKind::PrefillScheduled
            },
            Some(batch.clone()),
            None,
            "batch execution step scheduled",
        );
        Ok(BatchExecutionStep {
            batch: batch.clone(),
            phase,
            slots: ids,
            total_sequence_tokens,
            provider,
            device,
        })
    }

    fn observe(
        &mut self,
        kind: BatchObservationKind,
        batch: Option<BatchId>,
        slot: Option<BatchSlotId>,
        message: impl Into<String>,
    ) {
        self.observations
            .push(BatchObservation::redacted(kind, batch, slot, message));
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum BatchingErrorCode {
    BatchUnavailable,
    BatchAdmissionRejected,
    QueueFull,
    OperationNotFound,
    OperationCancelled,
    OperationTimedOut,
    SessionConcurrencyLimit,
    ModelIncompatible,
    TokenizerIncompatible,
    ProviderUnavailable,
    ProviderNotReady,
    ProviderSaturated,
    DeviceUnavailable,
    DeviceMemoryInsufficient,
    MemoryAdmissionFailed,
    ResourceAffinityConflict,
    KvCacheUnavailable,
    KvCacheIncompatible,
    PrefixCacheReuseDenied,
    BatchCompatibilityFailed,
    BatchSizeUnsupported,
    SequenceLengthUnsupported,
    StreamingBackpressure,
    SchedulingPolicyDenied,
    RuntimeShutdown,
    BrowserFeatureUnsupported,
    InternalBatching,
    InvalidOperationState,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum BatchingError {
    BatchUnavailable,
    BatchAdmissionRejected {
        reason: String,
    },
    QueueFull,
    OperationNotFound,
    OperationCancelled,
    OperationTimedOut,
    SessionConcurrencyLimit,
    ModelIncompatible,
    TokenizerIncompatible,
    ProviderUnavailable,
    ProviderNotReady,
    ProviderSaturated,
    DeviceUnavailable,
    DeviceMemoryInsufficient,
    MemoryAdmissionFailed,
    ResourceAffinityConflict,
    KvCacheUnavailable,
    KvCacheIncompatible,
    PrefixCacheReuseDenied,
    BatchCompatibilityFailed {
        reason: String,
    },
    BatchSizeUnsupported,
    SequenceLengthUnsupported,
    StreamingBackpressure {
        reason: String,
    },
    SchedulingPolicyDenied,
    RuntimeShutdown,
    BrowserFeatureUnsupported,
    InternalBatching,
    InvalidOperationState {
        from: BatchedOperationState,
        to: BatchedOperationState,
    },
}

impl BatchingError {
    pub const fn code(&self) -> BatchingErrorCode {
        match self {
            Self::BatchUnavailable => BatchingErrorCode::BatchUnavailable,
            Self::BatchAdmissionRejected { .. } => BatchingErrorCode::BatchAdmissionRejected,
            Self::QueueFull => BatchingErrorCode::QueueFull,
            Self::OperationNotFound => BatchingErrorCode::OperationNotFound,
            Self::OperationCancelled => BatchingErrorCode::OperationCancelled,
            Self::OperationTimedOut => BatchingErrorCode::OperationTimedOut,
            Self::SessionConcurrencyLimit => BatchingErrorCode::SessionConcurrencyLimit,
            Self::ModelIncompatible => BatchingErrorCode::ModelIncompatible,
            Self::TokenizerIncompatible => BatchingErrorCode::TokenizerIncompatible,
            Self::ProviderUnavailable => BatchingErrorCode::ProviderUnavailable,
            Self::ProviderNotReady => BatchingErrorCode::ProviderNotReady,
            Self::ProviderSaturated => BatchingErrorCode::ProviderSaturated,
            Self::DeviceUnavailable => BatchingErrorCode::DeviceUnavailable,
            Self::DeviceMemoryInsufficient => BatchingErrorCode::DeviceMemoryInsufficient,
            Self::MemoryAdmissionFailed => BatchingErrorCode::MemoryAdmissionFailed,
            Self::ResourceAffinityConflict => BatchingErrorCode::ResourceAffinityConflict,
            Self::KvCacheUnavailable => BatchingErrorCode::KvCacheUnavailable,
            Self::KvCacheIncompatible => BatchingErrorCode::KvCacheIncompatible,
            Self::PrefixCacheReuseDenied => BatchingErrorCode::PrefixCacheReuseDenied,
            Self::BatchCompatibilityFailed { .. } => BatchingErrorCode::BatchCompatibilityFailed,
            Self::BatchSizeUnsupported => BatchingErrorCode::BatchSizeUnsupported,
            Self::SequenceLengthUnsupported => BatchingErrorCode::SequenceLengthUnsupported,
            Self::StreamingBackpressure { .. } => BatchingErrorCode::StreamingBackpressure,
            Self::SchedulingPolicyDenied => BatchingErrorCode::SchedulingPolicyDenied,
            Self::RuntimeShutdown => BatchingErrorCode::RuntimeShutdown,
            Self::BrowserFeatureUnsupported => BatchingErrorCode::BrowserFeatureUnsupported,
            Self::InternalBatching => BatchingErrorCode::InternalBatching,
            Self::InvalidOperationState { .. } => BatchingErrorCode::InvalidOperationState,
        }
    }
}

impl fmt::Display for BatchingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BatchAdmissionRejected { reason }
            | Self::BatchCompatibilityFailed { reason }
            | Self::StreamingBackpressure { reason } => f.write_str(reason),
            Self::InvalidOperationState { from, to } => {
                write!(
                    f,
                    "invalid batching lifecycle transition from {from:?} to {to:?}"
                )
            }
            other => write!(f, "{:?}", other.code()),
        }
    }
}

impl Error for BatchingError {}

fn validate_batch_identity(value: &str, label: &str) -> Result<(), BatchingError> {
    if value.trim().is_empty()
        || value.contains('/')
        || value.contains('\\')
        || value.contains(':')
        || value.contains("provider")
        || value.contains("device")
        || value.contains("memory")
        || value.contains("0x")
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'-' | b'_'))
    {
        return Err(BatchingError::BatchAdmissionRejected {
            reason: format!("{label} must be an opaque portable Runtime-issued identifier"),
        });
    }
    Ok(())
}

pub fn portable_batch_dtype(dtype: ComputeDType) -> DTypeDescriptor {
    DTypeDescriptor::portable(dtype)
}
