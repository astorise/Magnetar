//! Portable execution stream and synchronization contract.
//!
//! Runtime owns logical dependency semantics. Providers may realize those
//! dependencies with native queues, streams, events, fences, or worker pools,
//! but those native objects stay outside this public contract.

use crate::{
    DeviceBinding, MemoryAllocationId, PreparedExecutionSegmentId, PreparedKernelId,
    ProviderBinding, TensorResourceId,
};
use std::{collections::BTreeSet, error::Error, fmt};

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ExecutionStreamId {
    value: u64,
    generation: u64,
}

impl ExecutionStreamId {
    pub const fn new(value: u64, generation: u64) -> Self {
        Self { value, generation }
    }

    pub const fn value(self) -> u64 {
        self.value
    }

    pub const fn generation(self) -> u64 {
        self.generation
    }
}

impl fmt::Display for ExecutionStreamId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "execution-stream:{}:{}", self.value, self.generation)
    }
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ExecutionStreamClass(String);

impl ExecutionStreamClass {
    pub fn new(value: impl Into<String>) -> Result<Self, ExecutionSynchronizationError> {
        let value = value.into();
        validate_logical_name(&value, "execution stream class")?;
        Ok(Self(value))
    }

    pub fn compute() -> Self {
        Self("magnetar:execution/compute".into())
    }

    pub fn transfer() -> Self {
        Self("magnetar:execution/transfer".into())
    }

    pub fn control() -> Self {
        Self("magnetar:execution/control".into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    pub fn exposes_native_queue_type(&self) -> bool {
        let lowered = self.0.to_ascii_lowercase();
        [
            "cuda",
            "custream",
            "hipstream",
            "mtlcommandqueue",
            "vkqueue",
            "webgpuqueue",
        ]
        .iter()
        .any(|needle| lowered.contains(needle))
    }
}

impl fmt::Display for ExecutionStreamClass {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum ExecutionPriorityHint {
    Background,
    #[default]
    Normal,
    LatencySensitive,
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum ExecutionStreamState {
    Creating,
    Ready,
    Draining,
    Failed,
    Closed,
}

impl ExecutionStreamState {
    pub const fn can_transition_to(self, next: Self) -> bool {
        matches!(
            (self, next),
            (Self::Creating, Self::Ready)
                | (Self::Creating, Self::Failed)
                | (Self::Ready, Self::Draining)
                | (Self::Ready, Self::Failed)
                | (Self::Draining, Self::Closed)
                | (Self::Draining, Self::Failed)
                | (Self::Failed, Self::Closed)
        )
    }

    pub const fn accepts_new_work(self) -> bool {
        matches!(self, Self::Ready)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExecutionStream {
    pub id: ExecutionStreamId,
    pub class: ExecutionStreamClass,
    pub provider: ProviderBinding,
    pub device: Option<DeviceBinding>,
    pub priority: ExecutionPriorityHint,
    pub state: ExecutionStreamState,
}

impl ExecutionStream {
    pub fn new(
        id: ExecutionStreamId,
        class: ExecutionStreamClass,
        provider: ProviderBinding,
    ) -> Result<Self, ExecutionSynchronizationError> {
        if class.exposes_native_queue_type() {
            return Err(ExecutionSynchronizationError::NativeSynchronizationLeak);
        }
        Ok(Self {
            id,
            class,
            provider,
            device: None,
            priority: ExecutionPriorityHint::Normal,
            state: ExecutionStreamState::Creating,
        })
    }

    pub fn with_device(mut self, device: DeviceBinding) -> Self {
        self.device = Some(device);
        self
    }

    pub const fn with_priority(mut self, priority: ExecutionPriorityHint) -> Self {
        self.priority = priority;
        self
    }

    pub fn transition_to(
        &mut self,
        next: ExecutionStreamState,
    ) -> Result<(), ExecutionSynchronizationError> {
        if self.state.can_transition_to(next) {
            self.state = next;
            Ok(())
        } else {
            Err(ExecutionSynchronizationError::InvalidStreamState)
        }
    }

    pub fn ensure_submittable(&self) -> Result<(), ExecutionSynchronizationError> {
        if self.state.accepts_new_work() {
            Ok(())
        } else {
            Err(match self.state {
                ExecutionStreamState::Creating => ExecutionSynchronizationError::StreamNotReady,
                ExecutionStreamState::Draining => ExecutionSynchronizationError::StreamDraining,
                ExecutionStreamState::Failed => ExecutionSynchronizationError::StreamFailed,
                ExecutionStreamState::Closed => ExecutionSynchronizationError::StreamClosed,
                ExecutionStreamState::Ready => unreachable!("ready streams accept work"),
            })
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct CompletionTokenId {
    value: u64,
    generation: u64,
}

impl CompletionTokenId {
    pub const fn new(value: u64, generation: u64) -> Self {
        Self { value, generation }
    }

    pub const fn value(self) -> u64 {
        self.value
    }

    pub const fn generation(self) -> u64 {
        self.generation
    }
}

impl fmt::Display for CompletionTokenId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "completion-token:{}:{}", self.value, self.generation)
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum CompletionTokenState {
    Pending,
    Completed,
    Failed,
    Cancelled,
    Lost,
}

impl CompletionTokenState {
    pub const fn is_terminal(self) -> bool {
        matches!(
            self,
            Self::Completed | Self::Failed | Self::Cancelled | Self::Lost
        )
    }

    pub const fn satisfies_dependency(self) -> bool {
        matches!(self, Self::Completed)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CompletionScope {
    Kernel(PreparedKernelId),
    Transfer,
    PreparedSegment(PreparedExecutionSegmentId),
    GroupedSubmission,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CompletionToken {
    pub id: CompletionTokenId,
    pub stream: ExecutionStreamId,
    pub provider: ProviderBinding,
    pub device: Option<DeviceBinding>,
    pub scope: CompletionScope,
    pub state: CompletionTokenState,
    pub failure_reason: Option<String>,
    retained_dependencies: usize,
}

impl CompletionToken {
    pub fn pending(
        id: CompletionTokenId,
        stream: &ExecutionStream,
        scope: CompletionScope,
    ) -> Self {
        Self {
            id,
            stream: stream.id,
            provider: stream.provider.clone(),
            device: stream.device.clone(),
            scope,
            state: CompletionTokenState::Pending,
            failure_reason: None,
            retained_dependencies: 0,
        }
    }

    pub fn completed(
        id: CompletionTokenId,
        stream: &ExecutionStream,
        scope: CompletionScope,
    ) -> Self {
        let mut token = Self::pending(id, stream, scope);
        token.state = CompletionTokenState::Completed;
        token
    }

    pub fn transition_to(
        &mut self,
        next: CompletionTokenState,
    ) -> Result<(), ExecutionSynchronizationError> {
        if self.state.is_terminal() {
            return Err(ExecutionSynchronizationError::CompletionAlreadyTerminal);
        }
        if !next.is_terminal() {
            return Err(ExecutionSynchronizationError::CompletionNotTerminal);
        }
        self.state = next;
        Ok(())
    }

    pub fn mark_failed(
        &mut self,
        reason: impl Into<String>,
    ) -> Result<(), ExecutionSynchronizationError> {
        self.transition_to(CompletionTokenState::Failed)?;
        self.failure_reason = Some(reason.into());
        Ok(())
    }

    pub fn retain_for_dependency(&mut self) {
        self.retained_dependencies = self.retained_dependencies.saturating_add(1);
    }

    pub fn release_dependency(&mut self) {
        self.retained_dependencies = self.retained_dependencies.saturating_sub(1);
    }

    pub const fn can_release_provider_state(&self) -> bool {
        self.state.is_terminal() && self.retained_dependencies == 0
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExecutionDependency {
    pub predecessors: Vec<CompletionTokenId>,
    pub provider: ProviderBinding,
    pub device: Option<DeviceBinding>,
    pub allow_runtime_mediation: bool,
}

impl ExecutionDependency {
    pub fn new(
        provider: ProviderBinding,
        predecessors: impl IntoIterator<Item = CompletionTokenId>,
    ) -> Result<Self, ExecutionSynchronizationError> {
        let predecessors = predecessors.into_iter().collect::<Vec<_>>();
        if predecessors.is_empty() {
            return Err(ExecutionSynchronizationError::DependencyInvalid);
        }
        ensure_unique_tokens(&predecessors)?;
        Ok(Self {
            predecessors,
            provider,
            device: None,
            allow_runtime_mediation: true,
        })
    }

    pub fn with_device(mut self, device: DeviceBinding) -> Self {
        self.device = Some(device);
        self
    }

    pub fn validate_against_tokens(
        &self,
        tokens: impl IntoIterator<Item = CompletionToken>,
    ) -> Result<DependencyReadiness, ExecutionSynchronizationError> {
        let mut pending = Vec::new();
        for token in tokens {
            if !self.predecessors.contains(&token.id) {
                return Err(ExecutionSynchronizationError::DependencyInvalid);
            }
            if token.provider != self.provider && !self.allow_runtime_mediation {
                return Err(ExecutionSynchronizationError::CrossProviderUnsupported);
            }
            if !token.state.satisfies_dependency() {
                if token.state.is_terminal() {
                    return Err(ExecutionSynchronizationError::DependencyFailed);
                }
                pending.push(token.id);
            }
        }
        Ok(if pending.is_empty() {
            DependencyReadiness::Ready
        } else {
            DependencyReadiness::Pending(pending)
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum DependencyReadiness {
    Ready,
    Pending(Vec<CompletionTokenId>),
}

impl DependencyReadiness {
    pub const fn is_ready(&self) -> bool {
        matches!(self, Self::Ready)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ExecutionSubmissionTarget {
    PreparedKernel(PreparedKernelId),
    PreparedSegment(PreparedExecutionSegmentId),
    Transfer,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExecutionSubmission {
    pub stream: ExecutionStream,
    pub target: ExecutionSubmissionTarget,
    pub resources: Vec<TensorResourceId>,
    pub dependencies: Vec<ExecutionDependency>,
    pub deadline_millis: Option<u64>,
    pub cancellation_scope: Option<String>,
}

impl ExecutionSubmission {
    pub fn new(stream: ExecutionStream, target: ExecutionSubmissionTarget) -> Self {
        Self {
            stream,
            target,
            resources: Vec::new(),
            dependencies: Vec::new(),
            deadline_millis: None,
            cancellation_scope: None,
        }
    }

    pub fn validate(&self) -> Result<(), ExecutionSynchronizationError> {
        self.stream.ensure_submittable()?;
        for dependency in &self.dependencies {
            if dependency.provider != self.stream.provider && !dependency.allow_runtime_mediation {
                return Err(ExecutionSynchronizationError::CrossProviderUnsupported);
            }
            if let (Some(expected), Some(found)) = (&self.stream.device, &dependency.device)
                && expected != found
            {
                return Err(ExecutionSynchronizationError::DeviceMismatch);
            }
        }
        Ok(())
    }

    pub fn requires_predecessor(&self, token: CompletionTokenId) -> bool {
        self.dependencies
            .iter()
            .any(|dependency| dependency.predecessors.contains(&token))
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum ResourceAccessScope {
    HostRead,
    DeviceRead,
    DeviceWrite,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResourceReadiness {
    pub resource: TensorResourceId,
    pub writer: Option<CompletionTokenId>,
    pub host_ready: bool,
    pub device_ready: bool,
    pub access_scope: ResourceAccessScope,
}

impl ResourceReadiness {
    pub fn ready(resource: TensorResourceId, access_scope: ResourceAccessScope) -> Self {
        Self {
            resource,
            writer: None,
            host_ready: true,
            device_ready: true,
            access_scope,
        }
    }

    pub fn pending_write(resource: TensorResourceId, writer: CompletionTokenId) -> Self {
        Self {
            resource,
            writer: Some(writer),
            host_ready: false,
            device_ready: false,
            access_scope: ResourceAccessScope::DeviceWrite,
        }
    }

    pub const fn blocks_host_read(&self) -> bool {
        !self.host_ready
    }

    pub const fn blocks_device_consumer(&self) -> bool {
        !self.device_ready
    }

    pub fn mark_completed_by(
        &mut self,
        token: CompletionTokenId,
    ) -> Result<(), ExecutionSynchronizationError> {
        if self.writer != Some(token) {
            return Err(ExecutionSynchronizationError::DependencyInvalid);
        }
        self.host_ready = true;
        self.device_ready = true;
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MemoryReuseFence {
    pub allocation: MemoryAllocationId,
    pub active_tokens: BTreeSet<CompletionTokenId>,
}

impl MemoryReuseFence {
    pub fn new(allocation: MemoryAllocationId) -> Self {
        Self {
            allocation,
            active_tokens: BTreeSet::new(),
        }
    }

    pub fn retain(mut self, token: CompletionTokenId) -> Self {
        self.active_tokens.insert(token);
        self
    }

    pub fn release(&mut self, token: CompletionTokenId) {
        self.active_tokens.remove(&token);
    }

    pub fn is_reusable(&self) -> bool {
        self.active_tokens.is_empty()
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum ProviderCancellationLevel {
    NotSupported,
    BeforeSubmitOnly,
    QueuedWork,
    Cooperative,
    Interruptible,
    ProviderSpecific,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProviderSynchronizationCapability {
    pub asynchronous_submission: bool,
    pub ordered_streams: bool,
    pub cross_stream_dependencies: bool,
    pub device_side_dependencies: bool,
    pub host_wait: bool,
    pub non_blocking_poll: bool,
    pub transfer_overlap: bool,
    pub priority: bool,
    pub deadlines: bool,
    pub multi_device_dependencies: bool,
    pub cancellation: ProviderCancellationLevel,
}

impl ProviderSynchronizationCapability {
    pub fn synchronous_baseline() -> Self {
        Self {
            asynchronous_submission: false,
            ordered_streams: true,
            cross_stream_dependencies: false,
            device_side_dependencies: false,
            host_wait: true,
            non_blocking_poll: true,
            transfer_overlap: false,
            priority: false,
            deadlines: false,
            multi_device_dependencies: false,
            cancellation: ProviderCancellationLevel::NotSupported,
        }
    }

    pub fn async_capable() -> Self {
        Self {
            asynchronous_submission: true,
            ordered_streams: true,
            cross_stream_dependencies: true,
            device_side_dependencies: true,
            host_wait: true,
            non_blocking_poll: true,
            transfer_overlap: true,
            priority: false,
            deadlines: false,
            multi_device_dependencies: false,
            cancellation: ProviderCancellationLevel::BeforeSubmitOnly,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum SynchronizationObservationKind {
    StreamCreated,
    StreamDraining,
    StreamClosed,
    SubmissionCreated,
    SubmissionStarted,
    SubmissionCompleted,
    SubmissionFailed,
    DependencyWait,
    CompletionPending,
    CompletionCompleted,
    CompletionFailed,
    ResourceReadinessUpdated,
    MemoryReuseDelayed,
    CancellationRequested,
    CancellationDeferred,
    DeadlineExceeded,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SynchronizationObservation {
    pub kind: SynchronizationObservationKind,
    pub stream: Option<ExecutionStreamId>,
    pub completion: Option<CompletionTokenId>,
    pub provider: Option<ProviderBinding>,
    pub device: Option<DeviceBinding>,
    pub message: String,
}

impl SynchronizationObservation {
    pub fn new(kind: SynchronizationObservationKind, message: impl AsRef<str>) -> Self {
        Self {
            kind,
            stream: None,
            completion: None,
            provider: None,
            device: None,
            message: redact_native_details(message.as_ref()),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ExecutionSynchronizationError {
    StreamUnavailable,
    StreamCreateFailed,
    StreamNotReady,
    StreamDraining,
    StreamFailed,
    StreamClosed,
    ProviderMismatch,
    DeviceMismatch,
    PriorityUnsupported,
    SubmissionInvalid,
    SubmissionFailed,
    DependencyInvalid,
    DependencyCycle,
    DependencyFailed,
    CrossProviderUnsupported,
    CompletionInvalid,
    CompletionFailed,
    CompletionLost,
    CompletionTimeout,
    CompletionAlreadyReleased,
    CompletionAlreadyTerminal,
    CompletionNotTerminal,
    ResourceNotReady,
    ResourceWriteConflict,
    ResourceReadConflict,
    ResourceAffinityInvalid,
    CancellationUnsupported,
    CancellationTooLate,
    CancellationFailed,
    DeadlineExceeded,
    ProviderSynchronizationFailed,
    DeviceLost,
    NativeSynchronizationLeak,
    InvalidStreamState,
}

impl fmt::Display for ExecutionSynchronizationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "execution synchronization error: {self:?}")
    }
}

impl Error for ExecutionSynchronizationError {}

pub fn same_stream_ordered(
    first: &ExecutionSubmission,
    second: &ExecutionSubmission,
) -> Result<bool, ExecutionSynchronizationError> {
    first.validate()?;
    second.validate()?;
    Ok(first.stream.id == second.stream.id)
}

pub fn cross_stream_ordered_by_dependency(
    producer: CompletionTokenId,
    consumer: &ExecutionSubmission,
) -> bool {
    consumer.requires_predecessor(producer)
}

pub fn cancellation_preserves_physical_lifetime(token: &CompletionToken) -> bool {
    matches!(
        token.state,
        CompletionTokenState::Pending | CompletionTokenState::Completed
    )
}

pub fn redact_native_details(value: &str) -> String {
    let lowered = value.to_ascii_lowercase();
    if [
        "custream",
        "cudaevent",
        "vkqueue",
        "vksemaphore",
        "mtlcommandqueue",
        "handle=0x",
        "0xdead",
        "native pointer",
    ]
    .iter()
    .any(|needle| lowered.contains(needle))
    {
        "[redacted]".into()
    } else {
        value.into()
    }
}

fn ensure_unique_tokens(tokens: &[CompletionTokenId]) -> Result<(), ExecutionSynchronizationError> {
    let unique = tokens.iter().collect::<BTreeSet<_>>();
    if unique.len() == tokens.len() {
        Ok(())
    } else {
        Err(ExecutionSynchronizationError::DependencyCycle)
    }
}

fn validate_logical_name(value: &str, name: &str) -> Result<(), ExecutionSynchronizationError> {
    if value.trim().is_empty() {
        return Err(ExecutionSynchronizationError::SubmissionInvalid);
    }
    let lowered = value.to_ascii_lowercase();
    if [
        "custream",
        "cuda stream",
        "cudaevent",
        "cuda event",
        "hipstream",
        "mtlcommandqueue",
        "vkqueue",
        "vksemaphore",
        "handle",
        "*",
    ]
    .iter()
    .any(|token| lowered.contains(token))
    {
        return Err(ExecutionSynchronizationError::NativeSynchronizationLeak);
    }
    let _ = name;
    Ok(())
}
