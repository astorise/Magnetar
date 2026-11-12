//! Identities of Model Instances, Inference Sessions, Adapters and
//! Tokenizers, with the error types their constructors return.
//!
//! Model-data layer (`quality/layers.toml`): the kernel pipeline and the
//! Component engine name these types without depending on Model Instance
//! lifecycle, session, adapter or tokenizer code. Each type is re-exported
//! from the module where it was first defined, so existing paths are
//! unchanged.

use crate::compute_types::ComputeDType;
use crate::model::{ModelArchitecture, ModelDType};
use std::{error::Error, fmt};

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct InferenceSessionId(String);

impl InferenceSessionId {
    pub fn new(value: impl Into<String>) -> Result<Self, SessionError> {
        let value = value.into();
        validate_session_identity(&value)?;
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for InferenceSessionId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SessionError {
    SessionCreationFailed {
        reason: String,
    },
    SessionNotFound,
    SessionNotReady,
    SessionActive,
    SessionClosed,
    SessionExpired,
    SessionCancelled,
    SessionDraining,
    SessionPolicyDenied {
        reason: String,
    },
    ModelUnavailable {
        reason: String,
    },
    TokenizerIncompatible {
        reason: String,
    },
    MemoryAdmissionFailed {
        reason: String,
    },
    MemoryBudgetExceeded {
        reason: String,
    },
    GenerationFailed {
        reason: String,
    },
    StreamingFailed {
        reason: String,
    },
    CancellationFailed {
        reason: String,
    },
    OperationQueued,
    OperationRejected {
        reason: String,
    },
    ConcurrencyViolation,
    ResourceCleanupFailed {
        reason: String,
    },
    RuntimeShutdown,
    Unauthorized,
    InvalidLifecycleTransition {
        from: SessionLifecycleState,
        to: SessionLifecycleState,
    },
    InternalSession {
        reason: String,
    },
}

impl fmt::Display for SessionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SessionCreationFailed { reason } => {
                write!(f, "session creation failed: {reason}")
            }
            Self::SessionNotFound => f.write_str("session not found"),
            Self::SessionNotReady => f.write_str("session not ready"),
            Self::SessionActive => f.write_str("session active"),
            Self::SessionClosed => f.write_str("session closed"),
            Self::SessionExpired => f.write_str("session expired"),
            Self::SessionCancelled => f.write_str("session cancelled"),
            Self::SessionDraining => f.write_str("session draining"),
            Self::SessionPolicyDenied { reason } => write!(f, "session policy denied: {reason}"),
            Self::ModelUnavailable { reason } => write!(f, "model unavailable: {reason}"),
            Self::TokenizerIncompatible { reason } => write!(f, "tokenizer incompatible: {reason}"),
            Self::MemoryAdmissionFailed { reason } => {
                write!(f, "session memory admission failed: {reason}")
            }
            Self::MemoryBudgetExceeded { reason } => {
                write!(f, "session memory budget exceeded: {reason}")
            }
            Self::GenerationFailed { reason } => write!(f, "session generation failed: {reason}"),
            Self::StreamingFailed { reason } => write!(f, "session streaming failed: {reason}"),
            Self::CancellationFailed { reason } => {
                write!(f, "session cancellation failed: {reason}")
            }
            Self::OperationQueued => f.write_str("session operation queued"),
            Self::OperationRejected { reason } => write!(f, "session operation rejected: {reason}"),
            Self::ConcurrencyViolation => f.write_str("session concurrency violation"),
            Self::ResourceCleanupFailed { reason } => {
                write!(f, "session resource cleanup failed: {reason}")
            }
            Self::RuntimeShutdown => f.write_str("runtime shutdown"),
            Self::Unauthorized => f.write_str("session access unauthorized"),
            Self::InvalidLifecycleTransition { from, to } => {
                write!(
                    f,
                    "invalid session lifecycle transition from {from:?} to {to:?}"
                )
            }
            Self::InternalSession { reason } => write!(f, "internal session error: {reason}"),
        }
    }
}

impl Error for SessionError {}

pub(crate) fn validate_session_identity(value: &str) -> Result<(), SessionError> {
    if value.trim().is_empty()
        || value.contains('/')
        || value.contains('\\')
        || value.contains(':')
        || value.contains("provider")
        || value.contains("device")
        || value.contains("0x")
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'-' | b'_'))
    {
        return Err(SessionError::SessionCreationFailed {
            reason: "session id must be opaque, portable, and free of raw handles".into(),
        });
    }
    Ok(())
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct AdapterRevision(String);

impl AdapterRevision {
    pub fn new(value: impl Into<String>) -> Result<Self, AdapterError> {
        let value = value.into();
        validate_portable_identity(&value, "adapter revision")?;
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum AdapterError {
    AdapterArtifactNotFound,
    AdapterArtifactInvalid {
        reason: String,
    },
    AdapterArtifactUntrusted,
    AdapterArtifactRevoked,
    AdapterMethodUnsupported {
        method: String,
    },
    BaseModelIncompatible,
    ArchitectureIncompatible,
    TargetModuleMissing {
        target: String,
    },
    TargetTensorMismatch {
        target: String,
    },
    TokenizerIncompatible,
    StorageDTypeUnsupported {
        dtype: ModelDType,
    },
    ComputeDTypeUnsupported {
        dtype: ComputeDType,
    },
    QuantizationUnsupported,
    AdapterRankUnsupported {
        rank: u32,
    },
    AdapterShapeMismatch {
        tensor: String,
    },
    MemoryFeasibilityFailed {
        reason: String,
    },
    MemoryAllocationFailed {
        reason: String,
    },
    AdapterLoadingQueued,
    AdapterLoadingTimeout,
    ProviderCapabilityUnavailable,
    ProviderAdapterUnsupported,
    ProviderNotReady,
    ProviderSaturated,
    DeviceUnavailable,
    DeviceMemoryInsufficient,
    ActivationDenied,
    ActivationConflict,
    MultipleAdaptersUnsupported,
    MergeUnsupported,
    MergeFailed,
    UnmergeUnsupported,
    UnmergeFailed,
    KvCacheIncompatible,
    PrefixCacheInvalidated,
    UnloadFailed,
    BrowserFeatureUnsupported,
    InvalidLifecycleTransition {
        from: AdapterLifecycleState,
        to: AdapterLifecycleState,
    },
    InternalAdapter {
        reason: String,
    },
}

impl fmt::Display for AdapterError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::AdapterArtifactNotFound => f.write_str("adapter artifact not found"),
            Self::AdapterArtifactInvalid { reason } => {
                write!(f, "adapter artifact invalid: {reason}")
            }
            Self::AdapterArtifactUntrusted => f.write_str("adapter artifact untrusted"),
            Self::AdapterArtifactRevoked => f.write_str("adapter artifact revoked"),
            Self::AdapterMethodUnsupported { method } => {
                write!(f, "adapter method unsupported: {method}")
            }
            Self::BaseModelIncompatible => f.write_str("base model incompatible"),
            Self::ArchitectureIncompatible => f.write_str("architecture incompatible"),
            Self::TargetModuleMissing { target } => write!(f, "target module missing: {target}"),
            Self::TargetTensorMismatch { target } => write!(f, "target tensor mismatch: {target}"),
            Self::TokenizerIncompatible => f.write_str("tokenizer incompatible"),
            Self::StorageDTypeUnsupported { dtype } => {
                write!(f, "storage dtype unsupported: {dtype:?}")
            }
            Self::ComputeDTypeUnsupported { dtype } => {
                write!(f, "compute dtype unsupported: {dtype:?}")
            }
            Self::QuantizationUnsupported => f.write_str("adapter quantization unsupported"),
            Self::AdapterRankUnsupported { rank } => {
                write!(f, "adapter rank unsupported: {rank}")
            }
            Self::AdapterShapeMismatch { tensor } => {
                write!(f, "adapter shape mismatch: {tensor}")
            }
            Self::MemoryFeasibilityFailed { reason } => {
                write!(f, "adapter memory feasibility failed: {reason}")
            }
            Self::MemoryAllocationFailed { reason } => {
                write!(f, "adapter memory allocation failed: {reason}")
            }
            Self::AdapterLoadingQueued => f.write_str("adapter loading queued"),
            Self::AdapterLoadingTimeout => f.write_str("adapter loading timeout"),
            Self::ProviderCapabilityUnavailable => f.write_str("Provider capability unavailable"),
            Self::ProviderAdapterUnsupported => f.write_str("Provider adapter unsupported"),
            Self::ProviderNotReady => f.write_str("Provider not ready"),
            Self::ProviderSaturated => f.write_str("Provider saturated"),
            Self::DeviceUnavailable => f.write_str("Device unavailable"),
            Self::DeviceMemoryInsufficient => f.write_str("Device memory insufficient"),
            Self::ActivationDenied => f.write_str("adapter activation denied"),
            Self::ActivationConflict => f.write_str("adapter activation conflict"),
            Self::MultipleAdaptersUnsupported => f.write_str("multiple adapters unsupported"),
            Self::MergeUnsupported => f.write_str("adapter merge unsupported"),
            Self::MergeFailed => f.write_str("adapter merge failed"),
            Self::UnmergeUnsupported => f.write_str("adapter unmerge unsupported"),
            Self::UnmergeFailed => f.write_str("adapter unmerge failed"),
            Self::KvCacheIncompatible => f.write_str("KV cache incompatible"),
            Self::PrefixCacheInvalidated => f.write_str("Prefix Cache invalidated"),
            Self::UnloadFailed => f.write_str("adapter unload failed"),
            Self::BrowserFeatureUnsupported => f.write_str("browser adapter feature unsupported"),
            Self::InvalidLifecycleTransition { from, to } => {
                write!(
                    f,
                    "invalid adapter lifecycle transition from {from:?} to {to:?}"
                )
            }
            Self::InternalAdapter { reason } => write!(f, "internal adapter error: {reason}"),
        }
    }
}

impl Error for AdapterError {}

#[doc(hidden)]
pub fn validate_portable_identity(value: &str, label: &str) -> Result<(), AdapterError> {
    if value.trim().is_empty()
        || value.contains('/')
        || value.contains('\\')
        || value.contains(':')
        || value.to_ascii_lowercase().contains("provider")
        || value.to_ascii_lowercase().contains("device")
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'-' | b'_'))
    {
        return Err(AdapterError::AdapterArtifactInvalid {
            reason: format!("{label} must use a portable non-Provider, non-Device identifier"),
        });
    }
    Ok(())
}

pub type TokenId = u32;

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct TokenizerFamily(String);

impl TokenizerFamily {
    pub fn new(value: impl Into<String>) -> Result<Self, TokenizerError> {
        let value = value.into();
        validate_identity(&value, "tokenizer family")?;
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

#[doc(hidden)]
pub fn validate_identity(value: &str, label: &str) -> Result<(), TokenizerError> {
    if value.trim().is_empty()
        || value.contains('/')
        || value.contains('\\')
        || value.contains(':')
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'-' | b'_'))
    {
        return Err(TokenizerError::InvalidTokenizerArtifact {
            artifact: format!("{label}:{value}"),
        });
    }
    Ok(())
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TokenizerError {
    TokenizerArtifactMissing { artifact: String },
    InvalidTokenizerArtifact { artifact: String },
    TokenizerIncompatibleWithModel,
    UnsupportedTokenizerFamily { family: String },
    InvalidTokenId { token_id: TokenId },
    UnknownToken { token: String },
    InvalidUtf8 { message: String },
    DecodePendingPartial,
    OffsetsUnsupported,
    PaddingTokenMissing,
    TruncationRequired { token_count: usize, limit: usize },
    TruncationForbidden,
    PromptTooLong { token_count: usize, limit: usize },
    BatchInputInvalid { message: String },
    SpecialTokenMissing,
    SpecialTokenConflict,
    VocabularyMismatch { expected: u32, actual: u32 },
    AddedTokenMismatch { expected: u32, actual: u32 },
    MemoryAllocationFailed,
    StreamingStateInvalid,
    ImplementationUnavailable,
    UnsupportedTokenTypeIds,
}

impl fmt::Display for TokenizerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TokenizerArtifactMissing { artifact } => {
                write!(f, "tokenizer artifact '{artifact}' is missing")
            }
            Self::InvalidTokenizerArtifact { artifact } => {
                write!(f, "tokenizer artifact '{artifact}' is invalid")
            }
            Self::TokenizerIncompatibleWithModel => {
                f.write_str("tokenizer is incompatible with model artifact")
            }
            Self::UnsupportedTokenizerFamily { family } => {
                write!(f, "unsupported tokenizer family '{family}'")
            }
            Self::InvalidTokenId { token_id } => write!(f, "invalid token id {token_id}"),
            Self::UnknownToken { token } => write!(f, "unknown token '{token}'"),
            Self::InvalidUtf8 { message } => write!(f, "invalid utf-8 during decode: {message}"),
            Self::DecodePendingPartial => f.write_str("decode has pending partial output"),
            Self::OffsetsUnsupported => f.write_str("token offsets are unsupported"),
            Self::PaddingTokenMissing => f.write_str("padding token is missing"),
            Self::TruncationRequired { token_count, limit } => write!(
                f,
                "truncation is required for {token_count} tokens with limit {limit}"
            ),
            Self::TruncationForbidden => f.write_str("truncation is forbidden"),
            Self::PromptTooLong { token_count, limit } => {
                write!(f, "prompt has {token_count} tokens, limit is {limit}")
            }
            Self::BatchInputInvalid { message } => write!(f, "batch input invalid: {message}"),
            Self::SpecialTokenMissing => f.write_str("required special token is missing"),
            Self::SpecialTokenConflict => f.write_str("special token conflict"),
            Self::VocabularyMismatch { expected, actual } => {
                write!(
                    f,
                    "vocabulary mismatch: expected {expected}, actual {actual}"
                )
            }
            Self::AddedTokenMismatch { expected, actual } => {
                write!(
                    f,
                    "added token mismatch: expected {expected}, actual {actual}"
                )
            }
            Self::MemoryAllocationFailed => f.write_str("tokenizer memory allocation failed"),
            Self::StreamingStateInvalid => f.write_str("streaming decode state is invalid"),
            Self::ImplementationUnavailable => f.write_str("tokenizer implementation unavailable"),
            Self::UnsupportedTokenTypeIds => f.write_str("token type ids are unsupported"),
        }
    }
}

impl Error for TokenizerError {}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ModelInstanceId(String);

impl ModelInstanceId {
    pub fn new(value: impl Into<String>) -> Result<Self, ModelInstanceError> {
        let value = value.into();
        validate_instance_identity(&value)?;
        Ok(Self(value))
    }

    #[doc(hidden)]
    pub fn runtime_issued(sequence: u64) -> Self {
        Self(format!("model-instance-{sequence}"))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for ModelInstanceId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum ModelInstanceLifecycleState {
    Creating,
    Loading,
    Warming,
    Ready,
    Active,
    Idle,
    Draining,
    Suspended,
    Reloading,
    Unloading,
    Unloaded,
    Failed,
    Invalid,
    Removed,
}

impl ModelInstanceLifecycleState {
    pub const fn is_terminal(self) -> bool {
        matches!(
            self,
            Self::Unloaded | Self::Failed | Self::Invalid | Self::Removed
        )
    }

    /// Whether this lifecycle state legitimately permits accepting
    /// inference usage. `readiness == Ready` alone SHALL NOT be trusted as
    /// sufficient (Correctif: Runtime-owned ModelInstance readiness
    /// authority) -- a caller-forged or internally inconsistent
    /// `readiness` value must not grant usage while the instance's own
    /// lifecycle has not actually reached one of these states.
    pub const fn supports_inference_use(self) -> bool {
        matches!(self, Self::Ready | Self::Idle | Self::Active)
    }

    pub const fn allows_transition_to(self, next: Self) -> bool {
        matches!(
            (self, next),
            (Self::Creating, Self::Loading)
                | (Self::Creating, Self::Failed)
                | (Self::Creating, Self::Invalid)
                | (Self::Loading, Self::Warming)
                | (Self::Loading, Self::Ready)
                | (Self::Loading, Self::Failed)
                | (Self::Warming, Self::Ready)
                | (Self::Warming, Self::Failed)
                | (Self::Ready, Self::Active)
                | (Self::Ready, Self::Idle)
                | (Self::Ready, Self::Draining)
                | (Self::Ready, Self::Suspended)
                | (Self::Ready, Self::Reloading)
                | (Self::Ready, Self::Unloading)
                | (Self::Ready, Self::Failed)
                | (Self::Ready, Self::Invalid)
                | (Self::Active, Self::Idle)
                | (Self::Active, Self::Draining)
                | (Self::Active, Self::Failed)
                | (Self::Idle, Self::Active)
                | (Self::Idle, Self::Draining)
                | (Self::Idle, Self::Suspended)
                | (Self::Idle, Self::Reloading)
                | (Self::Idle, Self::Unloading)
                | (Self::Idle, Self::Failed)
                | (Self::Idle, Self::Invalid)
                | (Self::Draining, Self::Unloading)
                | (Self::Draining, Self::Reloading)
                | (Self::Draining, Self::Failed)
                | (Self::Suspended, Self::Loading)
                | (Self::Suspended, Self::Reloading)
                | (Self::Suspended, Self::Unloading)
                | (Self::Suspended, Self::Failed)
                | (Self::Reloading, Self::Loading)
                | (Self::Reloading, Self::Ready)
                | (Self::Reloading, Self::Failed)
                | (Self::Unloading, Self::Unloaded)
                | (Self::Unloading, Self::Failed)
                | (Self::Unloaded, Self::Removed)
                | (Self::Failed, Self::Unloading)
                | (Self::Failed, Self::Invalid)
                | (Self::Invalid, Self::Unloading)
                | (Self::Invalid, Self::Removed)
        )
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ModelInstanceWarmupPolicy {
    Disabled,
    ValidateMetadataOnly,
    ProviderInitialization,
    Full,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ModelInstanceWarmupStep {
    ProviderInitialization,
    KernelPreparationPlaceholder,
    OperatorGraphPreparationPlaceholder,
    ShapePlanPreparationPlaceholder,
    TokenizerModelMetadataValidation,
    SmallTestExecutionPlaceholder,
    MemoryResidencyVerification,
    AdapterReadinessVerification,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ModelInstanceWarmupPlan {
    pub policy: ModelInstanceWarmupPolicy,
    pub steps: Vec<ModelInstanceWarmupStep>,
}

impl ModelInstanceWarmupPlan {
    pub fn for_policy(policy: ModelInstanceWarmupPolicy) -> Self {
        let steps = match policy {
            ModelInstanceWarmupPolicy::Disabled => Vec::new(),
            ModelInstanceWarmupPolicy::ValidateMetadataOnly => {
                vec![ModelInstanceWarmupStep::TokenizerModelMetadataValidation]
            }
            ModelInstanceWarmupPolicy::ProviderInitialization => vec![
                ModelInstanceWarmupStep::ProviderInitialization,
                ModelInstanceWarmupStep::TokenizerModelMetadataValidation,
                ModelInstanceWarmupStep::MemoryResidencyVerification,
            ],
            ModelInstanceWarmupPolicy::Full => vec![
                ModelInstanceWarmupStep::ProviderInitialization,
                ModelInstanceWarmupStep::KernelPreparationPlaceholder,
                ModelInstanceWarmupStep::OperatorGraphPreparationPlaceholder,
                ModelInstanceWarmupStep::ShapePlanPreparationPlaceholder,
                ModelInstanceWarmupStep::TokenizerModelMetadataValidation,
                ModelInstanceWarmupStep::SmallTestExecutionPlaceholder,
                ModelInstanceWarmupStep::MemoryResidencyVerification,
                ModelInstanceWarmupStep::AdapterReadinessVerification,
            ],
        };
        Self { policy, steps }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ModelInstanceError {
    ModelInstanceNotFound,
    ModelInstanceNotReady,
    ModelInstanceLoading,
    ModelInstanceWarming,
    ModelInstanceDraining,
    ModelInstanceSuspended,
    ModelInstanceUnloading,
    ModelInstanceUnloaded,
    ModelInstanceFailed,
    ModelInstanceInvalid,
    ModelInstanceRemoved,
    ModelInstanceActive,
    ModelInstanceBusy,
    ModelInstanceSharingDenied,
    ModelInstancePolicyDenied,
    ModelInstanceReloadRequired,
    ModelInstanceReloadFailed,
    ModelInstanceUnloadFailed,
    ModelInstanceWarmupFailed,
    ModelInstanceProviderUnavailable,
    ModelInstanceProviderNotReady,
    ModelInstanceProviderFailed,
    ModelInstanceDeviceUnavailable,
    ModelInstanceDeviceLost,
    ModelInstanceMemoryPressure,
    ModelInstanceResidencyMissing,
    ModelInstanceAdapterIncompatible,
    ModelInstanceKernelPreparationFailed,
    ModelInstanceAutotuningIncomplete,
    ModelInstanceWeightsNotMaterialized,
    ModelInstanceKvCacheInvalidated,
    ModelInstancePrefixCacheInvalidated,
    ModelInstanceBrowserFeatureUnsupported,
    InvalidLifecycleTransition {
        from: ModelInstanceLifecycleState,
        to: ModelInstanceLifecycleState,
    },
    /// The caller-supplied architecture implementation's architecture
    /// identity disagrees with the one the loading phase already resolved
    /// for this artifact (`seal-runtime-model-trust-and-provenance-authority`).
    ArchitectureMismatch {
        expected: ModelArchitecture,
        actual: ModelArchitecture,
    },
    /// The caller-supplied Resource Affinity names a provider or device
    /// that disagrees with one the loading phase already resolved for this
    /// artifact (`seal-runtime-model-trust-and-provenance-authority`).
    AffinityMismatch {
        reason: String,
    },
    InternalModelInstance {
        reason: String,
    },
}

impl fmt::Display for ModelInstanceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ModelInstanceNotFound => f.write_str("model instance not found"),
            Self::ModelInstanceNotReady => f.write_str("model instance not ready"),
            Self::ModelInstanceLoading => f.write_str("model instance loading"),
            Self::ModelInstanceWarming => f.write_str("model instance warming"),
            Self::ModelInstanceDraining => f.write_str("model instance draining"),
            Self::ModelInstanceSuspended => f.write_str("model instance suspended"),
            Self::ModelInstanceUnloading => f.write_str("model instance unloading"),
            Self::ModelInstanceUnloaded => f.write_str("model instance unloaded"),
            Self::ModelInstanceFailed => f.write_str("model instance failed"),
            Self::ModelInstanceInvalid => f.write_str("model instance invalid"),
            Self::ModelInstanceRemoved => f.write_str("model instance removed"),
            Self::ModelInstanceActive => f.write_str("model instance active"),
            Self::ModelInstanceBusy => f.write_str("model instance busy"),
            Self::ModelInstanceSharingDenied => f.write_str("model instance sharing denied"),
            Self::ModelInstancePolicyDenied => f.write_str("model instance policy denied"),
            Self::ModelInstanceReloadRequired => f.write_str("model instance reload required"),
            Self::ModelInstanceReloadFailed => f.write_str("model instance reload failed"),
            Self::ModelInstanceUnloadFailed => f.write_str("model instance unload failed"),
            Self::ModelInstanceWarmupFailed => f.write_str("model instance warmup failed"),
            Self::ModelInstanceProviderUnavailable => {
                f.write_str("model instance Provider unavailable")
            }
            Self::ModelInstanceProviderNotReady => f.write_str("model instance Provider not ready"),
            Self::ModelInstanceProviderFailed => f.write_str("model instance Provider failed"),
            Self::ModelInstanceDeviceUnavailable => {
                f.write_str("model instance Device unavailable")
            }
            Self::ModelInstanceDeviceLost => f.write_str("model instance Device lost"),
            Self::ModelInstanceMemoryPressure => f.write_str("model instance memory pressure"),
            Self::ModelInstanceResidencyMissing => f.write_str("model instance residency missing"),
            Self::ModelInstanceAdapterIncompatible => {
                f.write_str("model instance adapter incompatible")
            }
            Self::ModelInstanceKernelPreparationFailed => {
                f.write_str("model instance required Kernel preparation failed")
            }
            Self::ModelInstanceAutotuningIncomplete => {
                f.write_str("model instance required Kernel Autotuning incomplete")
            }
            Self::ModelInstanceWeightsNotMaterialized => {
                f.write_str("model instance weights not materialized")
            }
            Self::ModelInstanceKvCacheInvalidated => {
                f.write_str("model instance KV cache invalidated")
            }
            Self::ModelInstancePrefixCacheInvalidated => {
                f.write_str("model instance Prefix Cache invalidated")
            }
            Self::ModelInstanceBrowserFeatureUnsupported => {
                f.write_str("model instance browser feature unsupported")
            }
            Self::InvalidLifecycleTransition { from, to } => {
                write!(
                    f,
                    "invalid model instance transition from {from:?} to {to:?}"
                )
            }
            Self::ArchitectureMismatch { expected, actual } => {
                write!(
                    f,
                    "model instance architecture {actual:?} disagrees with the loaded artifact's resolved architecture {expected:?}"
                )
            }
            Self::AffinityMismatch { reason } => {
                write!(f, "model instance affinity mismatch: {reason}")
            }
            Self::InternalModelInstance { reason } => {
                write!(f, "internal model instance: {reason}")
            }
        }
    }
}

impl Error for ModelInstanceError {}

pub(crate) fn validate_instance_identity(value: &str) -> Result<(), ModelInstanceError> {
    let lower = value.to_ascii_lowercase();
    if value.trim().is_empty()
        || value.contains('/')
        || value.contains('\\')
        || value.contains(':')
        || value.contains("0x")
        || lower.contains("provider")
        || lower.contains("device")
        || lower.contains("ptr")
        || lower.contains("weight")
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'-' | b'_'))
    {
        return Err(ModelInstanceError::ModelInstancePolicyDenied);
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SessionLifecycleState {
    Creating,
    Ready,
    Active,
    Idle,
    Draining,
    Cancelled,
    Failed,
    Closed,
    Expired,
}

impl SessionLifecycleState {
    pub const fn is_terminal(self) -> bool {
        matches!(
            self,
            Self::Cancelled | Self::Failed | Self::Closed | Self::Expired
        )
    }

    pub const fn allows_transition_to(self, next: Self) -> bool {
        matches!(
            (self, next),
            (Self::Creating, Self::Ready)
                | (Self::Creating, Self::Failed)
                | (Self::Ready, Self::Active)
                | (Self::Ready, Self::Draining)
                | (Self::Ready, Self::Closed)
                | (Self::Ready, Self::Expired)
                | (Self::Active, Self::Idle)
                | (Self::Active, Self::Draining)
                | (Self::Active, Self::Cancelled)
                | (Self::Active, Self::Failed)
                | (Self::Idle, Self::Active)
                | (Self::Idle, Self::Draining)
                | (Self::Idle, Self::Closed)
                | (Self::Idle, Self::Expired)
                | (Self::Draining, Self::Closed)
                | (Self::Draining, Self::Expired)
                | (Self::Cancelled, Self::Closed)
                | (Self::Failed, Self::Closed)
        )
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AdapterLifecycleState {
    Requested,
    Validating,
    Planning,
    Allocating,
    Materializing,
    Ready,
    Active,
    Inactive,
    Merging,
    Merged,
    Unmerging,
    Draining,
    Unloading,
    Unloaded,
    Failed,
    Invalid,
}

impl AdapterLifecycleState {
    pub const fn allows_transition_to(self, next: Self) -> bool {
        matches!(
            (self, next),
            (Self::Requested, Self::Validating)
                | (Self::Requested, Self::Failed)
                | (Self::Validating, Self::Planning)
                | (Self::Validating, Self::Invalid)
                | (Self::Validating, Self::Failed)
                | (Self::Planning, Self::Allocating)
                | (Self::Planning, Self::Failed)
                | (Self::Allocating, Self::Materializing)
                | (Self::Allocating, Self::Failed)
                | (Self::Materializing, Self::Ready)
                | (Self::Materializing, Self::Failed)
                | (Self::Ready, Self::Active)
                | (Self::Ready, Self::Inactive)
                | (Self::Ready, Self::Merging)
                | (Self::Ready, Self::Draining)
                | (Self::Active, Self::Inactive)
                | (Self::Active, Self::Draining)
                | (Self::Active, Self::Merging)
                | (Self::Inactive, Self::Active)
                | (Self::Inactive, Self::Unloading)
                | (Self::Merging, Self::Merged)
                | (Self::Merging, Self::Failed)
                | (Self::Merged, Self::Unmerging)
                | (Self::Merged, Self::Draining)
                | (Self::Unmerging, Self::Ready)
                | (Self::Unmerging, Self::Failed)
                | (Self::Draining, Self::Unloading)
                | (Self::Unloading, Self::Unloaded)
                | (Self::Unloading, Self::Failed)
        )
    }
}
