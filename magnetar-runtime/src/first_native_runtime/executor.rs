//! Model executor contract and first-native error model.

use super::*;

/// Structured E2E conformance error categories. Every variant maps to a
/// stable `e2e-*` code via [`E2eConformanceError::code`].
#[derive(Clone, Debug, Eq, PartialEq, Deserialize, Serialize)]
pub enum E2eConformanceError {
    SuiteUnavailable { reason: String },
    FixtureInvalid { reason: String },
    ModelResolutionFailed { reason: String },
    ModelLoadingFailed { reason: String },
    ModelComponentFailed { reason: String },
    TokenizerFailed { reason: String },
    SessionFailed { reason: String },
    GenerationFailed { reason: String },
    SamplingFailed { reason: String },
    StreamingFailed { reason: String },
    GraphValidationFailed { reason: String },
    OperatorCoverageMissing { reason: String },
    KernelCoverageMissing { reason: String },
    MemoryValidationFailed { reason: String },
    RedactionFailed { reason: String },
    BoundaryViolation { reason: String },
    DeterminismFailed { reason: String },
    Internal { reason: String },
}

impl E2eConformanceError {
    pub const fn code(&self) -> &'static str {
        match self {
            Self::SuiteUnavailable { .. } => "e2e-suite-unavailable",
            Self::FixtureInvalid { .. } => "e2e-fixture-invalid",
            Self::ModelResolutionFailed { .. } => "e2e-model-resolution-failed",
            Self::ModelLoadingFailed { .. } => "e2e-model-loading-failed",
            Self::ModelComponentFailed { .. } => "e2e-model-component-failed",
            Self::TokenizerFailed { .. } => "e2e-tokenizer-failed",
            Self::SessionFailed { .. } => "e2e-session-failed",
            Self::GenerationFailed { .. } => "e2e-generation-failed",
            Self::SamplingFailed { .. } => "e2e-sampling-failed",
            Self::StreamingFailed { .. } => "e2e-streaming-failed",
            Self::GraphValidationFailed { .. } => "e2e-graph-validation-failed",
            Self::OperatorCoverageMissing { .. } => "e2e-operator-coverage-missing",
            Self::KernelCoverageMissing { .. } => "e2e-kernel-coverage-missing",
            Self::MemoryValidationFailed { .. } => "e2e-memory-validation-failed",
            Self::RedactionFailed { .. } => "e2e-redaction-failed",
            Self::BoundaryViolation { .. } => "e2e-boundary-violation",
            Self::DeterminismFailed { .. } => "e2e-determinism-failed",
            Self::Internal { .. } => "internal-e2e-conformance",
        }
    }

    pub fn reason(&self) -> &str {
        match self {
            Self::SuiteUnavailable { reason }
            | Self::FixtureInvalid { reason }
            | Self::ModelResolutionFailed { reason }
            | Self::ModelLoadingFailed { reason }
            | Self::ModelComponentFailed { reason }
            | Self::TokenizerFailed { reason }
            | Self::SessionFailed { reason }
            | Self::GenerationFailed { reason }
            | Self::SamplingFailed { reason }
            | Self::StreamingFailed { reason }
            | Self::GraphValidationFailed { reason }
            | Self::OperatorCoverageMissing { reason }
            | Self::KernelCoverageMissing { reason }
            | Self::MemoryValidationFailed { reason }
            | Self::RedactionFailed { reason }
            | Self::BoundaryViolation { reason }
            | Self::DeterminismFailed { reason }
            | Self::Internal { reason } => reason,
        }
    }
}

impl fmt::Display for E2eConformanceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.code(), self.reason())
    }
}

impl std::error::Error for E2eConformanceError {}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FirstNativeRuntimeError {
    pub(super) code: &'static str,
    pub(super) reason: String,
}

impl FirstNativeRuntimeError {
    pub fn new(code: &'static str, reason: impl Into<String>) -> Self {
        Self {
            code,
            reason: reason.into(),
        }
    }

    pub fn model_not_found(model_ref: &ModelRef) -> Self {
        Self {
            code: "model-not-found",
            reason: format!(
                "first-native fixture model '{}' is not available; supported model_ref is 'qwen-test'",
                model_ref.as_str()
            ),
        }
    }

    pub fn from_conformance(error: E2eConformanceError) -> Self {
        Self {
            code: error.code(),
            reason: error.reason().to_string(),
        }
    }

    pub const fn code(&self) -> &'static str {
        self.code
    }

    pub fn reason(&self) -> &str {
        &self.reason
    }
}

impl fmt::Display for FirstNativeRuntimeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.code, self.reason)
    }
}

impl std::error::Error for FirstNativeRuntimeError {}

#[derive(Clone, Debug, PartialEq)]
pub struct PrefillInput {
    pub tokens: Vec<TokenId>,
}

impl PrefillInput {
    pub fn new(tokens: Vec<TokenId>) -> Self {
        Self { tokens }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ModelDecodeStepInput {
    pub new_tokens: Vec<TokenId>,
    pub kv_cache: KvCacheId,
    pub absolute_position: u64,
}

impl ModelDecodeStepInput {
    pub fn new(new_tokens: Vec<TokenId>, kv_cache: KvCacheId, absolute_position: u64) -> Self {
        Self {
            new_tokens,
            kv_cache,
            absolute_position,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct PrefillExecutionResult {
    pub logits: Vec<f32>,
    pub kv_cache: Option<KvCacheId>,
    pub plan_generation: PreparedExecutionPlanGeneration,
}

#[derive(Clone, Debug, PartialEq)]
pub struct DecodeExecutionResult {
    pub logits: Vec<f32>,
    pub kv_cache: KvCacheId,
    pub plan_generation: PreparedExecutionPlanGeneration,
}

/// Runtime-owned model execution boundary for first-native generation.
///
/// The executor receives a ready ModelInstance and a prepared plan. Logits are
/// outputs of these methods, never caller-provided inputs.
pub trait RuntimeModelExecutor: Send + Sync {
    fn execute_prefill(
        &self,
        instance: &ModelInstance,
        plan: &PreparedExecutionPlan,
        input: PrefillInput,
    ) -> Result<PrefillExecutionResult, InferenceApiError>;

    fn execute_decode_step(
        &self,
        instance: &ModelInstance,
        plan: &PreparedExecutionPlan,
        input: ModelDecodeStepInput,
    ) -> Result<DecodeExecutionResult, InferenceApiError>;
}

impl From<InferenceApiError> for E2eConformanceError {
    fn from(error: InferenceApiError) -> Self {
        Self::GenerationFailed {
            reason: error.to_string(),
        }
    }
}

impl From<ReferenceCpuError> for E2eConformanceError {
    fn from(error: ReferenceCpuError) -> Self {
        Self::GenerationFailed {
            reason: error.to_string(),
        }
    }
}

impl From<ModelLoadingError> for E2eConformanceError {
    fn from(error: ModelLoadingError) -> Self {
        Self::ModelLoadingFailed {
            reason: error.to_string(),
        }
    }
}

impl From<ModelArtifactError> for E2eConformanceError {
    fn from(error: ModelArtifactError) -> Self {
        Self::FixtureInvalid {
            reason: error.to_string(),
        }
    }
}

impl From<QwenComponentError> for E2eConformanceError {
    fn from(error: QwenComponentError) -> Self {
        Self::ModelComponentFailed {
            reason: error.to_string(),
        }
    }
}

impl From<TokenizerError> for E2eConformanceError {
    fn from(error: TokenizerError) -> Self {
        Self::TokenizerFailed {
            reason: error.to_string(),
        }
    }
}

impl From<GraphError> for E2eConformanceError {
    fn from(error: GraphError) -> Self {
        Self::GraphValidationFailed {
            reason: error.to_string(),
        }
    }
}

impl From<FirstScopeError> for E2eConformanceError {
    fn from(error: FirstScopeError) -> Self {
        Self::OperatorCoverageMissing {
            reason: error.to_string(),
        }
    }
}

impl From<SessionError> for E2eConformanceError {
    fn from(error: SessionError) -> Self {
        Self::SessionFailed {
            reason: error.to_string(),
        }
    }
}

impl From<ModelInstanceError> for E2eConformanceError {
    fn from(error: ModelInstanceError) -> Self {
        Self::ModelComponentFailed {
            reason: error.to_string(),
        }
    }
}

impl From<PreparedExecutionPlanError> for E2eConformanceError {
    fn from(error: PreparedExecutionPlanError) -> Self {
        Self::GenerationFailed {
            reason: error.to_string(),
        }
    }
}

impl From<KernelRegistryError> for E2eConformanceError {
    fn from(error: KernelRegistryError) -> Self {
        Self::KernelCoverageMissing {
            reason: error.to_string(),
        }
    }
}

impl From<CliBoundaryError> for E2eConformanceError {
    fn from(error: CliBoundaryError) -> Self {
        Self::BoundaryViolation {
            reason: error.to_string(),
        }
    }
}

impl From<ProviderError> for E2eConformanceError {
    fn from(error: ProviderError) -> Self {
        Self::SuiteUnavailable {
            reason: error.to_string(),
        }
    }
}

impl From<GenerationError> for E2eConformanceError {
    fn from(error: GenerationError) -> Self {
        Self::GenerationFailed {
            reason: error.to_string(),
        }
    }
}

impl From<KvCacheError> for E2eConformanceError {
    fn from(error: KvCacheError) -> Self {
        Self::MemoryValidationFailed {
            reason: error.to_string(),
        }
    }
}

impl From<PrefixCacheError> for E2eConformanceError {
    fn from(error: PrefixCacheError) -> Self {
        Self::MemoryValidationFailed {
            reason: error.to_string(),
        }
    }
}

impl From<TensorError> for E2eConformanceError {
    fn from(error: TensorError) -> Self {
        Self::MemoryValidationFailed {
            reason: error.to_string(),
        }
    }
}

impl From<MemoryError> for E2eConformanceError {
    fn from(error: MemoryError) -> Self {
        Self::MemoryValidationFailed {
            reason: error.to_string(),
        }
    }
}

#[cfg(test)]
mod tests;
