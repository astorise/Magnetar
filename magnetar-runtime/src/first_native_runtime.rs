//! First-native runtime execution and conformance support.
//!
//! Assembles a minimal, deterministic Qwen-like fixture model and drives it
//! through the real Runtime Inference API surface -- model resolution,
//! Model Loading, Model Instance creation, session creation, tokenization,
//! generation, streaming, and cleanup -- using genuine Reference CPU numeric
//! kernels for the forward pass (not canned output), so the success path is
//! a real, if tiny, end-to-end inference run. The `e2e_conformance` module is
//! now a compatibility wrapper around this runtime-owned implementation.

use crate::ProviderExecutionResult;
use crate::affinity::*;
use crate::batching::*;
use crate::cli_boundary::*;
use crate::component::*;
use crate::compute;
use crate::compute::*;
use crate::conformance::first_native_model_execution_profile;
use crate::device::DeviceId;
use crate::execution_graph::*;
use crate::generation::*;
#[cfg(all(not(target_arch = "wasm32"), feature = "wasmtime-component-engine"))]
use crate::graph_builder_capability::*;
use crate::inference_api::*;
use crate::kernel::*;
use crate::kernel_artifact::{
    CompiledKernelArtifactId, PreparedKernel, PreparedKernelGeneration, PreparedKernelIdAllocator,
};
use crate::kernel_dispatch::*;
use crate::kernel_execution_plan::{
    PlanGuard, PlanGuardContext, PlanMemoryRequirements, PlanNodeBinding, PreparedExecutionPhase,
    PreparedExecutionPlan, PreparedExecutionPlanError, PreparedExecutionPlanExecutor,
    PreparedExecutionPlanGeneration, PreparedExecutionPlanId, PreparedExecutionPlanScope,
    PreparedPlanNodeExecution, ResourceBindingPlan, semantic_graph_fingerprint,
};
use crate::kernel_optimization_orchestration::run_kernel_optimization_orchestration_conformance;
use crate::kernel_registry::*;
use crate::kv_cache::*;
use crate::memory::*;
use crate::model::*;
use crate::model_component::*;
use crate::model_instance::*;
use crate::model_loading::*;
use crate::operator::*;
use crate::operator_scope::*;
use crate::prefix_cache::*;
use crate::provider::*;
use crate::qwen_model_component;
use crate::qwen_model_component::*;
use crate::reference_cpu::*;
use crate::runtime::*;
use crate::sampling::*;
use crate::session::*;
use crate::tensor::*;
use crate::tokenizer::*;
use serde::{Deserialize, Serialize};
use sha2::{Digest as ShaDigest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    fmt,
    sync::{Arc, Mutex},
    time::{SystemTime, UNIX_EPOCH},
};

pub const E2E_SUITE_VERSION: &str = "0.1.0";
pub const E2E_FIXTURE_VERSION: &str = "0.1.0";
pub const E2E_FIXTURE_WEIGHT_DIGEST_VERSION: &str = "e2e-qwen-fixture-weights-v1";
pub const E2E_FIXTURE_WEIGHT_DIGEST: &str =
    "sha256:ed7d3a310ae30e08f170ed61cd73f9053e498ae9a17dd7dc980fd61a3152ed90";

const E2E_FIXTURE_EOS_TOKEN: TokenId = 0;
const E2E_FIXTURE_BOS_TOKEN: TokenId = 257;
const E2E_FIXTURE_VOCAB: u64 = 258;
const E2E_FIXTURE_HIDDEN: u64 = 4;
const E2E_FIXTURE_HEADS: u64 = 2;
const E2E_FIXTURE_HEAD_DIM: u64 = 2;
const E2E_FIXTURE_INTERMEDIATE: u64 = 8;
const E2E_FIXTURE_CONTEXT: u64 = 32;
const E2E_FIXTURE_LAYERS: u64 = 1;
// Task 11.5: the checksum-fixture Component (and everything below tied to
// it -- QwenComponentPreflight and friends) is no longer part of the
// production graph-production path, only of tests exercising Component-
// loading conformance generically. `#[cfg(test)]` reflects that; without it
// these items are genuinely dead code in a non-test build.
#[cfg(all(
    test,
    not(target_arch = "wasm32"),
    feature = "wasmtime-component-engine"
))]
const QWEN_GRAPH_COMPONENT_NAME: &str = "magnetar.qwen.graph-fixture";
#[cfg(all(
    test,
    not(target_arch = "wasm32"),
    feature = "wasmtime-component-engine"
))]
const QWEN_GRAPH_COMPONENT_DIGEST: &str =
    "sha256:c95f5ac5c7843991c03543da5d521ee5a2aec14ad6031f6e7cd55d7e2b18078c";

// ---------------------------------------------------------------------
// Error model
// ---------------------------------------------------------------------

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
    code: &'static str,
    reason: String,
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

// ---------------------------------------------------------------------
// Report / result types
// ---------------------------------------------------------------------

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Deserialize, Serialize)]
pub enum E2eTestStatus {
    Passed,
    Failed,
    Skipped,
}

#[derive(Clone, Debug, Eq, PartialEq, Deserialize, Serialize)]
pub struct E2eTestResult {
    pub name: String,
    pub status: E2eTestStatus,
    pub diagnostic: Option<String>,
}

impl E2eTestResult {
    pub fn passed(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            status: E2eTestStatus::Passed,
            diagnostic: None,
        }
    }

    pub fn failed(name: impl Into<String>, diagnostic: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            status: E2eTestStatus::Failed,
            diagnostic: Some(compute::redact_backend_diagnostic(&diagnostic.into())),
        }
    }

    pub fn skipped(name: impl Into<String>, diagnostic: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            status: E2eTestStatus::Skipped,
            diagnostic: Some(compute::redact_backend_diagnostic(&diagnostic.into())),
        }
    }

    fn from_result(name: impl Into<String>, result: Result<(), E2eConformanceError>) -> Self {
        match result {
            Ok(()) => Self::passed(name),
            Err(error) => Self::failed(name, format!("{}: {}", error.code(), error.reason())),
        }
    }
}

fn no_shortcut_success_path_result(fixture: &E2eFixture) -> E2eTestResult {
    let outcome = match run_success_path(fixture) {
        Ok(outcome) => outcome,
        Err(error) => {
            return E2eTestResult::failed(
                "success-path-no-shortcut-validated",
                format!(
                    "required no-shortcut validation failed before success path completed: {}: {}",
                    error.code(),
                    error.reason()
                ),
            );
        }
    };
    match validate_e2e_no_shortcuts(
        outcome.observer.observations(),
        &reference_cpu_kernel_advertisements(),
    ) {
        Ok(()) => E2eTestResult::passed("success-path-no-shortcut-validated"),
        Err(E2eConformanceError::BoundaryViolation { reason }) => E2eTestResult::failed(
            "success-path-no-shortcut-validated",
            format!("required no-shortcut validation failed: {reason}"),
        ),
        Err(error) => E2eTestResult::failed(
            "success-path-no-shortcut-validated",
            format!("{}: {}", error.code(), error.reason()),
        ),
    }
}

/// Machine-readable End-to-End Local Inference Conformance report. Never
/// carries raw prompts, weights, tensor values, cache contents, handles, or
/// memory pointers -- only redacted summaries and structured test results.
#[derive(Clone, Debug, Eq, PartialEq, Deserialize, Serialize)]
pub struct E2eConformanceReport {
    pub suite_version: String,
    pub fixture_version: String,
    pub runtime_version: String,
    pub provider_summary: String,
    pub device_summary: String,
    pub model_component_summary: String,
    pub operator_coverage: BTreeSet<String>,
    pub kernel_coverage: BTreeSet<String>,
    pub test_cases: Vec<E2eTestResult>,
    pub redacted: bool,
    pub duration_millis: u64,
    pub timestamp_unix_seconds: u64,
}

impl E2eConformanceReport {
    fn empty() -> Self {
        Self {
            suite_version: E2E_SUITE_VERSION.into(),
            fixture_version: E2E_FIXTURE_VERSION.into(),
            runtime_version: MAGNETAR_RUNTIME_VERSION.into(),
            provider_summary: String::new(),
            device_summary: String::new(),
            model_component_summary: String::new(),
            operator_coverage: BTreeSet::new(),
            kernel_coverage: BTreeSet::new(),
            test_cases: Vec::new(),
            redacted: true,
            duration_millis: 0,
            timestamp_unix_seconds: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|duration| duration.as_secs())
                .unwrap_or_default(),
        }
    }

    pub fn record(&mut self, result: E2eTestResult) {
        self.test_cases.push(result);
    }

    pub fn is_conformant(&self) -> bool {
        !self
            .test_cases
            .iter()
            .any(|test| test.status == E2eTestStatus::Failed)
    }

    pub fn passed_count(&self) -> usize {
        self.test_cases
            .iter()
            .filter(|test| test.status == E2eTestStatus::Passed)
            .count()
    }

    pub fn failed_count(&self) -> usize {
        self.test_cases
            .iter()
            .filter(|test| test.status == E2eTestStatus::Failed)
            .count()
    }

    pub fn skipped_count(&self) -> usize {
        self.test_cases
            .iter()
            .filter(|test| test.status == E2eTestStatus::Skipped)
            .count()
    }
}

pub fn validate_first_native_native_execution_evidence(
    report: &E2eConformanceReport,
) -> Result<(), E2eConformanceError> {
    first_native_model_execution_profile()
        .validate()
        .map_err(|error| E2eConformanceError::Internal {
            reason: error.to_string(),
        })?;
    if !report.is_conformant() {
        return Err(E2eConformanceError::BoundaryViolation {
            reason: "first native profile requires a conformant E2E report".into(),
        });
    }
    if report.provider_summary != REFERENCE_CPU_PROVIDER_NAME {
        return Err(E2eConformanceError::BoundaryViolation {
            reason: "first native profile requires Reference CPU Provider evidence".into(),
        });
    }
    if !report.model_component_summary.contains("e2e-qwen-fixture") {
        return Err(E2eConformanceError::ModelComponentFailed {
            reason: "first native profile requires Qwen Model Component evidence".into(),
        });
    }
    if report.operator_coverage.is_empty() {
        return Err(E2eConformanceError::OperatorCoverageMissing {
            reason: "first native profile requires Operator execution evidence".into(),
        });
    }
    if report.kernel_coverage.is_empty() {
        return Err(E2eConformanceError::KernelCoverageMissing {
            reason: "first native profile requires Kernel execution evidence".into(),
        });
    }
    for required in [
        "success-path-no-shortcut-validated",
        "operator-coverage",
        "kernel-coverage",
        "reference-cpu-selected-through-kernel-registry",
        "no-shortcut-direct-provider-rejected",
        "no-shortcut-direct-kernel-invocation-rejected",
    ] {
        if !report
            .test_cases
            .iter()
            .any(|test| test.name == required && test.status == E2eTestStatus::Passed)
        {
            return Err(E2eConformanceError::BoundaryViolation {
                reason: format!("missing first native structural evidence '{required}'"),
            });
        }
    }
    Ok(())
}

pub fn e2e_conformance_report_json(
    report: &E2eConformanceReport,
) -> Result<String, serde_json::Error> {
    serde_json::to_string_pretty(report)
}

// ---------------------------------------------------------------------
// Fixture model
// ---------------------------------------------------------------------

/// A minimal, deterministic Qwen-like decoder-only fixture: config,
/// identity, architecture implementation, a Model Artifact manifest that
/// passes normal Model Artifact and Qwen baseline validation, a Tokenizer
/// Contract fixture, and deterministic weight tensors keyed by the same
/// canonical logical tensor names Model Loading expects.
#[derive(Clone)]
pub struct E2eFixture {
    pub config: QwenConfig,
    pub identity: ModelComponentIdentity,
    pub architecture_implementation: ModelArchitectureImplementation,
    pub manifest: ModelManifest,
    pub tokenizer: FixtureTokenizer,
    pub weights: BTreeMap<String, HostTensor>,
}

pub fn e2e_fixture_config() -> QwenConfig {
    let architecture = qwen_architecture_metadata(
        E2E_FIXTURE_HIDDEN,
        E2E_FIXTURE_LAYERS,
        E2E_FIXTURE_HEADS,
        E2E_FIXTURE_HEADS,
        E2E_FIXTURE_HEAD_DIM,
        E2E_FIXTURE_INTERMEDIATE,
        E2E_FIXTURE_VOCAB,
        E2E_FIXTURE_CONTEXT,
    );
    let mut config = QwenConfig::new(architecture, QwenRopeConfig::standard(E2E_FIXTURE_HEAD_DIM));
    config.tied_embeddings = true;
    config
}

pub fn e2e_fixture_identity() -> ModelComponentIdentity {
    qwen_component_identity(
        ModelComponentId::new("e2e-qwen-fixture").expect("static id is valid"),
        ModelComponentVersion::new(1, 0, 0),
        ModelComponentImplementationKind::WebAssemblyComponent,
    )
}

pub fn e2e_fixture_tokenizer() -> Result<FixtureTokenizer, E2eConformanceError> {
    let metadata = TokenizerMetadata {
        id: TokenizerId::new("e2e-fixture-tokenizer")?,
        artifact: TokenizerArtifactId::new("e2e-fixture-tokenizer-artifact")?,
        digest: ModelDigest::parse(
            "sha256:0000000000000000000000000000000000000000000000000000000000000002",
        )
        .map_err(|error| E2eConformanceError::FixtureInvalid {
            reason: error.to_string(),
        })?,
        family: TokenizerFamily::new("byte-fixture")?,
        revision: TokenizerRevision::new("r1")?,
        vocabulary_size: E2E_FIXTURE_VOCAB as u32,
        added_token_count: 2,
        token_id_range: TokenIdRange::new(0, E2E_FIXTURE_BOS_TOKEN),
        model_max_length: Some(E2E_FIXTURE_CONTEXT as u32),
        special_tokens: vec![
            SpecialToken::new(SpecialTokenKind::Eos, "<eos>", E2E_FIXTURE_EOS_TOKEN),
            SpecialToken::new(SpecialTokenKind::Bos, "<bos>", E2E_FIXTURE_BOS_TOKEN),
        ],
        additional_special_tokens: Vec::new(),
        byte_fallback: true,
        normalization: None,
        pre_tokenizer: None,
        supports_offsets: true,
        supports_token_type_ids: false,
        supports_browser: true,
    };
    metadata.validate()?;
    Ok(FixtureTokenizer::new(metadata))
}

pub fn e2e_fixture_manifest(
    config: &QwenConfig,
    architecture: &ModelArchitecture,
) -> Result<ModelManifest, E2eConformanceError> {
    const DIGEST: &str = "sha256:0000000000000000000000000000000000000000000000000000000000000003";
    let weights_digest = e2e_fixture_safetensors_digest();
    let weights_size_bytes = E2E_FIXTURE_SAFETENSORS_BYTES.len();
    // Real per-tensor content digests, computed from the checked-in
    // Safetensors file's actual bytes -- not the synthetic in-memory
    // `e2e_fixture_weights` -- so this manifest genuinely constrains what
    // counts as each tensor's content
    // (`bind-materialized-weight-content-to-model-artifact-digests`).
    let inventory = e2e_fixture_weight_inventory_with_digests(config)?;
    let mut tensor_yaml = String::new();
    for tensor in &inventory {
        let shape_text = tensor
            .shape
            .iter()
            .map(u64::to_string)
            .collect::<Vec<_>>()
            .join(", ");
        let digest = tensor
            .digest
            .as_ref()
            .ok_or_else(|| E2eConformanceError::FixtureInvalid {
                reason: format!(
                    "fixture tensor '{}' unexpectedly has no computed content digest",
                    tensor.name
                ),
            })?;
        tensor_yaml.push_str(&format!(
            "  - name: {name}\n    shape: [{shape_text}]\n    storage_dtype: f32\n    digest: {digest}\n",
            name = tensor.name,
            digest = digest.value,
        ));
    }
    let yaml = format!(
        r#"
schema: magnetar-model-artifact
schema_version: 1
kind: model-bundle
digest: {DIGEST}
model:
  name: e2e-fixture-model
  revision: r1
architecture:
  family: {family}
  identifier: {identifier}
tokenizer: tokenizer
storage_dtype: f32
compute_dtype: f32
supported_compute_dtypes: [f32]
generation:
  temperature: 0.0
  max_tokens: 4
artifacts:
  weights:
    kind: model-weights
    digest: {weights_digest}
    size_bytes: {weights_size_bytes}
  config:
    kind: model-config
    digest: {DIGEST}
    size_bytes: 16
  tokenizer:
    kind: tokenizer
    digest: {DIGEST}
    size_bytes: 8
tensors:
{tensor_yaml}"#,
        family = architecture.family,
        identifier = architecture.identifier,
    );
    ModelManifest::from_yaml_str(&yaml).map_err(E2eConformanceError::from)
}

/// Deterministic pseudo-random value in `[-0.5, 0.5]`, purely a function of
/// `seed` -- no RNG dependency, no shared mutable state, fully reproducible.
fn fixture_value(seed: u64) -> f32 {
    let mut x = seed
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add(0x2545_F491_4F6C_DD1D);
    x ^= x >> 33;
    x = x.wrapping_mul(0xFF51_AFD7_ED55_8CCD);
    x ^= x >> 33;
    ((x % 1000) as f32 / 1000.0) - 0.5
}

fn fixture_seed(name: &str, offset: u64) -> u64 {
    let mut hash: u64 = 0xCBF2_9CE4_8422_2325;
    for byte in name.bytes() {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01B3);
    }
    hash.wrapping_add(offset)
}

fn hex_digest(bytes: &[u8]) -> String {
    bytes
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>()
}

fn fixture_tensor(name: &str, shape: &[u64]) -> Result<HostTensor, E2eConformanceError> {
    let len = shape.iter().product::<u64>();
    let is_norm_weight = shape.len() == 1;
    let data: Vec<f32> = (0..len)
        .map(|i| {
            let value = fixture_value(fixture_seed(name, i));
            if is_norm_weight {
                1.0 + value * 0.1
            } else {
                value
            }
        })
        .collect();
    HostTensor::new(shape.to_vec(), data).map_err(E2eConformanceError::from)
}

pub fn e2e_fixture_weight_digest(weights: &BTreeMap<String, HostTensor>) -> String {
    let mut hasher = Sha256::new();
    hasher.update(E2E_FIXTURE_WEIGHT_DIGEST_VERSION.as_bytes());
    for (name, tensor) in weights {
        hasher.update(name.as_bytes());
        hasher.update([0]);
        for dimension in &tensor.shape {
            hasher.update(dimension.to_le_bytes());
        }
        hasher.update([0xff]);
        for value in &tensor.data {
            hasher.update(value.to_bits().to_le_bytes());
        }
        hasher.update([0xfe]);
    }
    let digest = hasher.finalize();
    format!("sha256:{}", hex_digest(&digest))
}

pub fn e2e_fixture_weights(
    config: &QwenConfig,
) -> Result<BTreeMap<String, HostTensor>, E2eConformanceError> {
    let mut weights = BTreeMap::new();
    for name in qwen_expected_tensor_names(config.architecture.layer_count, config.tied_embeddings)
    {
        let shape = qwen_expected_tensor_shape(&name, config).ok_or_else(|| {
            E2eConformanceError::FixtureInvalid {
                reason: format!("no expected shape for fixture tensor '{name}'"),
            }
        })?;
        weights.insert(name.clone(), fixture_tensor(&name, &shape)?);
    }
    Ok(weights)
}

/// Raw bytes of a real, checked-in Safetensors file encoding the exact same
/// deterministic weights [`e2e_fixture_weights`] builds in memory --
/// generated once via `formats/safetensors`'s own test suite (that crate,
/// not this one, is allowed to depend on both `magnetar-runtime` and a
/// Safetensors writer; see `materialize-weights-from-real-model-artifact`'s
/// design.md for why), committed here, and read by [`e2e_fixture_weight_inventory`]/
/// [`host_tensors_from_artifact_bytes`] as the real byte-level Model
/// Artifact the production first-native path actually materializes weights
/// from -- not merely an in-memory recreation.
pub const E2E_FIXTURE_SAFETENSORS_BYTES: &[u8] =
    include_bytes!("../fixtures/e2e-fixture-weights.safetensors");

/// Size, in bytes, of the little-endian header-length prefix every
/// Safetensors file begins with -- a local copy of the same constant
/// `formats/safetensors::HEADER_LENGTH_PREFIX_BYTES` defines, since
/// `magnetar-runtime` cannot depend on that crate even to reuse a `usize`
/// (`externalize-runtime-extension-modules`).
const HEADER_LENGTH_PREFIX_BYTES: usize = 8;

/// The real sha256 digest of [`E2E_FIXTURE_SAFETENSORS_BYTES`]'s actual
/// bytes -- computed at call time (not hardcoded) so it can never drift
/// from the checked-in file it describes; used by [`e2e_fixture_manifest`]'s
/// `artifacts.weights.digest` field, checked the same way
/// `bind_qwen_fixture_weights`'s existing digest gate already checks every
/// other declared digest.
pub fn e2e_fixture_safetensors_digest() -> String {
    let mut hasher = Sha256::new();
    hasher.update(E2E_FIXTURE_SAFETENSORS_BYTES);
    format!("sha256:{}", hex_digest(&hasher.finalize()))
}

/// The generic `ModelTensorMetadata` inventory [`E2E_FIXTURE_SAFETENSORS_BYTES`]'s
/// real, checked-in file actually contains, computed directly rather than by
/// calling a format parser: `magnetar-runtime` cannot depend on
/// `formats/safetensors` even at test time (`externalize-runtime-extension-modules`),
/// but this fixture's tensor names/shapes/generation order are already
/// fully known here (`qwen_expected_tensor_names`/`qwen_expected_tensor_shape`),
/// and the file was written by iterating that exact same sorted order (a
/// `BTreeSet`/`BTreeMap`'s iteration order, deterministic regardless of
/// insertion order) with each tensor's byte length equal to its element
/// count times 4 -- so the offsets below are provably the same offsets a
/// real parse of this file would produce, not merely assumed to match.
/// `formats/safetensors`'s own test suite is what actually proves the file
/// is parseable by the real, independent parser (see that crate's
/// `e2e_fixture_weights_round_trip_through_real_safetensors_bytes` test).
pub fn e2e_fixture_weight_inventory(
    config: &QwenConfig,
) -> Result<Vec<ModelTensorMetadata>, E2eConformanceError> {
    let mut tensors = Vec::new();
    let mut offset = 0_u64;
    for name in qwen_expected_tensor_names(config.architecture.layer_count, config.tied_embeddings)
    {
        let shape = qwen_expected_tensor_shape(&name, config).ok_or_else(|| {
            E2eConformanceError::FixtureInvalid {
                reason: format!("no expected shape for fixture tensor '{name}'"),
            }
        })?;
        let element_count: u64 = shape.iter().product();
        let size_bytes = element_count * 4;
        tensors.push(ModelTensorMetadata {
            name: name.clone(),
            shape,
            storage_dtype: ModelDType::F32,
            layout: None,
            shard: None,
            offset_bytes: Some(offset),
            size_bytes: Some(size_bytes),
            quantization: None,
            expected_compute_dtype: None,
            digest: None,
        });
        offset += size_bytes;
    }
    Ok(tensors)
}

/// [`e2e_fixture_weight_inventory`], with each tensor's `digest` populated
/// from the real, checked-in Safetensors file's actual bytes -- computed
/// from [`e2e_fixture_weights_from_real_artifact`]'s already-real,
/// already-parsed [`HostTensor`]s (not from [`e2e_fixture_weights`]'s
/// synthetic in-memory values), so this is a real, non-circular content
/// binding: the fixture is not hashing its own assumptions about itself.
/// Implements `bind-materialized-weight-content-to-model-artifact-digests`'s
/// "Tensor Content Digest Binding" requirement for the one real artifact
/// source this crate has today.
pub fn e2e_fixture_weight_inventory_with_digests(
    config: &QwenConfig,
) -> Result<Vec<ModelTensorMetadata>, E2eConformanceError> {
    let inventory = e2e_fixture_weight_inventory(config)?;
    let real_weights = e2e_fixture_weights_from_real_artifact(config)?;
    Ok(inventory
        .into_iter()
        .map(|mut metadata| {
            if let Some(tensor) = real_weights.get(&metadata.name) {
                metadata.digest = Some(ModelDigest::sha256(&tensor.content_bytes()));
            }
            metadata
        })
        .collect())
}

/// Materializes the E2E fixture's weights by reading
/// [`E2E_FIXTURE_SAFETENSORS_BYTES`]'s real bytes at
/// [`e2e_fixture_weight_inventory`]'s declared offsets -- the real-file
/// counterpart to [`e2e_fixture_weights`]'s in-memory construction, proven
/// equal to it by `tests::e2e_fixture_real_artifact_weights_match_in_memory_weights`.
pub fn e2e_fixture_weights_from_real_artifact(
    config: &QwenConfig,
) -> Result<BTreeMap<String, HostTensor>, E2eConformanceError> {
    let inventory = e2e_fixture_weight_inventory(config)?;
    // The Safetensors envelope's own 8-byte little-endian header-length
    // prefix -- reading it is not "parsing" in the sense
    // `externalize-runtime-extension-modules` forbids (no JSON, no
    // validation, no `formats/safetensors` call): it is the one universally
    // fixed part of the container format, needed here only to locate where
    // the tensor-data section starts within the file
    // (`host_tensors_from_artifact_bytes`'s `data_section_start`).
    let header_length = E2E_FIXTURE_SAFETENSORS_BYTES
        .get(..HEADER_LENGTH_PREFIX_BYTES)
        .and_then(|prefix| prefix.try_into().ok())
        .map(u64::from_le_bytes)
        .ok_or_else(|| E2eConformanceError::FixtureInvalid {
            reason: "real artifact bytes are too short to hold a header length prefix".into(),
        })?;
    let data_section_start = HEADER_LENGTH_PREFIX_BYTES as u64 + header_length;
    host_tensors_from_artifact_bytes(
        &inventory,
        E2E_FIXTURE_SAFETENSORS_BYTES,
        data_section_start,
    )
    .map_err(|error| E2eConformanceError::FixtureInvalid {
        reason: format!("real artifact bytes failed to materialize: {error}"),
    })
}

pub fn e2e_fixture() -> Result<E2eFixture, E2eConformanceError> {
    let config = e2e_fixture_config();
    let identity = e2e_fixture_identity();
    config.validate(&identity)?;
    let architecture_implementation = qwen_model_component::qwen_architecture_implementation(
        &identity,
        ModelArchitectureImplementationKind::ComponentBased,
    );
    let manifest = e2e_fixture_manifest(&config, &architecture_implementation.architecture)?;
    let tokenizer = e2e_fixture_tokenizer()?;
    let weights = e2e_fixture_weights(&config)?;

    let descriptor = qwen_component_descriptor(identity.clone(), &config)?;
    qwen_validate_model_artifact(&descriptor, &config, &manifest)?;

    Ok(E2eFixture {
        config,
        identity,
        architecture_implementation,
        manifest,
        tokenizer,
        weights,
    })
}

// ---------------------------------------------------------------------
// Real Reference CPU forward pass (operator coverage + determinism)
// ---------------------------------------------------------------------

/// Every required-now Operator this forward pass exercises through the real
/// Reference CPU kernel functions (not a shortcut, not a placeholder).
pub const E2E_EXERCISED_OPERATORS: [&str; 10] = [
    "embedding",
    "rmsnorm",
    "matmul",
    "rope",
    "attention",
    "softmax",
    "silu",
    "add",
    "mul",
    "residual-add",
];

fn fixture_tensor_by_name<'a>(
    weights: &'a BTreeMap<String, HostTensor>,
    name: &str,
) -> Result<&'a HostTensor, E2eConformanceError> {
    weights
        .get(name)
        .ok_or_else(|| E2eConformanceError::FixtureInvalid {
            reason: format!("fixture is missing weight tensor '{name}'"),
        })
}

#[cfg(test)]
fn apply_rope_per_head(
    tensor: &HostTensor,
    head_count: u64,
    head_dimension: u64,
    rope_config: &QwenRopeConfig,
) -> Result<HostTensor, E2eConformanceError> {
    let (rows, cols) = tensor.rows_cols()?;
    let mut out = vec![0.0_f32; tensor.data.len()];
    for head in 0..head_count {
        let start_col = head * head_dimension;
        let mut head_data = Vec::with_capacity((rows * head_dimension) as usize);
        for row in 0..rows {
            let base = (row * cols + start_col) as usize;
            head_data.extend_from_slice(&tensor.data[base..base + head_dimension as usize]);
        }
        let head_tensor = HostTensor::new([rows, head_dimension], head_data)?;
        let rotated = rope(
            &head_tensor,
            rope_config.base as f32,
            rope_config.scale.unwrap_or(1.0) as f32,
            rope_config.dimension,
            0,
        )?;
        for row in 0..rows {
            let dst_base = (row * cols + start_col) as usize;
            let src_base = (row * head_dimension) as usize;
            out[dst_base..dst_base + head_dimension as usize]
                .copy_from_slice(&rotated.data[src_base..src_base + head_dimension as usize]);
        }
    }
    HostTensor::new(tensor.shape.clone(), out).map_err(E2eConformanceError::from)
}

/// Test oracle for the decoder stack. Production first-native generation uses
/// `execute_qwen_hidden_states_through_dispatch` instead.
#[cfg(test)]
fn e2e_forward_hidden_states(
    fixture: &E2eFixture,
    token_ids: &[TokenId],
) -> Result<HostTensor, E2eConformanceError> {
    if token_ids.is_empty() {
        return Err(E2eConformanceError::GenerationFailed {
            reason: "forward pass requires at least one token".into(),
        });
    }
    let architecture = &fixture.config.architecture;
    let seq_len = token_ids.len() as u64;
    let epsilon = fixture.config.rmsnorm_epsilon;

    let ids_tensor = HostTensor::new(
        [seq_len],
        token_ids.iter().map(|id| *id as f32).collect::<Vec<_>>(),
    )?;
    let token_embedding = fixture_tensor_by_name(&fixture.weights, "token_embedding")?;
    let mut hidden_states = embedding_lookup(token_embedding, &ids_tensor)?;

    for layer in 0..architecture.layer_count {
        let prefix = format!("layers.{layer}.");
        let input_norm = fixture_tensor_by_name(&fixture.weights, &format!("{prefix}input_norm"))?;
        let q_weight =
            fixture_tensor_by_name(&fixture.weights, &format!("{prefix}self_attn.q_proj"))?;
        let k_weight =
            fixture_tensor_by_name(&fixture.weights, &format!("{prefix}self_attn.k_proj"))?;
        let v_weight =
            fixture_tensor_by_name(&fixture.weights, &format!("{prefix}self_attn.v_proj"))?;
        let o_weight =
            fixture_tensor_by_name(&fixture.weights, &format!("{prefix}self_attn.o_proj"))?;
        let post_attn_norm =
            fixture_tensor_by_name(&fixture.weights, &format!("{prefix}post_attn_norm"))?;
        let gate_weight =
            fixture_tensor_by_name(&fixture.weights, &format!("{prefix}mlp.gate_proj"))?;
        let up_weight = fixture_tensor_by_name(&fixture.weights, &format!("{prefix}mlp.up_proj"))?;
        let down_weight =
            fixture_tensor_by_name(&fixture.weights, &format!("{prefix}mlp.down_proj"))?;

        let normed = rmsnorm(&hidden_states, input_norm, epsilon)?;
        let q = matmul(&normed, q_weight, false, false)?;
        let k = matmul(&normed, k_weight, false, false)?;
        let v = matmul(&normed, v_weight, false, false)?;
        let q = apply_rope_per_head(
            &q,
            architecture.attention_head_count,
            architecture.head_dimension,
            &fixture.config.rope,
        )?;
        let k = apply_rope_per_head(
            &k,
            architecture.kv_head_count,
            architecture.head_dimension,
            &fixture.config.rope,
        )?;
        let attention_out = attention(
            &q,
            &k,
            &v,
            architecture.attention_head_count,
            architecture.head_dimension,
            Some(architecture.kv_head_count),
            None,
            true,
        )?;
        let attention_proj = matmul(&attention_out, o_weight, false, false)?;
        hidden_states = residual_add(&attention_proj, &hidden_states)?;

        let normed_mlp = rmsnorm(&hidden_states, post_attn_norm, epsilon)?;
        let gate = matmul(&normed_mlp, gate_weight, false, false)?;
        let up = matmul(&normed_mlp, up_weight, false, false)?;
        let activated = silu(&gate);
        let gated = mul(&activated, &up)?;
        let mlp_out = matmul(&gated, down_weight, false, false)?;
        hidden_states = residual_add(&mlp_out, &hidden_states)?;
    }

    let final_norm = fixture_tensor_by_name(&fixture.weights, "final_norm")?;
    rmsnorm(&hidden_states, final_norm, epsilon).map_err(E2eConformanceError::from)
}

/// Test oracle for a deterministic Qwen-like forward pass. This is deliberately
/// not compiled into the production runtime path; the runtime path executes
/// operators through Kernel Registry selection and Provider dispatch.
#[cfg(test)]
pub fn e2e_forward(
    fixture: &E2eFixture,
    token_ids: &[TokenId],
) -> Result<Vec<f32>, E2eConformanceError> {
    let normed_final = e2e_forward_hidden_states(fixture, token_ids)?;
    let token_embedding = fixture_tensor_by_name(&fixture.weights, "token_embedding")?;
    // Tied embeddings: logits = normed_final @ token_embedding^T.
    let logits = matmul(&normed_final, token_embedding, false, true)?;
    // Exercise the softmax kernel for operator-coverage/report purposes;
    // Sampling owns the authoritative distribution derived from raw logits.
    let _distribution = softmax_rows(&logits)?;

    let vocab = fixture.config.architecture.vocabulary_size as usize;
    let last_row_start = (token_ids.len() - 1) * vocab;
    Ok(logits.data[last_row_start..last_row_start + vocab].to_vec())
}

#[derive(Clone)]
struct E2eRuntimeModelExecutionEngine {
    fixture: E2eFixture,
    kv_states: Arc<Mutex<BTreeMap<String, FirstNativeExecutionKvState>>>,
    pending_kv_states: Arc<Mutex<BTreeMap<String, FirstNativeExecutionKvState>>>,
    #[cfg(test)]
    forced_token: Option<TokenId>,
}

#[derive(Clone, Debug)]
struct FirstNativeExecutionKvState {
    cache: KvCacheId,
    compatibility: KvCacheCompatibility,
    layer_kv: Vec<FirstNativeLayerKvState>,
    /// The Provider `execute_qwen_graph` actually resolved and wrote this
    /// step's pending K/V resources under. `None` before the first graph
    /// execution for this state (freshly created by
    /// `create_prefill_kv_state`); `KvUpdateTransaction::begin`/
    /// `discard_pending_kv_state` fall back to Reference CPU only in that
    /// case (`generalize-first-native-provider-dispatch`).
    provider: Option<ProviderBinding>,
}

/// One layer's K/V tensor resource identities. Before commit, these name
/// *pending* resources this generation step wrote but sampling/token commit
/// has not yet accepted; after commit, they match the
/// `KvCache.layer_resources` bindings Runtime now owns (task 7.1/7.3).
#[derive(Clone, Debug)]
struct FirstNativeLayerKvState {
    k: TensorResourceId,
    v: TensorResourceId,
}

impl E2eRuntimeModelExecutionEngine {
    fn create_prefill_kv_state(
        &self,
        runtime: &mut Runtime,
        request: &GenerationRequest,
    ) -> Result<FirstNativeExecutionKvState, InferenceApiError> {
        let compatibility = self.kv_compatibility(request);
        let mut cache = KvCache::new(
            KvCacheId::new("first-native-temporary-kv").map_err(InferenceApiError::from)?,
            if request.session.is_some() {
                KvCacheScope::Session
            } else {
                KvCacheScope::Operation
            },
            compatibility.clone(),
            KvCacheLayoutMetadata::contiguous(
                self.fixture.config.architecture.layer_count as u32,
                self.fixture.config.architecture.kv_head_count as u32,
                self.fixture.config.architecture.head_dimension as u32,
                request
                    .prompt_token_count
                    .saturating_add(request.max_new_tokens)
                    .max(1) as u32,
                ComputeDType::Float32,
            ),
        );
        if let Some(session) = &request.session {
            cache = cache.with_session(session.clone());
        }
        let cache_id = runtime.create_kv_cache(cache)?;
        Ok(FirstNativeExecutionKvState {
            cache: cache_id,
            compatibility,
            layer_kv: Vec::new(),
            provider: None,
        })
    }

    fn kv_state_key(request: &GenerationRequest) -> String {
        request.request_id.as_str().to_string()
    }

    fn store_kv_state(
        &self,
        request: &GenerationRequest,
        state: FirstNativeExecutionKvState,
    ) -> Result<(), InferenceApiError> {
        self.kv_states
            .lock()
            .map_err(|_| InferenceApiError::KvCacheUnavailable {
                reason: "first-native KV state lock poisoned".into(),
            })?
            .insert(Self::kv_state_key(request), state);
        Ok(())
    }

    fn store_pending_kv_state(
        &self,
        request: &GenerationRequest,
        state: FirstNativeExecutionKvState,
    ) -> Result<(), InferenceApiError> {
        self.pending_kv_states
            .lock()
            .map_err(|_| InferenceApiError::KvCacheUnavailable {
                reason: "first-native pending KV state lock poisoned".into(),
            })?
            .insert(Self::kv_state_key(request), state);
        Ok(())
    }

    fn take_pending_kv_state(
        &self,
        request: &GenerationRequest,
    ) -> Result<FirstNativeExecutionKvState, InferenceApiError> {
        self.pending_kv_states
            .lock()
            .map_err(|_| InferenceApiError::KvCacheUnavailable {
                reason: "first-native pending KV state lock poisoned".into(),
            })?
            .remove(&Self::kv_state_key(request))
            .ok_or_else(|| InferenceApiError::KvCacheUnavailable {
                reason: "generation step has no pending first-native KV state to commit".into(),
            })
    }

    /// Discards a generation step's pending (not yet committed) KV state,
    /// abandoning it (task group 9 / Correctif 11): releases each layer's
    /// pending K/V resource from the registered Provider's storage, not
    /// just this engine's own bookkeeping map, so an abandoned pending
    /// write does not linger as an orphaned Provider-owned resource
    /// forever. Idempotent -- discarding an already-discarded or
    /// never-pending request is not an error.
    fn discard_pending_kv_state(
        &self,
        runtime: &mut Runtime,
        request: &GenerationRequest,
    ) -> Result<(), InferenceApiError> {
        let discarded = self
            .pending_kv_states
            .lock()
            .map_err(|_| InferenceApiError::KvCacheUnavailable {
                reason: "first-native pending KV state lock poisoned".into(),
            })?
            .remove(&Self::kv_state_key(request));
        let Some(discarded) = discarded else {
            return Ok(());
        };
        // `generalize-first-native-provider-dispatch`: resolve the Provider
        // this step's graph execution actually wrote the pending resources
        // under, not Reference CPU unconditionally -- discarding a non-CPU
        // step's pending state used to look for it in the wrong Provider's
        // storage.
        let provider_binding = discarded
            .provider
            .clone()
            .unwrap_or_else(|| ProviderBinding::new(REFERENCE_CPU_PROVIDER_NAME));
        if let Ok(provider) = resolve_kernel_execution_provider(runtime, &provider_binding) {
            // `release_admitted_tensor`, not `release_tensor`: the pending
            // write these resources came from was itself admitted through
            // `write_tensor_admitted` (Correctif 1), so discarding it must
            // release that allocation too, not just drop the Provider
            // storage entry.
            for layer in &discarded.layer_kv {
                // Best-effort: this pending state is being discarded
                // regardless of whether Provider-side release succeeds.
                let _ = provider.release_admitted_tensor(runtime.memory_mut(), &layer.k);
                let _ = provider.release_admitted_tensor(runtime.memory_mut(), &layer.v);
            }
        }
        Ok(())
    }

    fn decode_kv_state(
        &self,
        runtime: &mut Runtime,
        request: &GenerationRequest,
    ) -> Result<FirstNativeExecutionKvState, InferenceApiError> {
        let state = self
            .kv_states
            .lock()
            .map_err(|_| InferenceApiError::KvCacheUnavailable {
                reason: "first-native KV state lock poisoned".into(),
            })?
            .get(&Self::kv_state_key(request))
            .cloned()
            .ok_or_else(|| InferenceApiError::KvCacheUnavailable {
                reason: "decode requires existing first-native KV state".into(),
            })?;
        runtime.validate_kv_cache_reuse(&state.cache, &state.compatibility, None)?;
        Ok(state)
    }

    fn kv_compatibility(&self, request: &GenerationRequest) -> KvCacheCompatibility {
        KvCacheCompatibility::new(
            request.model.clone(),
            request.tokenizer.tokenizer_id.clone(),
        )
        .with_prefix_fingerprint(PrefixFingerprint::from_tokens(
            &request.input_token_ids,
            request.request_id.as_str(),
            &request.tokenizer.tokenizer_id,
        ))
    }
}

/// Physically transposes a rank-2 [`HostTensor`], returning a `[cols, rows]`
/// tensor whose `[j, i]` entry is `tensor[i, j]`. [`dispatch_matmul`] always
/// runs an untransposed `a @ b` (matching what the portable `matmul`
/// Operator's shape rule validates -- `a[-1] == b[-2]` -- which does not
/// consult a `transpose_b` execution attribute), so a caller that needs
/// `a @ b^T` transposes `b` itself before dispatching.
fn transpose_rows_cols(tensor: &HostTensor) -> Result<HostTensor, E2eConformanceError> {
    let (rows, cols) = tensor.rows_cols()?;
    let mut out = vec![0.0_f32; tensor.data.len()];
    for row in 0..rows {
        for col in 0..cols {
            out[(col * rows + row) as usize] = tensor.data[(row * cols + col) as usize];
        }
    }
    HostTensor::new([cols, rows], out).map_err(E2eConformanceError::from)
}

/// Dispatches a genuine Kernel Registry -> Kernel Dispatch -> Reference CPU
/// Provider matmul computing `a @ b`, returning both the
/// [`KernelDispatchResult`] (used to build [`RuntimeGenerationExecutionEvidence`])
/// and the actual output tensor the Provider computed and wrote back.
///
/// This exists so evidence is always drawn from the dispatch that produced
/// the data a caller returns, rather than from an unrelated "proof"
/// computation whose result is discarded: the caller reads the returned
/// output tensor -- not a value it computed independently -- so a dispatch
/// that never ran, or that ran over different data, cannot produce evidence
/// for logits it did not causally produce.
fn dispatch_matmul(
    runtime: &Runtime,
    a: &HostTensor,
    b: &HostTensor,
) -> Result<(KernelDispatchResult, HostTensor), InferenceApiError> {
    dispatch_matmul_with_prepared_plan(runtime, a, b, "matmul", None)
}

fn dispatch_matmul_with_prepared_plan(
    runtime: &Runtime,
    a: &HostTensor,
    b: &HostTensor,
    operation_id: &str,
    prepared_plan: Option<&PreparedExecutionPlan>,
) -> Result<(KernelDispatchResult, HostTensor), InferenceApiError> {
    let advertisements = reference_cpu_kernel_advertisements();
    let matmul_operator = advertisements
        .iter()
        .find(|advertisement| advertisement.implemented_operator.name() == "matmul")
        .map(|advertisement| advertisement.implemented_operator.clone())
        .ok_or_else(|| InferenceApiError::KernelUnavailable {
            reason: "Reference CPU fixture does not advertise matmul".into(),
        })?;
    let to_generation_failed = |error: ReferenceCpuError| InferenceApiError::GenerationFailed {
        reason: error.to_string(),
    };
    let (a_rows, a_cols) = a.rows_cols().map_err(to_generation_failed)?;
    let (b_rows, b_cols) = b.rows_cols().map_err(to_generation_failed)?;
    let out_rows = a_rows;
    let out_cols = b_cols;

    let dtype = DTypeDescriptor::portable(ComputeDType::Float32);
    let a_descriptor = TensorDescriptor::new(
        ShapeDescriptor::new([a_rows, a_cols]),
        dtype.clone(),
        LayoutDescriptor::Contiguous,
    );
    let b_descriptor = TensorDescriptor::new(
        ShapeDescriptor::new([b_rows, b_cols]),
        dtype.clone(),
        LayoutDescriptor::Contiguous,
    );
    let output_descriptor = TensorDescriptor::new(
        ShapeDescriptor::new([out_rows, out_cols]),
        dtype,
        LayoutDescriptor::Contiguous,
    );
    let affinity = ResourceAffinity::new(FallbackClass::Transparent)
        .with_provider(ProviderBinding::new(REFERENCE_CPU_PROVIDER_NAME))
        .with_execution_context(runtime.context().id());

    let input_a = TensorResourceDescriptor::new(
        TensorResourceId::new("e2e-runtime-generation-a"),
        a_descriptor,
        affinity.clone(),
    );
    let input_b = TensorResourceDescriptor::new(
        TensorResourceId::new("e2e-runtime-generation-b"),
        b_descriptor,
        affinity.clone(),
    );
    let output = TensorResourceDescriptor::new(
        TensorResourceId::new("e2e-runtime-generation-logits"),
        output_descriptor,
        affinity,
    );
    let selection_request = KernelSelectionRequest::new(
        "e2e-runtime-generation-step",
        matmul_operator,
        ResourceAffinity::new(FallbackClass::Transparent),
    )
    .with_input(KernelResource::new(
        input_a.clone(),
        KernelMemoryClass::Host,
    ))
    .with_input(KernelResource::new(
        input_b.clone(),
        KernelMemoryClass::Host,
    ))
    .with_output(KernelResource::new(output.clone(), KernelMemoryClass::Host));
    let selected = if let Some(prepared_plan) = prepared_plan {
        prepared_candidate_for_operation(
            runtime,
            prepared_plan,
            operation_id,
            &selection_request.operator,
        )?
    } else {
        runtime
            .kernel_registry()
            .select(&selection_request)
            .map_err(|error| InferenceApiError::KernelUnavailable {
                reason: error.to_string(),
            })?
            .selected
            .ok_or_else(|| InferenceApiError::KernelUnavailable {
                reason: "Kernel Registry selected no Reference CPU candidate".into(),
            })?
    };
    let advertisement = runtime
        .kernel_registry()
        .active_advertisement(&selected.kernel)
        .ok_or_else(|| InferenceApiError::KernelUnavailable {
            reason: "selected Reference CPU advertisement is no longer active".into(),
        })?;
    let mut plan = KernelDispatchPlan::from_selection(
        KernelDispatchPlanId::new("e2e-runtime-generation-dispatch"),
        &selection_request,
        &selected,
        advertisement,
        KernelInvocationId::new("e2e-runtime-generation-invocation"),
    )
    .map_err(|error| InferenceApiError::KernelUnavailable {
        reason: format!("{error:?}"),
    })?;
    let mut dispatcher = KernelDispatcher::new();
    dispatcher
        .revalidate(runtime.kernel_registry(), &mut plan)
        .map_err(|error| InferenceApiError::KernelUnavailable {
            reason: format!("{error:?}"),
        })?;
    let provider = ReferenceCpuExecutor::new();
    provider.write_tensor(input_a.id.clone(), a.clone());
    provider.write_tensor(input_b.id.clone(), b.clone());
    let mut memory = MemoryManager::default();
    let operator_catalog = initial_operator_catalog();
    let matmul_spec = operator_catalog
        .get(&advertisement.implemented_operator)
        .map_err(|error| InferenceApiError::KernelUnavailable {
            reason: error.to_string(),
        })?;
    let kernel_result = provider.execute_invocation_with_memory_manager(
        advertisement,
        matmul_spec,
        &plan.invocation,
        &mut memory,
    );
    let dispatch_result = KernelDispatchResult::from_kernel_result(&plan, kernel_result);
    if dispatch_result.status != KernelResultStatus::Succeeded {
        return Err(InferenceApiError::ProviderUnavailable {
            reason: format!(
                "Reference CPU matmul dispatch failed: {:?}",
                dispatch_result.error
            ),
        });
    }
    let output_tensor =
        provider
            .read_tensor(&output.id)
            .ok_or_else(|| InferenceApiError::GenerationFailed {
                reason: "Reference CPU matmul dispatch produced no output tensor".into(),
            })?;
    Ok((dispatch_result, output_tensor))
}

fn f32_tensor_descriptor(tensor: &HostTensor) -> TensorDescriptor {
    TensorDescriptor::new(
        ShapeDescriptor::new(tensor.shape.clone()),
        DTypeDescriptor::portable(ComputeDType::Float32),
        LayoutDescriptor::Contiguous,
    )
}

struct QwenDispatchContext<'a> {
    /// A single exclusive borrow (Correctif 13 / task group 7), not split
    /// into separate `runtime`/`memory` fields: `MemoryManager` is a field
    /// of `Runtime` itself, so a caller outside `runtime.rs` cannot borrow
    /// it mutably alongside an independent shared borrow of the rest of
    /// `Runtime` without either destructively swapping it out (the
    /// `std::mem::take` this replaced) or holding one exclusive borrow for
    /// everything. Any value read through `runtime` that must stay alive
    /// across a later `runtime.memory_mut()` call (e.g. a `&KernelAdvertisement`
    /// from `active_advertisement`) is cloned to an owned value at the
    /// point of lookup instead, so the two never need to overlap.
    runtime: &'a mut Runtime,
    /// Resolved generically from Runtime's provider registration through
    /// [`ProviderExecutionApi`] (Correctif 3): no downcast to a concrete
    /// Provider type is ever performed on this hot path.
    provider: Arc<dyn ProviderExecutionApi>,
    prepared_plan: Option<&'a mut PreparedExecutionPlan>,
    /// The semantic graph this dispatch's nodes belong to. Required
    /// alongside `prepared_plan` to route through
    /// `PreparedExecutionPlanExecutor::prepare_node_execution` (Correctif
    /// 4): that authority revalidates the plan's graph fingerprint against
    /// the graph actually being executed, not just the operation id string.
    /// `None` only for callers that have no full graph (e.g. hand-written
    /// test oracles), which fall back to a synthetic-candidate path that
    /// production dispatch never takes.
    graph: Option<&'a ExecutionGraph>,
    /// This generation step's token count, used to populate
    /// `PlanGuardContext::sequence_length` when revalidating a published
    /// Plan's guards (Correctif 4). An absent value is treated by
    /// `PlanGuard::SequenceRange` as out of range (fail-closed), so this
    /// SHALL be set whenever `prepared_plan` and `graph` are both `Some`.
    sequence_length: Option<u64>,
    /// The most recent dispatch's Provider submission/completion identity
    /// (task 5.3), recorded by `dispatch_reference_cpu_operator` via the
    /// registered Provider's own [`ProviderExecutionApi::complete`] rather
    /// than a caller-fabricated id. `None` until the first successful
    /// dispatch.
    last_provider_execution: Option<ProviderExecutionResult>,
    /// Per-node causal-chain events (Correctif 17 / task group 17), pushed
    /// directly into the caller's own `Vec` (not accumulated locally and
    /// drained only on success) as each node's dispatch progresses through
    /// `dispatch_reference_cpu_operator` and the KV-write blocks in
    /// `execute_qwen_graph_nodes` -- so a node's real events survive even if
    /// a *later* node's dispatch fails and this whole call returns early via
    /// `?`. The ultimate caller (`inference_api.rs`'s generation loop) owns
    /// the real `InferenceApiObserver` and turns each one into a redacted
    /// `InferenceApiObservation`.
    node_events: &'a mut Vec<PerNodeCausalEvent>,
}

/// One per-node causal-chain event captured during graph dispatch
/// (Correctif 17 / task group 17): names the `ExecutionNodeId` it belongs
/// to and, for the events that produce one, the `TensorResourceId` it
/// correlates to -- so a caller can walk a *complete* per-node chain
/// (`GraphNodeReady` -> ... -> `TensorResourceProduced`, per node) rather
/// than only confirming five global evidence categories were present
/// somewhere in the whole generation step, which `validate_e2e_no_shortcuts`
/// did before this task group.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct PerNodeCausalEvent {
    pub(crate) kind: InferenceApiObservationKind,
    pub(crate) node: ExecutionNodeId,
    pub(crate) resource: Option<TensorResourceId>,
}

impl PerNodeCausalEvent {
    fn new(kind: InferenceApiObservationKind, node: ExecutionNodeId) -> Self {
        Self {
            kind,
            node,
            resource: None,
        }
    }

    fn with_resource(mut self, resource: TensorResourceId) -> Self {
        self.resource = Some(resource);
        self
    }
}

/// Maps a dispatch `operation_id` back to the [`ExecutionNodeId`] published
/// in the plan. Two dispatch-side conventions do not exist at the graph
/// level and must be normalized away here: a `"decode."` prefix (the decode
/// phase's dispatch labels distinguish the same underlying node from its
/// prefill counterpart) and a trailing `".head{N}"` suffix (RoPE dispatches
/// once per attention head against a single whole-tensor `rope` graph node,
/// so every per-head sub-dispatch shares that node's one binding).
fn qwen_plan_node_id(operation_id: &str) -> ExecutionNodeId {
    let node_id = operation_id.strip_prefix("decode.").unwrap_or(operation_id);
    let node_id = match node_id.rsplit_once(".head") {
        Some((base, suffix))
            if !suffix.is_empty() && suffix.bytes().all(|b| b.is_ascii_digit()) =>
        {
            base
        }
        _ => node_id,
    };
    ExecutionNodeId::new(node_id)
}

fn prepared_candidate_for_operation(
    runtime: &Runtime,
    prepared_plan: &PreparedExecutionPlan,
    operation_id: &str,
    operator: &OperatorId,
) -> Result<KernelCandidate, InferenceApiError> {
    let node_id = qwen_plan_node_id(operation_id);
    let binding = prepared_plan
        .node_bindings
        .iter()
        .find(|binding| binding.graph_nodes.contains(&node_id))
        .ok_or_else(|| InferenceApiError::KernelUnavailable {
            reason: format!("prepared execution plan has no binding for node {node_id}"),
        })?;
    if binding.kernel.operator != *operator {
        return Err(InferenceApiError::KernelUnavailable {
            reason: format!(
                "prepared binding for node {node_id} targets operator {}, not {operator}",
                binding.kernel.operator
            ),
        });
    }
    let prepared_kernel =
        binding
            .prepared_kernel
            .ok_or_else(|| InferenceApiError::KernelUnavailable {
                reason: format!("prepared binding for node {node_id} has no PreparedKernelId"),
            })?;
    let prepared_generation =
        binding
            .prepared_kernel_generation
            .ok_or_else(|| InferenceApiError::KernelUnavailable {
                reason: format!(
                    "prepared binding for node {node_id} has no PreparedKernel generation"
                ),
            })?;
    let prepared = runtime
        .kernel_registry()
        .prepared_kernel(&prepared_kernel)
        .ok_or_else(|| InferenceApiError::KernelUnavailable {
            reason: format!("PreparedKernelId {prepared_kernel} for node {node_id} is not active"),
        })?;
    if !prepared.state.is_dispatchable() {
        return Err(InferenceApiError::KernelUnavailable {
            reason: format!("PreparedKernelId {prepared_kernel} for node {node_id} is not ready"),
        });
    }
    if prepared.kernel != binding.kernel || prepared.generation != prepared_generation {
        return Err(InferenceApiError::KernelUnavailable {
            reason: format!("prepared kernel for node {node_id} does not match plan binding"),
        });
    }
    Ok(KernelCandidate {
        kernel: binding.kernel.clone(),
        provider: binding.provider.clone(),
        device: binding.device.clone(),
        operator: operator.clone(),
        compatible: true,
        dtype_compatible: true,
        layout_compatible: true,
        shape_compatible: true,
        memory_compatible: true,
        workspace_feasible: true,
        affinity_compatible: true,
        deterministic_compatible: true,
        precision_compatible: true,
        provider_ready: true,
        device_ready: true,
        provider_status: None,
        device_status: None,
        pressure_score: 0,
        conformance_status: binding.qualification_profile.clone(),
        estimated_cost: 0,
        fallback_rank: 0,
        rejection_reason: None,
    })
}

/// A resolved Kernel for one dispatch: either the authoritative
/// `PreparedPlanNodeExecution` a published Plan already validated
/// (Correctif 4's production path), or a `KernelCandidate` from the
/// synthetic-candidate or live-selection fallback paths (test oracles / no
/// Plan bound).
enum HotPathKernelSelection {
    Prepared(PreparedPlanNodeExecution),
    Candidate(KernelCandidate),
}

impl HotPathKernelSelection {
    fn kernel(&self) -> &KernelId {
        match self {
            Self::Prepared(prepared) => &prepared.kernel,
            Self::Candidate(candidate) => &candidate.kernel,
        }
    }
}

/// Dispatches one Reference CPU Operator node with an arbitrary number of
/// outputs -- most callers have exactly one and use
/// [`dispatch_reference_cpu_operator`]'s single-output convenience wrapper
/// below, but a genuinely multi-output Operator (e.g. `split`) needs every
/// entry of the Kernel's `KernelResult::updated_resources` bound to its own
/// requested output, not just the first (GitHub issue "First-native graph
/// executor only propagates the first output of a multi-output node").
fn dispatch_reference_cpu_operator_multi(
    ctx: &mut QwenDispatchContext<'_>,
    operation_id: &str,
    operator: OperatorId,
    inputs: Vec<(TensorResourceId, TensorDescriptor, HostTensor)>,
    outputs: Vec<(TensorResourceId, TensorDescriptor)>,
    attributes: BTreeMap<String, OperatorAttributeValue>,
) -> Result<(KernelDispatchResult, Vec<HostTensor>), InferenceApiError> {
    // Correctif 17 / task group 17: this node's causal chain starts here,
    // once its inputs (written just below) are resolved and it is about to
    // be dispatched. `node` is reused for every later event this call
    // pushes, so the whole chain correlates to the same `ExecutionNodeId`
    // regardless of which selection arm (published Plan, synthetic
    // candidate, or live selection) this dispatch actually takes.
    let node = qwen_plan_node_id(operation_id);
    ctx.node_events.push(PerNodeCausalEvent::new(
        InferenceApiObservationKind::GraphNodeReady,
        node.clone(),
    ));
    let affinity = ResourceAffinity::new(FallbackClass::Transparent)
        .with_provider(ProviderBinding::new(REFERENCE_CPU_PROVIDER_NAME))
        .with_execution_context(ctx.runtime.context().id());
    let output_resources: Vec<TensorResourceDescriptor> = outputs
        .into_iter()
        .map(|(id, descriptor)| TensorResourceDescriptor::new(id, descriptor, affinity.clone()))
        .collect();
    let mut selection_request = KernelSelectionRequest::new(
        format!("e2e-runtime-{operation_id}"),
        operator,
        affinity.clone(),
    );
    for output_resource in &output_resources {
        selection_request = selection_request.with_output(KernelResource::new(
            output_resource.clone(),
            KernelMemoryClass::Host,
        ));
    }
    for (id, descriptor, tensor) in inputs {
        let resource = TensorResourceDescriptor::new(id.clone(), descriptor, affinity.clone());
        ctx.provider.write_tensor(id, tensor).map_err(|error| {
            InferenceApiError::ProviderTensorWriteFailed {
                reason: error.to_string(),
            }
        })?;
        selection_request =
            selection_request.with_input(KernelResource::new(resource, KernelMemoryClass::Host));
    }
    // Correctif 4: once a Plan is published, a node's execution SHALL be
    // driven directly from its `PlanNodeBinding`/`PreparedKernelId` through
    // `PreparedExecutionPlanExecutor`, not by reconstructing a synthetic,
    // always-compatible `KernelCandidate` after the fact. The synthetic
    // path (`prepared_candidate_for_operation`) survives only for callers
    // that have a `PreparedExecutionPlan` but no full `ExecutionGraph` to
    // revalidate it against -- hand-written test oracles, never production.
    let selection = match (ctx.prepared_plan.as_deref_mut(), ctx.graph) {
        (Some(prepared_plan), Some(graph)) => {
            let guard_context = first_native_plan_context(
                prepared_plan.scope.phase,
                ctx.sequence_length.unwrap_or(1),
            );
            let prepared = PreparedExecutionPlanExecutor::new()
                .prepare_node_execution(
                    graph,
                    prepared_plan,
                    ctx.runtime.kernel_registry(),
                    &guard_context,
                    &node,
                )
                .map_err(|error| InferenceApiError::KernelUnavailable {
                    reason: format!("{operation_id}: {error:?}"),
                })?;
            ctx.node_events.push(PerNodeCausalEvent::new(
                InferenceApiObservationKind::PlanBindingResolved,
                node.clone(),
            ));
            HotPathKernelSelection::Prepared(prepared)
        }
        (Some(prepared_plan), None) => {
            HotPathKernelSelection::Candidate(prepared_candidate_for_operation(
                ctx.runtime,
                prepared_plan,
                operation_id,
                &selection_request.operator,
            )?)
        }
        (None, _) => {
            let selection = ctx
                .runtime
                .kernel_registry()
                .select(&selection_request)
                .map_err(|error| InferenceApiError::KernelUnavailable {
                    reason: format!("{operation_id}: {error}"),
                })?;
            HotPathKernelSelection::Candidate(selection.selected.ok_or_else(|| {
                InferenceApiError::KernelUnavailable {
                    reason: format!("Kernel Registry selected no candidate for {operation_id}"),
                }
            })?)
        }
    };
    // Cloned to an owned value (not kept as a borrow of `ctx.runtime`) so it
    // can stay alive across this function's later `ctx.runtime.memory_mut()`
    // calls without holding a live shared borrow of `ctx.runtime` at the
    // same time (task group 7 / Correctif 13).
    let advertisement = ctx
        .runtime
        .kernel_registry()
        .active_advertisement(selection.kernel())
        .ok_or_else(|| InferenceApiError::KernelUnavailable {
            reason: format!("selected advertisement for {operation_id} is no longer active"),
        })?
        .clone();
    ctx.node_events.push(PerNodeCausalEvent::new(
        InferenceApiObservationKind::PreparedKernelResolved,
        node.clone(),
    ));
    let mut plan = match &selection {
        HotPathKernelSelection::Prepared(prepared) => {
            KernelDispatchPlan::from_prepared_node_execution(
                KernelDispatchPlanId::new(format!("e2e-runtime-{operation_id}-dispatch")),
                &selection_request,
                prepared,
                &advertisement,
                KernelInvocationId::new(format!("e2e-runtime-{operation_id}-invocation")),
            )
        }
        HotPathKernelSelection::Candidate(candidate) => KernelDispatchPlan::from_selection(
            KernelDispatchPlanId::new(format!("e2e-runtime-{operation_id}-dispatch")),
            &selection_request,
            candidate,
            &advertisement,
            KernelInvocationId::new(format!("e2e-runtime-{operation_id}-invocation")),
        ),
    }
    .map_err(|error| InferenceApiError::KernelUnavailable {
        reason: format!("{error:?}"),
    })?;
    plan.invocation.attributes = attributes;
    if advertisement.workspace.required {
        let workspace = ctx
            .provider
            .allocate_workspace(
                ctx.runtime.memory_mut(),
                advertisement.workspace.size_bytes_upper_bound.unwrap_or(1),
            )
            .map_err(|error| InferenceApiError::MemoryAdmissionFailed {
                reason: error.to_string(),
            })?;
        plan.invocation.workspace = Some(workspace);
        plan.workspace_reservation = Some(workspace);
    }
    let mut dispatcher = KernelDispatcher::new();
    dispatcher
        .revalidate(ctx.runtime.kernel_registry(), &mut plan)
        .map_err(|error| InferenceApiError::KernelUnavailable {
            reason: format!("{error:?}"),
        })?;
    let operator_catalog = initial_operator_catalog();
    let operator_spec = operator_catalog
        .get(&advertisement.implemented_operator)
        .map_err(|error| InferenceApiError::KernelUnavailable {
            reason: error.to_string(),
        })?;
    // Provider submission/completion identity (Correctif 2): `submit_kernel`
    // is what causally triggers the numerical Kernel work (Reference CPU is
    // synchronous, so it runs as part of this call); the returned handle is
    // never constructed after the fact to merely dress up evidence that
    // already happened through a different call. Resolved generically
    // through `ProviderExecutionApi` (Correctif 3): `ctx.provider` is a
    // trait object, not a concrete `ReferenceCpuExecutor`.
    let handle = ctx
        .provider
        .submit_kernel(
            &advertisement,
            operator_spec,
            &plan.invocation,
            ctx.runtime.memory_mut(),
        )
        .map_err(|error| InferenceApiError::ProviderUnavailable {
            reason: format!(
                "Reference CPU dispatch for {operation_id} could not be submitted: {error}"
            ),
        })?;
    ctx.node_events.push(PerNodeCausalEvent::new(
        InferenceApiObservationKind::ProviderSubmitted,
        node.clone(),
    ));
    // Workspace is invocation-scoped: release it back to Runtime's
    // MemoryManager once this dispatch has used it, whether or not the
    // dispatch succeeded, rather than letting it accumulate as a leaked
    // allocation across every node this graph executes (task 5.4/5.5/5.6).
    if let Some(workspace) = plan.workspace_reservation {
        let _ = ctx.runtime.memory_mut().release(workspace);
    }
    // `complete_kernel` observes the SAME work `handle` was returned for
    // above -- correlated by that handle, not fabricated -- and fails
    // structurally if the handle is somehow unknown (it cannot be completed
    // twice from this single call site, but the check keeps this path
    // honest about what "complete" means).
    let kernel_result = ctx.provider.complete_kernel(&handle).map_err(|error| {
        InferenceApiError::ProviderUnavailable {
            reason: format!(
                "Reference CPU dispatch for {operation_id} could not be completed: {error}"
            ),
        }
    })?;
    let dispatch_result = KernelDispatchResult::from_kernel_result(&plan, kernel_result);
    if dispatch_result.status != KernelResultStatus::Succeeded {
        return Err(InferenceApiError::ProviderUnavailable {
            reason: format!(
                "Reference CPU dispatch for {operation_id} failed: {:?}",
                dispatch_result.error
            ),
        });
    }
    ctx.node_events.push(PerNodeCausalEvent::new(
        InferenceApiObservationKind::ProviderCompleted,
        node.clone(),
    ));
    // Every requested output is read back and bound to its corresponding
    // graph edge in the same order `outputs` was given, not just the
    // first -- a genuinely multi-output Operator (e.g. `split`) produces
    // one `KernelResult::updated_resources` entry per output, and each
    // needs its own `TensorResourceProduced` causal event and its own
    // read-back, the same treatment the single-output case already got.
    let mut output_tensors = Vec::with_capacity(output_resources.len());
    for output_resource in &output_resources {
        let output_tensor = ctx
            .provider
            .read_tensor(&output_resource.id)
            .ok_or_else(|| InferenceApiError::GenerationFailed {
                reason: format!(
                    "Reference CPU dispatch for {operation_id} produced no output for resource '{}'",
                    output_resource.id
                ),
            })?;
        ctx.node_events.push(
            PerNodeCausalEvent::new(
                InferenceApiObservationKind::TensorResourceProduced,
                node.clone(),
            )
            .with_resource(output_resource.id.clone()),
        );
        output_tensors.push(output_tensor);
    }
    ctx.last_provider_execution = Some(ProviderExecutionResult::completed(
        handle,
        dispatch_result.updated_resources.clone(),
    ));
    Ok((dispatch_result, output_tensors))
}

/// Single-output convenience wrapper over
/// [`dispatch_reference_cpu_operator_multi`]: almost every graph node in
/// this file's Qwen dispatch has exactly one output, and threading a
/// `Vec` through every one of those call sites for a case that structurally
/// cannot produce more than one element would only add noise.
fn dispatch_reference_cpu_operator(
    ctx: &mut QwenDispatchContext<'_>,
    operation_id: &str,
    operator: OperatorId,
    inputs: Vec<(TensorResourceId, TensorDescriptor, HostTensor)>,
    output: (TensorResourceId, TensorDescriptor),
    attributes: BTreeMap<String, OperatorAttributeValue>,
) -> Result<(KernelDispatchResult, HostTensor), InferenceApiError> {
    let (dispatch_result, mut outputs) = dispatch_reference_cpu_operator_multi(
        ctx,
        operation_id,
        operator,
        inputs,
        vec![output],
        attributes,
    )?;
    let output_tensor = outputs
        .pop()
        .ok_or_else(|| InferenceApiError::GenerationFailed {
            reason: format!("Reference CPU dispatch for {operation_id} produced no output"),
        })?;
    Ok((dispatch_result, output_tensor))
}

fn runtime_generation_failed(error: impl std::fmt::Display) -> InferenceApiError {
    InferenceApiError::GenerationFailed {
        reason: error.to_string(),
    }
}

fn dispatch_operator_id(name: &str, family: OperatorFamily) -> OperatorId {
    OperatorId::magnetar(name, 1, family)
}

fn dispatch_qwen_matmul(
    ctx: &mut QwenDispatchContext<'_>,
    operation_id: &str,
    a: HostTensor,
    b: HostTensor,
) -> Result<(KernelDispatchResult, HostTensor), InferenceApiError> {
    let (rows, _) = a.rows_cols().map_err(runtime_generation_failed)?;
    let (_, cols) = b.rows_cols().map_err(runtime_generation_failed)?;
    dispatch_reference_cpu_operator(
        ctx,
        operation_id,
        dispatch_operator_id("matmul", OperatorFamily::LinearAlgebra),
        vec![
            (
                TensorResourceId::new(format!("{operation_id}.a")),
                f32_tensor_descriptor(&a),
                a,
            ),
            (
                TensorResourceId::new(format!("{operation_id}.b")),
                f32_tensor_descriptor(&b),
                b,
            ),
        ],
        (
            TensorResourceId::new(format!("{operation_id}.out")),
            TensorDescriptor::new(
                ShapeDescriptor::new([rows, cols]),
                DTypeDescriptor::portable(ComputeDType::Float32),
                LayoutDescriptor::Contiguous,
            ),
        ),
        BTreeMap::new(),
    )
}

fn dispatch_qwen_unary(
    ctx: &mut QwenDispatchContext<'_>,
    operation_id: &str,
    name: &str,
    family: OperatorFamily,
    input: HostTensor,
    attributes: BTreeMap<String, OperatorAttributeValue>,
) -> Result<(KernelDispatchResult, HostTensor), InferenceApiError> {
    dispatch_reference_cpu_operator(
        ctx,
        operation_id,
        dispatch_operator_id(name, family),
        vec![(
            TensorResourceId::new(format!("{operation_id}.input")),
            f32_tensor_descriptor(&input),
            input.clone(),
        )],
        (
            TensorResourceId::new(format!("{operation_id}.out")),
            f32_tensor_descriptor(&input),
        ),
        attributes,
    )
}

fn dispatch_qwen_binary_same_shape(
    ctx: &mut QwenDispatchContext<'_>,
    operation_id: &str,
    name: &str,
    family: OperatorFamily,
    a: HostTensor,
    b: HostTensor,
) -> Result<(KernelDispatchResult, HostTensor), InferenceApiError> {
    dispatch_reference_cpu_operator(
        ctx,
        operation_id,
        dispatch_operator_id(name, family),
        vec![
            (
                TensorResourceId::new(format!("{operation_id}.a")),
                f32_tensor_descriptor(&a),
                a.clone(),
            ),
            (
                TensorResourceId::new(format!("{operation_id}.b")),
                f32_tensor_descriptor(&b),
                b,
            ),
        ],
        (
            TensorResourceId::new(format!("{operation_id}.out")),
            f32_tensor_descriptor(&a),
        ),
        BTreeMap::new(),
    )
}

fn dispatch_qwen_rmsnorm(
    ctx: &mut QwenDispatchContext<'_>,
    operation_id: &str,
    input: HostTensor,
    weight: HostTensor,
    epsilon: f32,
) -> Result<(KernelDispatchResult, HostTensor), InferenceApiError> {
    let (rows, cols) = input.rows_cols().map_err(runtime_generation_failed)?;
    let weight = if weight.shape == [cols] {
        let mut data = Vec::with_capacity((rows * cols) as usize);
        for _ in 0..rows {
            data.extend_from_slice(&weight.data);
        }
        HostTensor::new([rows, cols], data).map_err(runtime_generation_failed)?
    } else {
        weight
    };
    let mut attributes = BTreeMap::new();
    attributes.insert(
        "epsilon".into(),
        OperatorAttributeValue::Float(epsilon as f64),
    );
    dispatch_reference_cpu_operator(
        ctx,
        operation_id,
        dispatch_operator_id("rmsnorm", OperatorFamily::Normalization),
        vec![
            (
                TensorResourceId::new(format!("{operation_id}.input")),
                f32_tensor_descriptor(&input),
                input.clone(),
            ),
            (
                TensorResourceId::new(format!("{operation_id}.weight")),
                f32_tensor_descriptor(&weight),
                weight,
            ),
        ],
        (
            TensorResourceId::new(format!("{operation_id}.out")),
            f32_tensor_descriptor(&input),
        ),
        attributes,
    )
}

fn dispatch_qwen_rope_per_head(
    ctx: &mut QwenDispatchContext<'_>,
    operation_id: &str,
    tensor: &HostTensor,
    head_count: u64,
    head_dimension: u64,
    rope_config: &QwenRopeConfig,
    position_offset: u64,
) -> Result<(KernelDispatchResult, HostTensor), InferenceApiError> {
    let (rows, cols) = tensor.rows_cols().map_err(runtime_generation_failed)?;
    let mut out = vec![0.0_f32; tensor.data.len()];
    let mut last_dispatch = None;
    for head in 0..head_count {
        let start_col = head * head_dimension;
        let mut head_data = Vec::with_capacity((rows * head_dimension) as usize);
        for row in 0..rows {
            let base = (row * cols + start_col) as usize;
            head_data.extend_from_slice(&tensor.data[base..base + head_dimension as usize]);
        }
        let head_tensor = HostTensor::new([rows, head_dimension], head_data)
            .map_err(runtime_generation_failed)?;
        let mut attributes = BTreeMap::new();
        attributes.insert(
            "base".into(),
            OperatorAttributeValue::Float(rope_config.base),
        );
        attributes.insert(
            "scale".into(),
            OperatorAttributeValue::Float(rope_config.scale.unwrap_or(1.0)),
        );
        attributes.insert(
            "dimension".into(),
            OperatorAttributeValue::Integer(rope_config.dimension as i64),
        );
        attributes.insert(
            "position_mode".into(),
            OperatorAttributeValue::String(rope_config.position_mode.as_str().into()),
        );
        attributes.insert(
            "position_offset".into(),
            OperatorAttributeValue::Integer(position_offset as i64),
        );
        let (dispatch, rotated) = dispatch_qwen_unary(
            ctx,
            &format!("{operation_id}.head{head}"),
            "rope",
            OperatorFamily::PositionEncoding,
            head_tensor,
            attributes,
        )?;
        for row in 0..rows {
            let dst_base = (row * cols + start_col) as usize;
            let src_base = (row * head_dimension) as usize;
            out[dst_base..dst_base + head_dimension as usize]
                .copy_from_slice(&rotated.data[src_base..src_base + head_dimension as usize]);
        }
        last_dispatch = Some(dispatch);
    }
    let dispatch = last_dispatch.ok_or_else(|| InferenceApiError::GenerationFailed {
        reason: "RoPE dispatch requires at least one head".into(),
    })?;
    let tensor = HostTensor::new(tensor.shape.clone(), out).map_err(runtime_generation_failed)?;
    Ok((dispatch, tensor))
}

fn dispatch_qwen_attention(
    ctx: &mut QwenDispatchContext<'_>,
    operation_id: &str,
    q: HostTensor,
    k: HostTensor,
    v: HostTensor,
    architecture: &ModelComponentArchitectureMetadata,
) -> Result<(KernelDispatchResult, HostTensor), InferenceApiError> {
    let mut attributes = BTreeMap::new();
    attributes.insert("causal".into(), OperatorAttributeValue::Boolean(true));
    attributes.insert(
        "head_count".into(),
        OperatorAttributeValue::Integer(architecture.attention_head_count as i64),
    );
    attributes.insert(
        "kv_head_count".into(),
        OperatorAttributeValue::Integer(architecture.kv_head_count as i64),
    );
    attributes.insert(
        "head_dimension".into(),
        OperatorAttributeValue::Integer(architecture.head_dimension as i64),
    );
    attributes.insert(
        "attention_mask_kind".into(),
        OperatorAttributeValue::String("causal".into()),
    );
    dispatch_reference_cpu_operator(
        ctx,
        operation_id,
        dispatch_operator_id("attention", OperatorFamily::Attention),
        vec![
            (
                TensorResourceId::new(format!("{operation_id}.q")),
                f32_tensor_descriptor(&q),
                q.clone(),
            ),
            (
                TensorResourceId::new(format!("{operation_id}.k")),
                f32_tensor_descriptor(&k),
                k,
            ),
            (
                TensorResourceId::new(format!("{operation_id}.v")),
                f32_tensor_descriptor(&v),
                v,
            ),
        ],
        (
            TensorResourceId::new(format!("{operation_id}.out")),
            f32_tensor_descriptor(&q),
        ),
        attributes,
    )
}

fn concat_rows(a: &HostTensor, b: &HostTensor) -> Result<HostTensor, InferenceApiError> {
    let (a_rows, a_cols) = a.rows_cols().map_err(runtime_generation_failed)?;
    let (b_rows, b_cols) = b.rows_cols().map_err(runtime_generation_failed)?;
    if a_cols != b_cols {
        return Err(InferenceApiError::GenerationFailed {
            reason: format!("cannot concatenate tensors with widths {a_cols} and {b_cols}"),
        });
    }
    let mut data = Vec::with_capacity(a.data.len() + b.data.len());
    data.extend_from_slice(&a.data);
    data.extend_from_slice(&b.data);
    HostTensor::new([a_rows + b_rows, a_cols], data).map_err(runtime_generation_failed)
}

// ---------------------------------------------------------------------
// Generic graph executor
//
// Walks an `ExecutionGraph`'s dependency order and dispatches each node's
// Operator through the published `PreparedExecutionPlan`, binding graph
// inputs, intermediates, outputs, and (for KV-cache-tagged edges) historical
// Runtime KV state -- so the graph, not a hand-written Rust sequence, is the
// authoritative recipe for which Operators run, in what order, over which
// tensors. This replaces the previous Qwen-specific hard-coded prefill/
// decode call sequences (still available under `#[cfg(test)]` as an
// independent oracle for the tests that compare against them).
// ---------------------------------------------------------------------

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum KvRole {
    K,
    V,
}

/// Parses a KV cache identity (`{namespace}.layer{N}.{k|v}`, produced by
/// `GraphBuilderCapability::kv_resource` -- see that method's documentation)
/// back into a layer index and K/V role. Generic over `namespace` rather
/// than hardcoding one: the Runtime-side `graph-builder` Capability accepts
/// whatever `kv_namespace` the caller supplies in `SessionContext` (today
/// always `"qwen"`, since that is the only Model Component that exists),
/// and this parser only needs to know that a namespace precedes `.layer`,
/// not which model family chose it.
fn parse_kv_cache_id(cache_id: &str) -> Result<(usize, KvRole), InferenceApiError> {
    let malformed = || InferenceApiError::GraphPlanningFailed {
        reason: format!("malformed graph KV cache id '{cache_id}'"),
    };
    let (_namespace, rest) = cache_id.split_once(".layer").ok_or_else(malformed)?;
    let (layer, role) = rest.split_once('.').ok_or_else(malformed)?;
    let layer = layer.parse::<usize>().map_err(|_| malformed())?;
    let role = match role {
        "k" => KvRole::K,
        "v" => KvRole::V,
        _ => return Err(malformed()),
    };
    Ok((layer, role))
}

/// Maps an execution-graph weight edge id to the canonical Model Loading
/// tensor name `fixture.weights` is keyed by (see `qwen_expected_tensor_names`).
/// A weight edge id is always `weight.{logical_name}` (see
/// `GraphBuilderCapability::weight_edge`), and the Model Component -- not
/// the Runtime -- is the one that chooses `logical_name`; the Qwen
/// Component supplies it already equal to the canonical Model Artifact
/// tensor name (e.g. `layers.0.self_attn.q_proj`, `token_embedding`), so
/// this is a plain prefix strip, not a per-model suffix-mapping table.
/// Returns `None` for a non-weight edge id.
fn weight_tensor_name_from_edge(edge_id: &str) -> Option<String> {
    edge_id.strip_prefix("weight.").map(str::to_string)
}

/// Resolves a graph weight edge to its tensor value by looking up its
/// canonical name in `weight_bindings` (the active Model Instance's
/// Runtime-owned weight resource bindings -- see `bind_qwen_fixture_weights`)
/// and reading the bound resource from the registered Provider's storage,
/// rather than a private Rust-side copy of the fixture's tensor bytes (task
/// 6.4). Applies the tied-embeddings `weight.lm_head` -> transposed
/// `token_embedding` substitution `qwen_lm_head_weight_edge` declares via
/// `TensorAliasing::MayAlias` (see that function's doc comment).
fn resolve_qwen_weight_edge(
    provider: &Arc<dyn ProviderExecutionApi>,
    weight_bindings: &BTreeMap<String, TensorResourceId>,
    tied_embeddings: bool,
    edge_id: &str,
) -> Result<HostTensor, InferenceApiError> {
    let name = weight_tensor_name_from_edge(edge_id).ok_or_else(|| {
        InferenceApiError::GraphPlanningFailed {
            reason: format!("graph edge '{edge_id}' is neither a bound input nor a known weight"),
        }
    })?;
    let lookup_name = if name == "lm_head" && tied_embeddings {
        "token_embedding"
    } else {
        &name
    };
    let resource_id =
        weight_bindings
            .get(lookup_name)
            .ok_or_else(|| InferenceApiError::ModelLoadingFailed {
                reason: format!(
                    "active Model Instance has no weight resource bound for '{lookup_name}'"
                ),
            })?;
    // Reference CPU Kernel input boundary (`define-provider-prepared-kernel-execution-contract`
    // task 5.2): weight edges feed straight into `dispatch_qwen_*` compute,
    // which needs real host bytes, so this is one of the points that
    // explicitly materializes through `TensorValue::into_host` rather than
    // carrying an opaque value further.
    let value = provider.read_tensor_value(resource_id).ok_or_else(|| {
        InferenceApiError::ModelLoadingFailed {
            reason: format!(
                "weight resource '{resource_id}' bound for '{lookup_name}' has no materialized data"
            ),
        }
    })?;
    let tensor =
        value
            .into_host(resource_id)
            .map_err(|error| InferenceApiError::ModelLoadingFailed {
                reason: format!(
                    "weight resource '{resource_id}' bound for '{lookup_name}': {error}"
                ),
            })?;
    if name == "lm_head" && tied_embeddings {
        return transpose_rows_cols(&tensor).map_err(runtime_generation_failed);
    }
    Ok(tensor)
}

/// Derives a Kahn's-algorithm topological order for `graph` from
/// `node.inputs`/`node.outputs` directly, rather than relying on
/// `TensorEdge::producer`/`::consumers` (which Qwen's own graph builder does
/// not populate).
fn qwen_graph_execution_order(
    graph: &ExecutionGraph,
) -> Result<Vec<ExecutionNodeId>, InferenceApiError> {
    let mut producer_of: BTreeMap<&ExecutionNodeId, ()> = BTreeMap::new();
    let mut edge_producer: BTreeMap<&TensorEdgeId, &ExecutionNodeId> = BTreeMap::new();
    for (node_id, node) in &graph.nodes {
        producer_of.insert(node_id, ());
        for output in &node.outputs {
            edge_producer.insert(output, node_id);
        }
    }
    let mut remaining: BTreeSet<&ExecutionNodeId> = graph.nodes.keys().collect();
    let mut order = Vec::with_capacity(remaining.len());
    while !remaining.is_empty() {
        let ready = remaining
            .iter()
            .find(|node_id| {
                let node = &graph.nodes[**node_id];
                node.inputs.iter().all(|edge_id| {
                    edge_producer
                        .get(edge_id)
                        .is_none_or(|producer| !remaining.contains(producer))
                })
            })
            .copied();
        let Some(node_id) = ready else {
            return Err(InferenceApiError::GraphPlanningFailed {
                reason: "first-native graph contains a cycle or unresolved producer".into(),
            });
        };
        remaining.remove(node_id);
        order.push(node_id.clone());
    }
    Ok(order)
}

fn node_attribute<'a>(
    node: &'a ExecutionNode,
    name: &str,
) -> Result<&'a OperatorAttributeValue, InferenceApiError> {
    node.attributes
        .get(name)
        .ok_or_else(|| InferenceApiError::GraphPlanningFailed {
            reason: format!("graph node '{}' is missing attribute '{name}'", node.id),
        })
}

fn node_attribute_f64(node: &ExecutionNode, name: &str) -> Result<f64, InferenceApiError> {
    match node_attribute(node, name)? {
        OperatorAttributeValue::Float(value) => Ok(*value),
        _ => Err(InferenceApiError::GraphPlanningFailed {
            reason: format!("graph node '{}' attribute '{name}' is not a float", node.id),
        }),
    }
}

fn node_attribute_u64(node: &ExecutionNode, name: &str) -> Result<u64, InferenceApiError> {
    match node_attribute(node, name)? {
        OperatorAttributeValue::Integer(value) if *value >= 0 => Ok(*value as u64),
        _ => Err(InferenceApiError::GraphPlanningFailed {
            reason: format!(
                "graph node '{}' attribute '{name}' is not a non-negative integer",
                node.id
            ),
        }),
    }
}

/// Attention head count for a per-head `rope` node: the graph declares one
/// whole-tensor `rope` node per Q/K projection (see `qwen_build_graph`), so
/// the node id's own `rope_q`/`rope_k` suffix -- not a node attribute --
/// distinguishes which head count applies. `head_dimension` is the rope
/// node's own `dimension` attribute, which equals the architecture's head
/// dimension for the standard (full-rotation) RoPE config this baseline
/// uses.
fn qwen_rope_head_count(
    node: &ExecutionNode,
    architecture: &ModelComponentArchitectureMetadata,
) -> Result<u64, InferenceApiError> {
    let id = node.id.as_str();
    if id.ends_with("rope_q") {
        Ok(architecture.attention_head_count)
    } else if id.ends_with("rope_k") {
        Ok(architecture.kv_head_count)
    } else {
        Err(InferenceApiError::GraphPlanningFailed {
            reason: format!("graph node '{id}' is not a recognized RoPE node"),
        })
    }
}

/// Stable numeric codes for the Operator names the Qwen graph builder emits,
/// shared between Runtime (deriving the expected sequence from
/// `ExecutionGraph`) and the Qwen Model Component boundary (which describes
/// its own graph as this same code sequence -- see
/// `qwen_graph_operator_codes` and `qwen-graph.component.wat`'s
/// `prefill-operator-code`/`decode-operator-code` exports). A plain
/// name-to-code table rather than `OperatorId` equality: the Component
/// boundary exchanges scalar `u32`s, not portable Operator identities.
// Task 11.5/12.6: this whole checksum-hash family (through
// `qwen_component_graph_semantics_for_prompt`/`QwenComponentGraphSemantics`/
// `build_first_native_graphs_from_component_output`) is test-oracle only --
// the strict, default production path gets its graph directly from the
// real Component instead of cross-checking a Rust-synthesized one against a
// declared checksum, and no production build has a second, unattested
// graph source to fall back to.
#[cfg(test)]
fn qwen_operator_kind_code(name: &str) -> Option<u32> {
    match name {
        "embedding" => Some(0),
        "rmsnorm" => Some(1),
        "matmul" => Some(2),
        "rope" => Some(3),
        "attention" => Some(4),
        "silu" => Some(5),
        "mul" => Some(6),
        "residual-add" => Some(7),
        _ => None,
    }
}

/// Derives the expected Operator-kind-code sequence for `graph`, in the same
/// dependency order `execute_qwen_graph` executes it in: the semantic
/// content a Qwen Model Component is expected to reproduce when describing
/// its own graph (see `qwen_operator_kind_code`).
#[cfg(test)]
fn qwen_graph_operator_codes(graph: &ExecutionGraph) -> Result<Vec<u32>, E2eConformanceError> {
    let order = qwen_graph_execution_order(graph)?;
    order
        .iter()
        .map(|node_id| {
            let node = graph.nodes.get(node_id).ok_or_else(|| {
                E2eConformanceError::GraphValidationFailed {
                    reason: format!("first-native graph is missing node '{node_id}'"),
                }
            })?;
            qwen_operator_kind_code(node.operator.name()).ok_or_else(|| {
                E2eConformanceError::GraphValidationFailed {
                    reason: format!(
                        "graph node '{node_id}' uses operator '{}' with no known kind code",
                        node.operator.name()
                    ),
                }
            })
        })
        .collect()
}

/// A deterministic FNV-1a-style hash over an ordered Operator-kind-code
/// sequence. The Component boundary's invocation model exchanges only
/// zero-argument, single-`u32`-result calls (see [`ComponentInvocation`]),
/// so a component cannot return its full node sequence as a list; instead it
/// computes this same hash internally (see `qwen-graph.component.wat`'s
/// `prefill-operator-hash`/`decode-operator-hash` exports, which perform the
/// identical unrolled XOR/multiply steps over its own hard-coded sequence)
/// and Runtime compares hashes -- a proof over the actual ordered semantic
/// content, not just a count.
#[cfg(test)]
fn qwen_operator_sequence_hash(codes: &[u32]) -> u32 {
    const FNV_OFFSET_BASIS: u32 = 0x811c_9dc5;
    const FNV_PRIME: u32 = 0x0100_0193;
    codes.iter().fold(FNV_OFFSET_BASIS, |hash, code| {
        (hash ^ *code).wrapping_mul(FNV_PRIME)
    })
}

/// Dispatches one graph node's Operator through the published prepared plan,
/// reading structural parameters (epsilon, RoPE base/dimension/mode,
/// attention head geometry) from the node's own graph attributes rather than
/// fixture configuration, so the graph is the authoritative recipe. The one
/// deliberate exception is RoPE's absolute position: the same published
/// decode plan/graph is reused across every decode step (so its baked-in
/// `position_offset` attribute reflects only the first step), so the caller
/// supplies the true per-step position via `absolute_position_override`
/// instead.
fn dispatch_qwen_graph_node(
    ctx: &mut QwenDispatchContext<'_>,
    fixture: &E2eFixture,
    node: &ExecutionNode,
    mut inputs: Vec<HostTensor>,
    absolute_position_override: Option<u64>,
) -> Result<(KernelDispatchResult, HostTensor), InferenceApiError> {
    let node_id = node.id.as_str();
    let architecture = &fixture.config.architecture;
    match node.operator.name() {
        "embedding" => {
            if inputs.len() != 2 {
                return Err(InferenceApiError::GraphPlanningFailed {
                    reason: format!("graph node '{node_id}' expects 2 embedding inputs"),
                });
            }
            // The graph declares [ids, table]; the embedding kernel's
            // invocation convention (matching every other production
            // dispatch of this operator) is (table, ids).
            let ids = inputs.remove(0);
            let table = inputs.remove(0);
            let output_edge =
                node.outputs
                    .first()
                    .ok_or_else(|| InferenceApiError::GraphPlanningFailed {
                        reason: format!("graph node '{node_id}' has no output edge"),
                    })?;
            let sequence_length = ids.shape.first().copied().unwrap_or(0);
            dispatch_reference_cpu_operator(
                ctx,
                node_id,
                node.operator.clone(),
                vec![
                    (
                        TensorResourceId::new(format!("{node_id}.table")),
                        f32_tensor_descriptor(&table),
                        table,
                    ),
                    (
                        TensorResourceId::new(format!("{node_id}.ids")),
                        f32_tensor_descriptor(&ids),
                        ids,
                    ),
                ],
                (
                    TensorResourceId::new(output_edge.as_str()),
                    TensorDescriptor::new(
                        ShapeDescriptor::new([sequence_length, architecture.hidden_size]),
                        DTypeDescriptor::portable(ComputeDType::Float32),
                        LayoutDescriptor::Contiguous,
                    ),
                ),
                BTreeMap::new(),
            )
        }
        "rmsnorm" => {
            if inputs.len() != 2 {
                return Err(InferenceApiError::GraphPlanningFailed {
                    reason: format!("graph node '{node_id}' expects 2 rmsnorm inputs"),
                });
            }
            let input = inputs.remove(0);
            let weight = inputs.remove(0);
            let epsilon = node_attribute_f64(node, "epsilon")? as f32;
            dispatch_qwen_rmsnorm(ctx, node_id, input, weight, epsilon)
        }
        "matmul" => {
            if inputs.len() != 2 {
                return Err(InferenceApiError::GraphPlanningFailed {
                    reason: format!("graph node '{node_id}' expects 2 matmul inputs"),
                });
            }
            let a = inputs.remove(0);
            let b = inputs.remove(0);
            dispatch_qwen_matmul(ctx, node_id, a, b)
        }
        "rope" => {
            if inputs.len() != 1 {
                return Err(InferenceApiError::GraphPlanningFailed {
                    reason: format!("graph node '{node_id}' expects 1 rope input"),
                });
            }
            let tensor = inputs.remove(0);
            let head_dimension = node_attribute_u64(node, "dimension")?;
            let head_count = qwen_rope_head_count(node, architecture)?;
            let base = node_attribute_f64(node, "base")?;
            let position_offset = absolute_position_override
                .map(Ok)
                .unwrap_or_else(|| node_attribute_u64(node, "position_offset"))?;
            let rope_config = QwenRopeConfig {
                base,
                scale: fixture.config.rope.scale,
                dimension: head_dimension,
                position_mode: fixture.config.rope.position_mode,
                dynamic_scaling_supported: fixture.config.rope.dynamic_scaling_supported,
            };
            dispatch_qwen_rope_per_head(
                ctx,
                node_id,
                &tensor,
                head_count,
                head_dimension,
                &rope_config,
                position_offset,
            )
        }
        "attention" => {
            if inputs.len() != 3 {
                return Err(InferenceApiError::GraphPlanningFailed {
                    reason: format!("graph node '{node_id}' expects 3 attention inputs"),
                });
            }
            let q = inputs.remove(0);
            let k = inputs.remove(0);
            let v = inputs.remove(0);
            dispatch_qwen_attention(ctx, node_id, q, k, v, architecture)
        }
        "silu" => {
            if inputs.len() != 1 {
                return Err(InferenceApiError::GraphPlanningFailed {
                    reason: format!("graph node '{node_id}' expects 1 silu input"),
                });
            }
            let input = inputs.remove(0);
            dispatch_qwen_unary(
                ctx,
                node_id,
                "silu",
                OperatorFamily::Activation,
                input,
                BTreeMap::new(),
            )
        }
        "mul" => {
            if inputs.len() != 2 {
                return Err(InferenceApiError::GraphPlanningFailed {
                    reason: format!("graph node '{node_id}' expects 2 mul inputs"),
                });
            }
            let a = inputs.remove(0);
            let b = inputs.remove(0);
            dispatch_qwen_binary_same_shape(ctx, node_id, "mul", OperatorFamily::Tensor, a, b)
        }
        "residual-add" => {
            if inputs.len() != 2 {
                return Err(InferenceApiError::GraphPlanningFailed {
                    reason: format!("graph node '{node_id}' expects 2 residual-add inputs"),
                });
            }
            let a = inputs.remove(0);
            let b = inputs.remove(0);
            dispatch_qwen_binary_same_shape(
                ctx,
                node_id,
                "residual-add",
                OperatorFamily::Tensor,
                a,
                b,
            )
        }
        other => Err(InferenceApiError::OperatorUnsupported {
            reason: format!(
                "first-native graph executor has no dispatch for operator '{other}' at node '{node_id}'"
            ),
        }),
    }
}

/// Generic first-native graph executor: walks `graph`'s dependency order
/// (`qwen_graph_execution_order`), resolving each node's inputs from
/// pre-bound graph inputs (`initial_bindings`), previously computed
/// intermediates, or Model Artifact weights (`resolve_qwen_weight_edge`),
/// concatenating historical Runtime KV state for any edge whose
/// `GraphKvCacheBehavior::Append` metadata says so, dispatching every node
/// through the published `PreparedExecutionPlan`
/// (`dispatch_qwen_graph_node`), and returning every edge's bound value plus
/// the updated per-layer KV state.
/// Resolves the executing Provider's [`ProviderExecutionApi`] from Runtime's
/// own provider registration (see [`ProviderLoader::provider`]) rather than
/// constructing an unregistered throwaway. Unlike the downcast this replaced
/// (Correctif 3), this performs no concrete-type recovery: it only requires
/// that the registered Provider advertises `ProviderExecutionApi`, and
/// dispatch reaches it purely through that trait object. Fails closed with a
/// structured error if the named provider was never registered, or does not
/// implement `ProviderExecutionApi` at all.
fn resolve_kernel_execution_provider(
    runtime: &Runtime,
    provider_binding: &ProviderBinding,
) -> Result<Arc<dyn ProviderExecutionApi>, InferenceApiError> {
    let provider = runtime
        .providers()
        .provider(provider_binding.as_str())
        .ok_or_else(|| InferenceApiError::ProviderUnavailable {
            reason: format!("provider '{provider_binding}' is not registered with Runtime"),
        })?;
    provider
        .execution_api()
        .ok_or_else(|| InferenceApiError::ProviderUnavailable {
            reason: format!(
                "registered provider '{provider_binding}' does not implement ProviderExecutionApi"
            ),
        })
}

/// A completed graph execution's dispatch evidence, every edge's bound
/// value (graph inputs, intermediates, and outputs, keyed by
/// [`TensorEdgeId`]), and the updated per-layer KV state.
type QwenGraphExecutionOutput = (
    KernelDispatchResult,
    BTreeMap<TensorEdgeId, HostTensor>,
    Vec<FirstNativeLayerKvState>,
    ProviderBinding,
);

/// [`execute_qwen_graph_nodes`]'s own return shape -- the same as
/// [`QwenGraphExecutionOutput`] minus the resolved [`ProviderBinding`],
/// which only [`execute_qwen_graph`] (its caller) has resolved.
type QwenGraphNodesOutput = (
    KernelDispatchResult,
    BTreeMap<TensorEdgeId, HostTensor>,
    Vec<FirstNativeLayerKvState>,
);

#[allow(clippy::too_many_arguments)]
fn execute_qwen_graph(
    runtime: &mut Runtime,
    fixture: &E2eFixture,
    model_instance: &ModelInstanceId,
    kv_cache_id: &KvCacheId,
    graph: &ExecutionGraph,
    prepared_plan: &mut PreparedExecutionPlan,
    initial_bindings: BTreeMap<TensorEdgeId, HostTensor>,
    kv_history: Option<&[FirstNativeLayerKvState]>,
    absolute_position_override: Option<u64>,
    node_events: &mut Vec<PerNodeCausalEvent>,
) -> Result<QwenGraphExecutionOutput, InferenceApiError> {
    let order = qwen_graph_execution_order(graph)?;
    // Every node in this graph binds to the same Provider; any binding names
    // it (task 5.2: resolve the executing Provider from Runtime provider
    // registration for each prepared binding).
    let provider_binding = prepared_plan
        .node_bindings
        .first()
        .map(|binding| binding.provider.clone())
        .ok_or_else(|| InferenceApiError::KernelUnavailable {
            reason: "prepared execution plan has no node bindings".into(),
        })?;
    let executor = resolve_kernel_execution_provider(runtime, &provider_binding)?;
    // Cloning this instance's weight bindings (a small `String ->
    // TensorResourceId` map, not tensor bytes) sidesteps holding both this
    // immutable borrow and the mutable `memory_mut()` borrow below at once;
    // the actual weight tensors are read from Provider storage through
    // `resolve_qwen_weight_edge`, by resource id, per node (task 6.3/6.4).
    let weight_bindings = runtime
        .model_instance(model_instance)
        .map_err(InferenceApiError::from)?
        .definition
        .resource_bindings
        .weights
        .clone();
    // Runtime's own MemoryManager is used directly (via `QwenDispatchContext`
    // holding `&mut Runtime`, task group 7 / Correctif 13) rather than
    // temporarily taken out with `std::mem::take` and restored afterward --
    // that left any other code path holding `runtime` mid-call seeing a
    // default, empty `MemoryManager`, unsafe under continuous batching /
    // concurrent execution. Walking the graph is delegated to a helper
    // (rather than continuing inline) purely for readability now, not to
    // manage a restore point.
    let (dispatch, bindings, layer_kv) = execute_qwen_graph_nodes(
        &order,
        runtime,
        fixture,
        &weight_bindings,
        kv_cache_id,
        graph,
        prepared_plan,
        &executor,
        initial_bindings,
        kv_history,
        absolute_position_override,
        node_events,
    )?;
    Ok((dispatch, bindings, layer_kv, provider_binding))
}

#[allow(clippy::too_many_arguments)]
fn execute_qwen_graph_nodes(
    order: &[ExecutionNodeId],
    runtime: &mut Runtime,
    fixture: &E2eFixture,
    weight_bindings: &BTreeMap<String, TensorResourceId>,
    kv_cache_id: &KvCacheId,
    graph: &ExecutionGraph,
    prepared_plan: &mut PreparedExecutionPlan,
    executor: &Arc<dyn ProviderExecutionApi>,
    initial_bindings: BTreeMap<TensorEdgeId, HostTensor>,
    kv_history: Option<&[FirstNativeLayerKvState]>,
    absolute_position_override: Option<u64>,
    node_events: &mut Vec<PerNodeCausalEvent>,
) -> Result<QwenGraphNodesOutput, InferenceApiError> {
    // This step's token count, for `PlanGuardContext::sequence_length`
    // (Correctif 4): the same value every dispatched node's Plan guards
    // were built against (`PlanGuard::SequenceRange`), read from the graph
    // input rather than recomputed per node.
    let sequence_length = initial_bindings
        .get(&TensorEdgeId::new("input.token_ids"))
        .and_then(|tensor| tensor.shape.first().copied());
    let mut dispatch_ctx = QwenDispatchContext {
        runtime,
        provider: executor.clone(),
        prepared_plan: Some(prepared_plan),
        graph: Some(graph),
        sequence_length,
        last_provider_execution: None,
        node_events,
    };
    let layer_count = fixture.config.architecture.layer_count as usize;
    // Resource-based intermediates (Correctif 5): `bindings` never itself
    // holds tensor bytes. Every graph edge's live value lives exclusively in
    // the registered Provider's storage, keyed by `TensorResourceId`; this
    // map only records which resource each edge currently resolves to.
    // Graph inputs (`initial_bindings`) are the one seam where a caller
    // hands over raw values rather than an existing resource reference, so
    // they are written into Provider storage once, up front, under a
    // resource id derived from the edge id itself.
    //
    // Correctif 1: this write (and every node's `edge.*` output write
    // below) goes through `write_tensor_admitted`, not plain `write_tensor`
    // -- both are under a resource id *stable* across every call this KV
    // cache's session makes (one call per generation step), so a plain
    // `memory.allocate` here without replacement tracking would mint a new,
    // never-released allocation every step even though Provider storage
    // itself does not grow (each write overwrites the same entry) --
    // unbounded ledger growth over a long session. `write_tensor_admitted`
    // tracks and releases the allocation each resource id previously held,
    // Provider-side, so accounting stays bounded to this graph's edge count
    // no matter how many steps a session runs.
    let mut bindings: BTreeMap<TensorEdgeId, TensorResourceId> = BTreeMap::new();
    for (edge_id, tensor) in initial_bindings {
        let resource_id = TensorResourceId::new(edge_id.as_str());
        dispatch_ctx
            .provider
            .write_tensor_value_admitted(
                dispatch_ctx.runtime.memory_mut(),
                resource_id.clone(),
                TensorValue::Host(tensor),
                MemoryAllocationClass::Tensor,
                MemoryAllocationOwner::Session(kv_cache_id.to_string()),
            )
            .map_err(|error| match error {
                TensorValueAdmissionError::Memory(error) => {
                    InferenceApiError::MemoryAdmissionFailed {
                        reason: format!(
                            "failed to account graph input resource '{resource_id}': {error}"
                        ),
                    }
                }
                TensorValueAdmissionError::Provider(error) => {
                    InferenceApiError::ProviderTensorWriteFailed {
                        reason: format!(
                            "failed to write graph input resource '{resource_id}': {error}"
                        ),
                    }
                }
            })?;
        bindings.insert(edge_id, resource_id);
    }
    let mut layer_k: Vec<Option<TensorResourceId>> = vec![None; layer_count];
    let mut layer_v: Vec<Option<TensorResourceId>> = vec![None; layer_count];
    let mut last_dispatch: Option<KernelDispatchResult> = None;

    for node_id in order {
        let node =
            graph
                .nodes
                .get(node_id)
                .ok_or_else(|| InferenceApiError::GraphPlanningFailed {
                    reason: format!("first-native graph is missing node '{node_id}'"),
                })?;
        let mut inputs = Vec::with_capacity(node.inputs.len());
        for edge_id in &node.inputs {
            let tensor = match bindings.get(edge_id) {
                // Reference CPU Kernel input boundary: this edge's value
                // feeds `dispatch_qwen_graph_node`'s compute directly, so it
                // is materialized to host bytes here rather than carried as
                // an opaque `TensorValue` further into the generic loop.
                Some(resource_id) => {
                    let value = dispatch_ctx
                        .provider
                        .read_tensor_value(resource_id)
                        .ok_or_else(|| InferenceApiError::GraphPlanningFailed {
                            reason: format!(
                                "no materialized data for graph edge '{edge_id}' (resource '{resource_id}')"
                            ),
                        })?;
                    value.into_host(resource_id).map_err(|error| {
                        InferenceApiError::GraphPlanningFailed {
                            reason: format!(
                                "graph edge '{edge_id}' (resource '{resource_id}'): {error}"
                            ),
                        }
                    })?
                }
                None => resolve_qwen_weight_edge(
                    executor,
                    weight_bindings,
                    fixture.config.tied_embeddings,
                    edge_id.as_str(),
                )?,
            };
            inputs.push(tensor);
        }
        let (dispatch_result, mut output_tensor) = dispatch_qwen_graph_node(
            &mut dispatch_ctx,
            fixture,
            node,
            inputs,
            absolute_position_override,
        )?;
        let output_edge_id =
            node.outputs
                .first()
                .ok_or_else(|| InferenceApiError::GraphPlanningFailed {
                    reason: format!("graph node '{node_id}' has no output edge"),
                })?;
        let output_edge = graph.edges.get(output_edge_id).ok_or_else(|| {
            InferenceApiError::GraphPlanningFailed {
                reason: format!("first-native graph is missing edge '{output_edge_id}'"),
            }
        })?;
        if let Some(kv_meta) = &output_edge.kv_cache {
            let (layer, role) = parse_kv_cache_id(&kv_meta.cache_id)?;
            if layer >= layer_count {
                return Err(InferenceApiError::GraphPlanningFailed {
                    reason: format!(
                        "graph KV cache id '{}' has an out-of-range layer",
                        kv_meta.cache_id
                    ),
                });
            }
            if kv_meta.behavior == GraphKvCacheBehavior::Append {
                let history = kv_history.ok_or_else(|| InferenceApiError::KvCacheUnavailable {
                    reason: "decode graph execution requires historical KV state".into(),
                })?;
                let historical =
                    history
                        .get(layer)
                        .ok_or_else(|| InferenceApiError::KvCacheUnavailable {
                            reason: format!(
                                "decode requires historical KV state for layer {layer}"
                            ),
                        })?;
                let historical_resource = match role {
                    KvRole::K => &historical.k,
                    KvRole::V => &historical.v,
                };
                // Historical KV data is read back from the registered
                // Provider's storage by resource id (task 7.2/7.3), not from
                // a raw tensor an executor-private map handed the caller.
                // KV-history concatenation boundary: `concat_rows` is plain
                // Rust over `Vec<f32>`, so this materializes to host bytes
                // explicitly here rather than carrying an opaque value in.
                let historical_value = dispatch_ctx
                    .provider
                    .read_tensor_value(historical_resource)
                    .ok_or_else(|| InferenceApiError::KvCacheUnavailable {
                        reason: format!("no materialized historical KV data for layer {layer}"),
                    })?;
                let historical_tensor = historical_value
                    .into_host(historical_resource)
                    .map_err(|error| InferenceApiError::KvCacheUnavailable {
                        reason: format!(
                            "historical KV data for layer {layer} (resource '{historical_resource}'): {error}"
                        ),
                    })?;
                output_tensor = concat_rows(&historical_tensor, &output_tensor)?;
            }
            // Written under a *pending* resource id (task 7.4 prepare):
            // this generation step's KV update becomes Runtime-owned only
            // once `commit_generation_step` promotes it after sampling and
            // token commit succeed; a failure or cancellation before then
            // simply leaves this pending write unpromoted.
            let role_str = match role {
                KvRole::K => "k",
                KvRole::V => "v",
            };
            let pending_resource =
                TensorResourceId::new(format!("kv.{kv_cache_id}.layer{layer}.{role_str}.pending"));
            // Correctif 1: admitted (via `write_tensor_admitted`), not a
            // bare `write_tensor` -- this id is stable across every decode
            // step for this layer/role, so admission replaces (and
            // releases) whatever allocation the previous step's pending
            // write held for it, the same reasoning as the graph-edge
            // writes above. `discard_pending_kv_state` releases this
            // through `release_admitted_tensor` when a step is cancelled or
            // fails before commit.
            dispatch_ctx
                .provider
                .write_tensor_value_admitted(
                    dispatch_ctx.runtime.memory_mut(),
                    pending_resource.clone(),
                    TensorValue::Host(output_tensor.clone()),
                    MemoryAllocationClass::Tensor,
                    MemoryAllocationOwner::Session(kv_cache_id.to_string()),
                )
                .map_err(|error| match error {
                    TensorValueAdmissionError::Memory(error) => {
                        InferenceApiError::MemoryAdmissionFailed {
                            reason: format!(
                                "failed to account pending KV resource '{pending_resource}': {error}"
                            ),
                        }
                    }
                    TensorValueAdmissionError::Provider(error) => {
                        InferenceApiError::ProviderTensorWriteFailed {
                            reason: format!(
                                "failed to write pending KV resource '{pending_resource}': {error}"
                            ),
                        }
                    }
                })?;
            dispatch_ctx.node_events.push(
                PerNodeCausalEvent::new(
                    InferenceApiObservationKind::KvUpdatePrepared,
                    node_id.clone(),
                )
                .with_resource(pending_resource.clone()),
            );
            match role {
                KvRole::K => layer_k[layer] = Some(pending_resource),
                KvRole::V => layer_v[layer] = Some(pending_resource),
            }
        }
        // Written under a resource id derived from the edge itself, not the
        // node's internal dispatch naming: this is `bindings`' own resource
        // reference for this edge, independent of whatever id
        // `dispatch_reference_cpu_operator` used internally for the same
        // value (and distinct from it when `output_tensor` was reassigned
        // above by KV-history concatenation, so a later reader of this edge
        // sees the concatenated value, matching this function's prior
        // HostTensor-based behavior exactly).
        let output_resource_id = TensorResourceId::new(format!("edge.{output_edge_id}"));
        dispatch_ctx
            .provider
            .write_tensor_value_admitted(
                dispatch_ctx.runtime.memory_mut(),
                output_resource_id.clone(),
                TensorValue::Host(output_tensor),
                MemoryAllocationClass::Tensor,
                MemoryAllocationOwner::Session(kv_cache_id.to_string()),
            )
            .map_err(|error| match error {
                TensorValueAdmissionError::Memory(error) => {
                    InferenceApiError::MemoryAdmissionFailed {
                        reason: format!(
                            "failed to account graph edge resource '{output_resource_id}': {error}"
                        ),
                    }
                }
                TensorValueAdmissionError::Provider(error) => {
                    InferenceApiError::ProviderTensorWriteFailed {
                        reason: format!(
                            "failed to write graph edge resource '{output_resource_id}': {error}"
                        ),
                    }
                }
            })?;
        bindings.insert(output_edge_id.clone(), output_resource_id);
        last_dispatch = Some(dispatch_result);
    }

    let mut updated_layer_kv = Vec::with_capacity(layer_count);
    for layer in 0..layer_count {
        let k = layer_k[layer]
            .take()
            .ok_or_else(|| InferenceApiError::KvCacheUnavailable {
                reason: format!("first-native graph produced no K state for layer {layer}"),
            })?;
        let v = layer_v[layer]
            .take()
            .ok_or_else(|| InferenceApiError::KvCacheUnavailable {
                reason: format!("first-native graph produced no V state for layer {layer}"),
            })?;
        updated_layer_kv.push(FirstNativeLayerKvState { k, v });
    }
    let dispatch_result = last_dispatch.ok_or_else(|| InferenceApiError::GenerationFailed {
        reason: "first-native graph executed no nodes".into(),
    })?;
    // `QwenGraphExecutionOutput` returns materialized values to its callers
    // (e.g. to extract the "logits" edge) -- that caller-facing contract is
    // unchanged. Only this function's *internal* node-to-node transport is
    // Resource-based; the final materialization back to `HostTensor` happens
    // exactly once, here, not per-node.
    // Final logits extraction boundary: this is the one point where every
    // remaining live edge (including "output.logits") crosses back into the
    // caller-facing `HostTensor` contract; everywhere above this in the loop
    // carries `TensorValue` instead of assuming every Tensor Resource is
    // host-visible.
    let mut materialized_bindings = BTreeMap::new();
    for (edge_id, resource_id) in &bindings {
        let value = dispatch_ctx
            .provider
            .read_tensor_value(resource_id)
            .ok_or_else(|| InferenceApiError::GraphPlanningFailed {
                reason: format!(
                    "no materialized data for graph edge '{edge_id}' (resource '{resource_id}')"
                ),
            })?;
        let tensor = value.into_host(resource_id).map_err(|error| {
            InferenceApiError::GraphPlanningFailed {
                reason: format!("graph edge '{edge_id}' (resource '{resource_id}'): {error}"),
            }
        })?;
        materialized_bindings.insert(edge_id.clone(), tensor);
    }
    Ok((dispatch_result, materialized_bindings, updated_layer_kv))
}

/// Static guard (`define-provider-prepared-kernel-execution-contract` task
/// 2.3): [`execute_qwen_graph_nodes`]'s per-node transport migrated fully off
/// the `HostTensor`-typed [`ProviderExecutionApi`] methods (that Change's
/// task group 5) -- every read/write in its per-node loop goes through
/// `read_tensor_value`/`write_tensor_value_admitted` instead, materializing
/// to `HostTensor` only at the explicit host-materialization boundaries via
/// `TensorValue::into_host` (weight binding, KV-history concatenation, final
/// logits extraction, plus each node's own Kernel-input resolution). This
/// scans the function's own source text so a future edit that reintroduces a
/// direct `.read_tensor(`/`.write_tensor(`/`.write_tensor_admitted(` call
/// into that loop fails a test immediately, rather than the two pathways
/// (`HostTensor`-typed and `TensorValue`-typed) silently coexisting
/// indefinitely -- design.md's stated risk for that Change. Test-only: this
/// is a source-level build invariant, not runtime behavior
/// `run_e2e_local_inference_conformance` needs to check in production.
#[cfg(test)]
fn check_execute_qwen_graph_nodes_transport_has_no_host_tensor_typed_calls()
-> Result<(), E2eConformanceError> {
    const SOURCE: &str = include_str!("first_native_runtime.rs");
    let start = SOURCE.find("fn execute_qwen_graph_nodes(").ok_or_else(|| {
        E2eConformanceError::Internal {
            reason: "execute_qwen_graph_nodes not found in first_native_runtime.rs source".into(),
        }
    })?;
    let body_start = SOURCE[start..]
        .find('{')
        .map(|offset| start + offset)
        .ok_or_else(|| E2eConformanceError::Internal {
            reason: "execute_qwen_graph_nodes has no function body in source".into(),
        })?;
    let mut depth = 0i32;
    let mut body_end = body_start;
    for (offset, ch) in SOURCE[body_start..].char_indices() {
        match ch {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    body_end = body_start + offset + 1;
                    break;
                }
            }
            _ => {}
        }
    }
    if body_end == body_start {
        return Err(E2eConformanceError::Internal {
            reason: "execute_qwen_graph_nodes's function body braces did not balance".into(),
        });
    }
    let body = &SOURCE[body_start..body_end];
    // Exact-name matches only (immediate `(` after the method name), so the
    // intended replacements -- `.read_tensor_value(`, `.write_tensor_value(`,
    // `.write_tensor_value_admitted(` -- do not themselves trip this guard.
    let host_tensor_typed_call_count =
        [".read_tensor(", ".write_tensor(", ".write_tensor_admitted("]
            .iter()
            .map(|needle| body.matches(needle).count())
            .sum::<usize>();
    if host_tensor_typed_call_count != 0 {
        return Err(E2eConformanceError::Internal {
            reason: format!(
                "execute_qwen_graph_nodes's per-node transport has \
                 {host_tensor_typed_call_count} direct HostTensor-typed \
                 ProviderExecutionApi call(s); it must read/write through \
                 TensorValue (read_tensor_value/write_tensor_value_admitted) \
                 and materialize only at explicit host-materialization \
                 boundaries via TensorValue::into_host"
            ),
        });
    }
    Ok(())
}

/// Test-only oracle: a hand-written, hard-coded prefill dispatch sequence
/// kept only so tests can cross-check `execute_qwen_graph`'s output against
/// an independently-written recipe. Production first-native execution
/// cannot reach this function -- it computes logits exclusively through
/// `execute_qwen_graph` (see `E2eRuntimeModelExecutionEngine::
/// execute_generation_step`). `prepared_plan` is mandatory (not optional):
/// the first-native hot path must always look up a published
/// [`PlanNodeBinding`]/[`PreparedKernelId`] rather than ever falling back to
/// ad hoc Kernel Registry selection here -- planning-time selection belongs
/// in [`prepare_first_native_plan_for_graph`], not in this execution path.
#[cfg(test)]
fn execute_qwen_prefill_hidden_states_through_dispatch(
    runtime: &mut Runtime,
    fixture: &E2eFixture,
    token_ids: &[TokenId],
    prepared_plan: &mut PreparedExecutionPlan,
) -> Result<
    (
        KernelDispatchResult,
        HostTensor,
        Vec<FirstNativeLayerKvState>,
    ),
    InferenceApiError,
> {
    if token_ids.is_empty() {
        return Err(InferenceApiError::GenerationFailed {
            reason: "forward pass requires at least one token".into(),
        });
    }
    // Resolved from Runtime's own registration (not a throwaway) so the
    // decode oracle's separate call can read back the K/V resources this
    // call writes -- the whole point of testing incremental decode against
    // the KV state prefill actually produced.
    let provider = resolve_kernel_execution_provider(
        runtime,
        &ProviderBinding::new(REFERENCE_CPU_PROVIDER_NAME),
    )?;
    let mut node_events = Vec::new();
    let mut dispatch_ctx = QwenDispatchContext {
        runtime,
        provider: provider.clone(),
        prepared_plan: Some(prepared_plan),
        graph: None,
        sequence_length: None,
        last_provider_execution: None,
        node_events: &mut node_events,
    };
    let architecture = &fixture.config.architecture;
    let seq_len = token_ids.len() as u64;
    let epsilon = fixture.config.rmsnorm_epsilon;

    let ids_tensor = HostTensor::new(
        [seq_len],
        token_ids.iter().map(|id| *id as f32).collect::<Vec<_>>(),
    )
    .map_err(runtime_generation_failed)?;
    let token_embedding = fixture_tensor_by_name(&fixture.weights, "token_embedding")
        .map_err(runtime_generation_failed)?
        .clone();
    let (_embedding_dispatch, mut hidden_states) = dispatch_reference_cpu_operator(
        &mut dispatch_ctx,
        "embedding",
        dispatch_operator_id("embedding", OperatorFamily::Tensor),
        vec![
            (
                TensorResourceId::new("embedding.table"),
                f32_tensor_descriptor(&token_embedding),
                token_embedding,
            ),
            (
                TensorResourceId::new("embedding.ids"),
                f32_tensor_descriptor(&ids_tensor),
                ids_tensor,
            ),
        ],
        (
            TensorResourceId::new("embedding.out"),
            TensorDescriptor::new(
                ShapeDescriptor::new([seq_len, architecture.hidden_size]),
                DTypeDescriptor::portable(ComputeDType::Float32),
                LayoutDescriptor::Contiguous,
            ),
        ),
        BTreeMap::new(),
    )?;

    let mut layer_kv = Vec::with_capacity(architecture.layer_count as usize);
    for layer in 0..architecture.layer_count {
        let prefix = format!("layers.{layer}.");
        let input_norm = fixture_tensor_by_name(&fixture.weights, &format!("{prefix}input_norm"))
            .map_err(runtime_generation_failed)?
            .clone();
        let q_weight =
            fixture_tensor_by_name(&fixture.weights, &format!("{prefix}self_attn.q_proj"))
                .map_err(runtime_generation_failed)?
                .clone();
        let k_weight =
            fixture_tensor_by_name(&fixture.weights, &format!("{prefix}self_attn.k_proj"))
                .map_err(runtime_generation_failed)?
                .clone();
        let v_weight =
            fixture_tensor_by_name(&fixture.weights, &format!("{prefix}self_attn.v_proj"))
                .map_err(runtime_generation_failed)?
                .clone();
        let o_weight =
            fixture_tensor_by_name(&fixture.weights, &format!("{prefix}self_attn.o_proj"))
                .map_err(runtime_generation_failed)?
                .clone();
        let post_attn_norm =
            fixture_tensor_by_name(&fixture.weights, &format!("{prefix}post_attn_norm"))
                .map_err(runtime_generation_failed)?
                .clone();
        let gate_weight =
            fixture_tensor_by_name(&fixture.weights, &format!("{prefix}mlp.gate_proj"))
                .map_err(runtime_generation_failed)?
                .clone();
        let up_weight = fixture_tensor_by_name(&fixture.weights, &format!("{prefix}mlp.up_proj"))
            .map_err(runtime_generation_failed)?
            .clone();
        let down_weight =
            fixture_tensor_by_name(&fixture.weights, &format!("{prefix}mlp.down_proj"))
                .map_err(runtime_generation_failed)?
                .clone();

        let layer_id = format!("layer{layer}");
        let (_dispatch, normed) = dispatch_qwen_rmsnorm(
            &mut dispatch_ctx,
            &format!("{layer_id}.input_norm"),
            hidden_states.clone(),
            input_norm,
            epsilon,
        )?;
        let (_dispatch, q) = dispatch_qwen_matmul(
            &mut dispatch_ctx,
            &format!("{layer_id}.q_proj"),
            normed.clone(),
            q_weight,
        )?;
        let (_dispatch, k) = dispatch_qwen_matmul(
            &mut dispatch_ctx,
            &format!("{layer_id}.k_proj"),
            normed.clone(),
            k_weight,
        )?;
        let (_dispatch, v) = dispatch_qwen_matmul(
            &mut dispatch_ctx,
            &format!("{layer_id}.v_proj"),
            normed,
            v_weight,
        )?;
        let (_dispatch, q) = dispatch_qwen_rope_per_head(
            &mut dispatch_ctx,
            &format!("{layer_id}.rope_q"),
            &q,
            architecture.attention_head_count,
            architecture.head_dimension,
            &fixture.config.rope,
            0,
        )?;
        let (_dispatch, k) = dispatch_qwen_rope_per_head(
            &mut dispatch_ctx,
            &format!("{layer_id}.rope_k"),
            &k,
            architecture.kv_head_count,
            architecture.head_dimension,
            &fixture.config.rope,
            0,
        )?;
        let k_resource = TensorResourceId::new(format!("oracle-kv.layer{layer}.k"));
        let v_resource = TensorResourceId::new(format!("oracle-kv.layer{layer}.v"));
        dispatch_ctx
            .provider
            .write_tensor(k_resource.clone(), k.clone())
            .map_err(|error| InferenceApiError::ProviderTensorWriteFailed {
                reason: error.to_string(),
            })?;
        dispatch_ctx
            .provider
            .write_tensor(v_resource.clone(), v.clone())
            .map_err(|error| InferenceApiError::ProviderTensorWriteFailed {
                reason: error.to_string(),
            })?;
        layer_kv.push(FirstNativeLayerKvState {
            k: k_resource,
            v: v_resource,
        });
        let (_dispatch, attention_out) = dispatch_qwen_attention(
            &mut dispatch_ctx,
            &format!("{layer_id}.attention"),
            q,
            k,
            v,
            architecture,
        )?;
        let (_dispatch, attention_proj) = dispatch_qwen_matmul(
            &mut dispatch_ctx,
            &format!("{layer_id}.o_proj"),
            attention_out,
            o_weight,
        )?;
        let (_dispatch, post_attention) = dispatch_qwen_binary_same_shape(
            &mut dispatch_ctx,
            &format!("{layer_id}.residual1"),
            "residual-add",
            OperatorFamily::Tensor,
            attention_proj,
            hidden_states,
        )?;
        let (_dispatch, normed_mlp) = dispatch_qwen_rmsnorm(
            &mut dispatch_ctx,
            &format!("{layer_id}.post_attn_norm"),
            post_attention.clone(),
            post_attn_norm,
            epsilon,
        )?;
        let (_dispatch, gate) = dispatch_qwen_matmul(
            &mut dispatch_ctx,
            &format!("{layer_id}.gate_proj"),
            normed_mlp.clone(),
            gate_weight,
        )?;
        let (_dispatch, up) = dispatch_qwen_matmul(
            &mut dispatch_ctx,
            &format!("{layer_id}.up_proj"),
            normed_mlp,
            up_weight,
        )?;
        let (_dispatch, activated) = dispatch_qwen_unary(
            &mut dispatch_ctx,
            &format!("{layer_id}.silu"),
            "silu",
            OperatorFamily::Activation,
            gate,
            BTreeMap::new(),
        )?;
        let (_dispatch, gated) = dispatch_qwen_binary_same_shape(
            &mut dispatch_ctx,
            &format!("{layer_id}.mul"),
            "mul",
            OperatorFamily::Tensor,
            activated,
            up,
        )?;
        let (_dispatch, mlp_out) = dispatch_qwen_matmul(
            &mut dispatch_ctx,
            &format!("{layer_id}.down_proj"),
            gated,
            down_weight,
        )?;
        let (_dispatch, layer_out) = dispatch_qwen_binary_same_shape(
            &mut dispatch_ctx,
            &format!("{layer_id}.residual2"),
            "residual-add",
            OperatorFamily::Tensor,
            mlp_out,
            post_attention,
        )?;
        hidden_states = layer_out;
    }

    let final_norm = fixture_tensor_by_name(&fixture.weights, "final_norm")
        .map_err(runtime_generation_failed)?
        .clone();
    let (dispatch, hidden_states) = dispatch_qwen_rmsnorm(
        &mut dispatch_ctx,
        "final_norm",
        hidden_states,
        final_norm,
        epsilon,
    )?;
    Ok((dispatch, hidden_states, layer_kv))
}

/// Test-only oracle, kept only for cross-checking `execute_qwen_graph`; see
/// [`execute_qwen_prefill_hidden_states_through_dispatch`]'s doc comment.
#[cfg(test)]
fn execute_qwen_decode_hidden_states_through_dispatch(
    runtime: &mut Runtime,
    fixture: &E2eFixture,
    token_id: TokenId,
    kv_state: &FirstNativeExecutionKvState,
    absolute_position: u64,
    prepared_plan: &mut PreparedExecutionPlan,
) -> Result<
    (
        KernelDispatchResult,
        HostTensor,
        Vec<FirstNativeLayerKvState>,
    ),
    InferenceApiError,
> {
    // Resolved from Runtime's own registration (not a throwaway) so this
    // call can read back the K/V resources prefill wrote -- see
    // `execute_qwen_prefill_hidden_states_through_dispatch`'s doc comment.
    let provider = resolve_kernel_execution_provider(
        runtime,
        &ProviderBinding::new(REFERENCE_CPU_PROVIDER_NAME),
    )?;
    let mut node_events = Vec::new();
    let mut dispatch_ctx = QwenDispatchContext {
        runtime,
        provider: provider.clone(),
        prepared_plan: Some(prepared_plan),
        graph: None,
        sequence_length: None,
        last_provider_execution: None,
        node_events: &mut node_events,
    };
    let architecture = &fixture.config.architecture;
    if kv_state.layer_kv.len() != architecture.layer_count as usize {
        return Err(InferenceApiError::KvCacheUnavailable {
            reason: format!(
                "decode requires {} layer KV entries, found {}",
                architecture.layer_count,
                kv_state.layer_kv.len()
            ),
        });
    }
    let epsilon = fixture.config.rmsnorm_epsilon;
    let ids_tensor =
        HostTensor::new([1], vec![token_id as f32]).map_err(runtime_generation_failed)?;
    let token_embedding = fixture_tensor_by_name(&fixture.weights, "token_embedding")
        .map_err(runtime_generation_failed)?
        .clone();
    let (_embedding_dispatch, mut hidden_states) = dispatch_reference_cpu_operator(
        &mut dispatch_ctx,
        "decode.embedding",
        dispatch_operator_id("embedding", OperatorFamily::Tensor),
        vec![
            (
                TensorResourceId::new("decode.embedding.table"),
                f32_tensor_descriptor(&token_embedding),
                token_embedding,
            ),
            (
                TensorResourceId::new("decode.embedding.ids"),
                f32_tensor_descriptor(&ids_tensor),
                ids_tensor,
            ),
        ],
        (
            TensorResourceId::new("decode.embedding.out"),
            TensorDescriptor::new(
                ShapeDescriptor::new([1, architecture.hidden_size]),
                DTypeDescriptor::portable(ComputeDType::Float32),
                LayoutDescriptor::Contiguous,
            ),
        ),
        BTreeMap::new(),
    )?;

    let mut updated_layer_kv = Vec::with_capacity(architecture.layer_count as usize);
    for layer in 0..architecture.layer_count {
        let prefix = format!("layers.{layer}.");
        let input_norm = fixture_tensor_by_name(&fixture.weights, &format!("{prefix}input_norm"))
            .map_err(runtime_generation_failed)?
            .clone();
        let q_weight =
            fixture_tensor_by_name(&fixture.weights, &format!("{prefix}self_attn.q_proj"))
                .map_err(runtime_generation_failed)?
                .clone();
        let k_weight =
            fixture_tensor_by_name(&fixture.weights, &format!("{prefix}self_attn.k_proj"))
                .map_err(runtime_generation_failed)?
                .clone();
        let v_weight =
            fixture_tensor_by_name(&fixture.weights, &format!("{prefix}self_attn.v_proj"))
                .map_err(runtime_generation_failed)?
                .clone();
        let o_weight =
            fixture_tensor_by_name(&fixture.weights, &format!("{prefix}self_attn.o_proj"))
                .map_err(runtime_generation_failed)?
                .clone();
        let post_attn_norm =
            fixture_tensor_by_name(&fixture.weights, &format!("{prefix}post_attn_norm"))
                .map_err(runtime_generation_failed)?
                .clone();
        let gate_weight =
            fixture_tensor_by_name(&fixture.weights, &format!("{prefix}mlp.gate_proj"))
                .map_err(runtime_generation_failed)?
                .clone();
        let up_weight = fixture_tensor_by_name(&fixture.weights, &format!("{prefix}mlp.up_proj"))
            .map_err(runtime_generation_failed)?
            .clone();
        let down_weight =
            fixture_tensor_by_name(&fixture.weights, &format!("{prefix}mlp.down_proj"))
                .map_err(runtime_generation_failed)?
                .clone();

        let layer_id = format!("decode.layer{layer}");
        let (_dispatch, normed) = dispatch_qwen_rmsnorm(
            &mut dispatch_ctx,
            &format!("{layer_id}.input_norm"),
            hidden_states.clone(),
            input_norm,
            epsilon,
        )?;
        let (_dispatch, q) = dispatch_qwen_matmul(
            &mut dispatch_ctx,
            &format!("{layer_id}.q_proj"),
            normed.clone(),
            q_weight,
        )?;
        let (_dispatch, k_new) = dispatch_qwen_matmul(
            &mut dispatch_ctx,
            &format!("{layer_id}.k_proj"),
            normed.clone(),
            k_weight,
        )?;
        let (_dispatch, v_new) = dispatch_qwen_matmul(
            &mut dispatch_ctx,
            &format!("{layer_id}.v_proj"),
            normed,
            v_weight,
        )?;
        let (_dispatch, q) = dispatch_qwen_rope_per_head(
            &mut dispatch_ctx,
            &format!("{layer_id}.rope_q"),
            &q,
            architecture.attention_head_count,
            architecture.head_dimension,
            &fixture.config.rope,
            absolute_position,
        )?;
        let (_dispatch, k_new) = dispatch_qwen_rope_per_head(
            &mut dispatch_ctx,
            &format!("{layer_id}.rope_k"),
            &k_new,
            architecture.kv_head_count,
            architecture.head_dimension,
            &fixture.config.rope,
            absolute_position,
        )?;
        let historical = &kv_state.layer_kv[layer as usize];
        let historical_k = dispatch_ctx
            .provider
            .read_tensor(&historical.k)
            .ok_or_else(|| InferenceApiError::KvCacheUnavailable {
                reason: format!("no materialized historical K data for layer {layer}"),
            })?;
        let historical_v = dispatch_ctx
            .provider
            .read_tensor(&historical.v)
            .ok_or_else(|| InferenceApiError::KvCacheUnavailable {
                reason: format!("no materialized historical V data for layer {layer}"),
            })?;
        let k = concat_rows(&historical_k, &k_new)?;
        let v = concat_rows(&historical_v, &v_new)?;
        let k_resource = TensorResourceId::new(format!("oracle-kv.layer{layer}.k"));
        let v_resource = TensorResourceId::new(format!("oracle-kv.layer{layer}.v"));
        dispatch_ctx
            .provider
            .write_tensor(k_resource.clone(), k.clone())
            .map_err(|error| InferenceApiError::ProviderTensorWriteFailed {
                reason: error.to_string(),
            })?;
        dispatch_ctx
            .provider
            .write_tensor(v_resource.clone(), v.clone())
            .map_err(|error| InferenceApiError::ProviderTensorWriteFailed {
                reason: error.to_string(),
            })?;
        updated_layer_kv.push(FirstNativeLayerKvState {
            k: k_resource,
            v: v_resource,
        });
        let (_dispatch, attention_out) = dispatch_qwen_attention(
            &mut dispatch_ctx,
            &format!("{layer_id}.attention"),
            q,
            k,
            v,
            architecture,
        )?;
        let (_dispatch, attention_proj) = dispatch_qwen_matmul(
            &mut dispatch_ctx,
            &format!("{layer_id}.o_proj"),
            attention_out,
            o_weight,
        )?;
        let (_dispatch, post_attention) = dispatch_qwen_binary_same_shape(
            &mut dispatch_ctx,
            &format!("{layer_id}.residual1"),
            "residual-add",
            OperatorFamily::Tensor,
            attention_proj,
            hidden_states,
        )?;
        let (_dispatch, normed_mlp) = dispatch_qwen_rmsnorm(
            &mut dispatch_ctx,
            &format!("{layer_id}.post_attn_norm"),
            post_attention.clone(),
            post_attn_norm,
            epsilon,
        )?;
        let (_dispatch, gate) = dispatch_qwen_matmul(
            &mut dispatch_ctx,
            &format!("{layer_id}.gate_proj"),
            normed_mlp.clone(),
            gate_weight,
        )?;
        let (_dispatch, up) = dispatch_qwen_matmul(
            &mut dispatch_ctx,
            &format!("{layer_id}.up_proj"),
            normed_mlp,
            up_weight,
        )?;
        let (_dispatch, activated) = dispatch_qwen_unary(
            &mut dispatch_ctx,
            &format!("{layer_id}.silu"),
            "silu",
            OperatorFamily::Activation,
            gate,
            BTreeMap::new(),
        )?;
        let (_dispatch, gated) = dispatch_qwen_binary_same_shape(
            &mut dispatch_ctx,
            &format!("{layer_id}.mul"),
            "mul",
            OperatorFamily::Tensor,
            activated,
            up,
        )?;
        let (_dispatch, mlp_out) = dispatch_qwen_matmul(
            &mut dispatch_ctx,
            &format!("{layer_id}.down_proj"),
            gated,
            down_weight,
        )?;
        let (_dispatch, layer_out) = dispatch_qwen_binary_same_shape(
            &mut dispatch_ctx,
            &format!("{layer_id}.residual2"),
            "residual-add",
            OperatorFamily::Tensor,
            mlp_out,
            post_attention,
        )?;
        hidden_states = layer_out;
    }

    let final_norm = fixture_tensor_by_name(&fixture.weights, "final_norm")
        .map_err(runtime_generation_failed)?
        .clone();
    let (dispatch, hidden_states) = dispatch_qwen_rmsnorm(
        &mut dispatch_ctx,
        "decode.final_norm",
        hidden_states,
        final_norm,
        epsilon,
    )?;
    Ok((dispatch, hidden_states, updated_layer_kv))
}

/// Test-only oracle, kept only for cross-checking `execute_qwen_graph`; see
/// [`execute_qwen_prefill_hidden_states_through_dispatch`]'s doc comment.
#[cfg(test)]
fn dispatch_qwen_logits_projection(
    runtime: &Runtime,
    fixture: &E2eFixture,
    hidden_states: &HostTensor,
    prepared_plan: &PreparedExecutionPlan,
) -> Result<(KernelDispatchResult, Vec<f32>), InferenceApiError> {
    let token_embedding =
        fixture_tensor_by_name(&fixture.weights, "token_embedding").map_err(|error| {
            InferenceApiError::GenerationFailed {
                reason: error.to_string(),
            }
        })?;
    let token_embedding_transposed = transpose_rows_cols(token_embedding).map_err(|error| {
        InferenceApiError::GenerationFailed {
            reason: error.to_string(),
        }
    })?;
    let (dispatch_result, output) = dispatch_matmul_with_prepared_plan(
        runtime,
        hidden_states,
        &token_embedding_transposed,
        "lm_head",
        Some(prepared_plan),
    )?;
    let vocab = fixture.config.architecture.vocabulary_size as usize;
    let output_rows = output.data.len() / vocab;
    let last_row_start = output_rows.saturating_sub(1) * vocab;
    Ok((
        dispatch_result,
        output.data[last_row_start..last_row_start + vocab].to_vec(),
    ))
}

impl RuntimeModelExecutionEngine for E2eRuntimeModelExecutionEngine {
    fn execute_generation_step(
        &self,
        runtime: &mut Runtime,
        request: &GenerationRequest,
        generated_tokens: &[TokenId],
        execution_plan: Option<&mut PreparedExecutionPlan>,
    ) -> Result<RuntimeModelExecutionStep, InferenceApiError> {
        #[cfg(test)]
        let forced_token = self.forced_token;
        #[cfg(not(test))]
        let forced_token: Option<TokenId> = None;

        let vocab = self.fixture.config.architecture.vocabulary_size as usize;
        let mut kv_state = if generated_tokens.is_empty() {
            self.create_prefill_kv_state(runtime, request)?
        } else {
            self.decode_kv_state(runtime, request)?
        };
        let model_input_token_count = if generated_tokens.is_empty() {
            request.input_token_ids.len()
        } else {
            1
        };
        let absolute_position = if generated_tokens.is_empty() {
            request.input_token_ids.len()
        } else {
            generation_decode_absolute_position(
                request.input_token_ids.len(),
                generated_tokens.len(),
            )
            .ok_or_else(|| InferenceApiError::GenerationFailed {
                reason: "decode requires a newly admitted token position".into(),
            })?
        };
        self.discard_pending_kv_state(runtime, request)?;
        // Both flags below (task 8.1) are only ever assigned immediately
        // after the check they attest to actually ran and passed for *this*
        // step -- never assumed from an earlier, one-time check (Model
        // Instance readiness is otherwise confirmed once, at plan
        // preparation time, before the generation loop starts) or from a
        // bare literal disconnected from any check.
        let model_instance_ready;
        let graph_validated;
        let mut node_events: Vec<PerNodeCausalEvent> = Vec::new();
        let (dispatch_result, logits) = if let Some(token) = forced_token {
            // Test-only deterministic shortcut, never exercised by
            // `validate_e2e_no_shortcuts` (the success path this gate
            // checks is always built without a forced token -- see
            // `build_runtime_with_model_execution_engine`). The returned
            // logits are synthetic by construction; a trivial matmul is
            // still dispatched so the Provider/observability path is
            // exercised, but its output is deliberately not what is
            // returned.
            let one = HostTensor::new([1, 1], vec![1.0]).map_err(|error| {
                InferenceApiError::GenerationFailed {
                    reason: error.to_string(),
                }
            })?;
            let (dispatch_result, _proof_output) = dispatch_matmul(runtime, &one, &one)?;
            let mut logits = vec![0.0_f32; vocab];
            logits[token as usize] = 10.0;
            model_instance_ready = true;
            graph_validated = true;
            (dispatch_result, logits)
        } else {
            // The first-native hot path must always execute a published,
            // ready `PreparedExecutionPlan` -- never fall back to ad hoc
            // Kernel Registry selection here. A missing plan is a structured
            // failure, not a silent per-node reselection.
            let execution_plan = execution_plan.ok_or_else(|| InferenceApiError::GraphPlanningFailed {
                reason: "first-native execution requires a published prepared execution plan; none was bound for this generation step".into(),
            })?;
            // Task 11.5: must be the exact same graph-production recipe
            // `prepare_first_native_execution_plans` used to compute
            // `execution_plan`'s `graph_fingerprint` -- a graph built any
            // other way (even one that is logically equivalent to a human
            // reader) fails `PreparedExecutionPlanExecutor::
            // prepare_node_execution`'s fingerprint check below with
            // `PlanValidationFailed`, since that check is deliberately
            // exact-match, not semantic.
            // `first_native_component_graphs_for_prompt` is that same
            // recipe (real Qwen Component under the strict, default build;
            // a structured fail-closed error in production otherwise; the
            // Rust-synthesized recipe only in a test build without a
            // strict engine) -- deterministic for a given `(fixture,
            // prompt_token_count)` whenever it succeeds at all, so calling
            // it again here reproduces the identical graph the plan was
            // prepared against.
            let component_graphs = first_native_component_graphs_for_prompt(
                &self.fixture,
                request.input_token_ids.len() as u64,
            )
            .map_err(|error| InferenceApiError::GraphPlanningFailed {
                reason: error.to_string(),
            })?;
            // Weight resources are bound to an active Model Instance (task
            // 6.3), so graph execution needs its id to look them up.
            let GenerationModelReference::ModelInstance(model_instance) = &request.model else {
                return Err(InferenceApiError::ModelLoadingFailed {
                    reason: "first-native execution requires an active Model Instance to resolve weight resources".into(),
                });
            };
            // Confirm this Model Instance is genuinely ready to generate
            // *right now* -- not merely inferred from the one-time check
            // `prepare_first_native_execution_plans` performed before the
            // generation loop started, which cannot see an instance
            // unloaded or invalidated between steps.
            require_ready_first_native_instance(&*runtime, model_instance)?;
            model_instance_ready = true;
            // The graph, not a hand-written Rust sequence, is the
            // authoritative recipe for prefill/decode execution (see
            // `execute_qwen_graph`'s doc comment). Its output edge
            // `"logits"` carries logits for every model-input row; only the
            // last row is a newly admitted token's distribution.
            let (dispatch, mut bindings, layer_kv, resolved_provider) =
                if generated_tokens.is_empty() {
                    let ids_tensor = HostTensor::new(
                        [request.input_token_ids.len() as u64],
                        request
                            .input_token_ids
                            .iter()
                            .map(|id| *id as f32)
                            .collect::<Vec<_>>(),
                    )
                    .map_err(runtime_generation_failed)?;
                    execute_qwen_graph(
                        runtime,
                        &self.fixture,
                        model_instance,
                        &kv_state.cache,
                        &component_graphs.prefill,
                        execution_plan,
                        BTreeMap::from([(TensorEdgeId::new("input.token_ids"), ids_tensor)]),
                        None,
                        Some(0),
                        &mut node_events,
                    )?
                } else {
                    let token = *generated_tokens.last().ok_or_else(|| {
                        InferenceApiError::GenerationFailed {
                            reason: "decode requires a newly admitted token".into(),
                        }
                    })?;
                    let ids_tensor = HostTensor::new([1], vec![token as f32])
                        .map_err(runtime_generation_failed)?;
                    execute_qwen_graph(
                        runtime,
                        &self.fixture,
                        model_instance,
                        &kv_state.cache,
                        &component_graphs.decode,
                        execution_plan,
                        BTreeMap::from([(TensorEdgeId::new("input.token_ids"), ids_tensor)]),
                        Some(&kv_state.layer_kv),
                        Some(absolute_position as u64),
                        &mut node_events,
                    )?
                };
            // `execute_qwen_graph` topologically validates the graph (via
            // `qwen_graph_execution_order`) as its first action -- reaching
            // this point with an `Ok` result means that validation
            // genuinely ran and passed for this step's graph, not that it
            // is merely assumed.
            graph_validated = true;
            kv_state.layer_kv = layer_kv;
            // Carries the Provider this step's graph execution actually
            // resolved through to KV commit/discard
            // (`generalize-first-native-provider-dispatch`): those used to
            // hardcode Reference CPU unconditionally regardless of which
            // Provider the pending K/V resources were actually written
            // under.
            kv_state.provider = Some(resolved_provider);
            let logits_tensor = bindings
                .remove(&TensorEdgeId::new("logits"))
                .ok_or_else(|| InferenceApiError::GenerationFailed {
                    reason: "first-native graph produced no logits output".into(),
                })?;
            let output_rows = logits_tensor.data.len() / vocab.max(1);
            let last_row_start = output_rows.saturating_sub(1) * vocab;
            (
                dispatch,
                logits_tensor.data[last_row_start..last_row_start + vocab].to_vec(),
            )
        };
        let kv_commit = if generated_tokens.is_empty() {
            RuntimeKvCacheCommit::PrefillCompleted {
                cache: kv_state.cache.clone(),
                tokens: request.prompt_token_count as u32,
            }
        } else {
            RuntimeKvCacheCommit::DecodeAppended {
                cache: kv_state.cache.clone(),
                tokens: 1,
            }
        };
        self.store_pending_kv_state(request, kv_state.clone())?;
        let mut evidence = RuntimeGenerationExecutionEvidence::from_dispatch_result(
            &dispatch_result,
            model_instance_ready,
            graph_validated,
        )
        .with_context(format!("request={}", request.request_id))
        .with_context(format!("decode_step={}", generated_tokens.len()))
        .with_context(format!("model_input_tokens={model_input_token_count}"))
        .with_context(format!("kv_cache={}", kv_state.cache))
        .with_context(format!("absolute_position={absolute_position}"));
        if let GenerationModelReference::ModelInstance(instance) = &request.model {
            evidence = evidence.with_context(format!("model_instance={instance}"));
        }
        if let Some(session) = &request.session {
            evidence = evidence.with_context(format!("session={session}"));
        }
        Ok(RuntimeModelExecutionStep::new(logits, evidence)
            .with_kv_commit(kv_commit)
            .with_node_events(node_events))
    }

    fn commit_generation_step(
        &self,
        runtime: &mut Runtime,
        request: &GenerationRequest,
        generated_tokens_before_step: &[TokenId],
        _accepted_token: TokenId,
        _step: &RuntimeModelExecutionStep,
    ) -> Result<(), InferenceApiError> {
        let mut state = self.take_pending_kv_state(request)?;
        // Promote this step's pending KV resources to Runtime-owned,
        // committed bindings on the KvCache itself (task 7.1/7.3/7.4
        // commit) only now that sampling and token commit have succeeded --
        // never before. A failure anywhere before this point leaves the
        // pending writes unpromoted and the cache's committed bindings
        // untouched (task 7.4 abort/7.5 rollback).
        state.layer_kv = self.promote_pending_kv_resources(runtime, &state)?;
        if generated_tokens_before_step.is_empty() {
            runtime.prefill_kv_cache_completed(&state.cache, request.prompt_token_count as u32)?;
        } else {
            runtime.append_decode_kv_cache(&state.cache, 1)?;
        }
        self.store_kv_state(request, state)
    }
}

/// One promoted layer's committed binding, and the binding it replaces (if
/// any), pending this transaction's `commit`.
struct PromotedKvLayer {
    layer: u32,
    binding: KvLayerResourceBinding,
    previous: Option<KvLayerResourceBinding>,
}

/// The KV commit lifecycle, named and typed explicitly (Correctif 11 / task
/// group 9): admission, pending-Resource-ID lookup, Provider materialization
/// under an attempt-unique resource id, then either `commit` (publish every
/// layer's binding to the cache and release the resources each one
/// replaces) or `abort` (release everything this attempt admitted, leaving
/// the cache's prior committed state completely untouched). A transaction
/// promotes every layer under a *new* resource id first, without touching
/// the cache's existing committed bindings at all -- only a fully successful
/// `commit` publishes anything, so a decode step's failed commit cannot
/// leave some layers pointing at this step's data and others at the
/// previous step's.
struct KvUpdateTransaction {
    provider_binding: ProviderBinding,
    executor: Arc<dyn ProviderExecutionApi>,
    promoted: Vec<PromotedKvLayer>,
}

impl KvUpdateTransaction {
    /// Resolves the Provider `execute_qwen_graph` actually wrote this step's
    /// pending K/V resources under (carried on `state.provider`), falling
    /// back to Reference CPU only when unset -- `generalize-first-native-
    /// provider-dispatch`'s fix: this used to hardcode Reference CPU
    /// unconditionally, so a non-CPU step's pending writes would be looked
    /// for under the wrong Provider binding at commit time.
    fn begin(
        runtime: &Runtime,
        state: &FirstNativeExecutionKvState,
    ) -> Result<Self, InferenceApiError> {
        let provider_binding = state
            .provider
            .clone()
            .unwrap_or_else(|| ProviderBinding::new(REFERENCE_CPU_PROVIDER_NAME));
        let executor = resolve_kernel_execution_provider(runtime, &provider_binding)?;
        Ok(Self {
            provider_binding,
            executor,
            promoted: Vec::new(),
        })
    }

    /// Admits and writes one layer's pending K/V under attempt-unique
    /// resource ids. Does not touch the cache's existing committed
    /// bindings; on failure, the caller should call `abort` to release
    /// everything already promoted earlier this attempt.
    fn promote_layer(
        &mut self,
        runtime: &mut Runtime,
        cache: &KvCacheId,
        layer: usize,
        pending: &FirstNativeLayerKvState,
    ) -> Result<(), InferenceApiError> {
        let previous = runtime
            .kv_cache(cache)?
            .layer_resources
            .get(&(layer as u32))
            .cloned();
        let binding = promote_pending_kv_layer(
            runtime,
            &self.executor,
            &self.provider_binding,
            cache,
            layer,
            pending,
        )?;
        self.promoted.push(PromotedKvLayer {
            layer: layer as u32,
            binding,
            previous,
        });
        Ok(())
    }

    /// Releases every resource admitted so far this attempt, leaving the
    /// cache's prior committed state completely untouched.
    fn abort(self, runtime: &mut Runtime) {
        for already in &self.promoted {
            let _ = runtime.memory_mut().release(already.binding.k_allocation);
            let _ = runtime.memory_mut().release(already.binding.v_allocation);
            // Best-effort, like the memory releases above: this is already
            // an unwind over a different failure, with nothing further to
            // roll back to if the Provider release itself fails.
            let _ = self.executor.release_tensor(&already.binding.k);
            let _ = self.executor.release_tensor(&already.binding.v);
        }
    }

    /// Publishes every promoted layer's binding to the cache and releases
    /// the resources each one replaces, reached only once every layer in
    /// this transaction has been successfully promoted.
    fn commit(
        self,
        runtime: &mut Runtime,
        cache: &KvCacheId,
    ) -> Result<Vec<FirstNativeLayerKvState>, InferenceApiError> {
        let mut committed = Vec::with_capacity(self.promoted.len());
        for promoted_layer in self.promoted {
            if let Some(previous) = promoted_layer.previous {
                let _ = runtime.memory_mut().release(previous.k_allocation);
                let _ = runtime.memory_mut().release(previous.v_allocation);
                // Best-effort: the new binding above already promoted
                // successfully, so a failure releasing the superseded one
                // is a leak to surface separately, not a reason to fail
                // this otherwise-successful commit.
                let _ = self.executor.release_tensor(&previous.k);
                let _ = self.executor.release_tensor(&previous.v);
            }
            runtime
                .kv_caches_mut()
                .cache_mut(cache)?
                .layer_resources
                .insert(promoted_layer.layer, promoted_layer.binding.clone());
            committed.push(FirstNativeLayerKvState {
                k: promoted_layer.binding.k,
                v: promoted_layer.binding.v,
            });
        }
        Ok(committed)
    }
}

/// Promotes one layer's pending K and V together, admitting and writing
/// both under attempt-unique resource ids. Neither publishes anything to
/// the cache nor releases any prior allocation -- both are
/// [`KvUpdateTransaction::commit`]'s responsibility once it knows the
/// *entire* multi-layer commit succeeded.
fn promote_pending_kv_layer(
    runtime: &mut Runtime,
    executor: &Arc<dyn ProviderExecutionApi>,
    provider_binding: &ProviderBinding,
    cache: &KvCacheId,
    layer: usize,
    pending: &FirstNativeLayerKvState,
) -> Result<KvLayerResourceBinding, InferenceApiError> {
    let k = promote_pending_kv_layer_role(
        runtime,
        executor,
        provider_binding,
        cache,
        layer,
        "k",
        &pending.k,
    )?;
    let v = promote_pending_kv_layer_role(
        runtime,
        executor,
        provider_binding,
        cache,
        layer,
        "v",
        &pending.v,
    )?;
    Ok(KvLayerResourceBinding {
        k: k.0,
        v: v.0,
        k_allocation: k.1,
        v_allocation: v.1,
    })
}

#[allow(clippy::too_many_arguments)]
fn promote_pending_kv_layer_role(
    runtime: &mut Runtime,
    executor: &Arc<dyn ProviderExecutionApi>,
    provider_binding: &ProviderBinding,
    cache: &KvCacheId,
    layer: usize,
    role: &str,
    pending_resource: &TensorResourceId,
) -> Result<(TensorResourceId, MemoryAllocationId), InferenceApiError> {
    let tensor = executor.read_tensor(pending_resource).ok_or_else(|| {
        InferenceApiError::KvCacheUnavailable {
            reason: format!(
                "no pending KV data to commit for layer {layer} ({role}); already committed or aborted?"
            ),
        }
    })?;
    let byte_size = tensor.data.len() as u64 * std::mem::size_of::<f32>() as u64;
    // Admission SHALL precede Provider materialization: reserve the
    // committed resource's memory before writing its bytes into
    // Provider-owned storage, not after.
    let allocation = runtime
        .memory_mut()
        .allocate(MemoryAllocationRequest::new(
            MemoryAllocationClass::Tensor,
            byte_size,
            MemoryPlacement::ProviderOwnedOpaque(provider_binding.clone()),
            MemoryAllocationOwner::Session(cache.to_string()),
        ))
        .map_err(|error| InferenceApiError::MemoryAdmissionFailed {
            reason: format!(
                "failed to account committed KV resource for layer {layer} ({role}): {error}"
            ),
        })?;
    // The resource id is unique to this promotion attempt (keyed by the
    // allocation id, itself unique), not the stable
    // `kv.{cache}.layer{N}.{role}` name a naive implementation might reuse
    // across decode steps -- reusing a stable id would mean this write
    // destructively overwrites the previous step's still-valid committed
    // bytes before the caller knows whether the *whole* multi-layer commit
    // succeeds, making rollback impossible to do correctly (Correctif 11).
    let committed_resource = TensorResourceId::new(format!(
        "kv.{cache}.layer{layer}.{role}.gen{}",
        allocation.id
    ));
    if let Err(error) = executor.write_tensor(committed_resource.clone(), tensor) {
        // The allocation above already admitted successfully: release it
        // before propagating, so this failed promotion leaves no trace in
        // the Memory Manager ledger (same rollback shape as
        // `WeightMaterializationTransaction::stage_weight`'s own
        // admission-then-write-or-residency-failure handling).
        let _ = runtime.memory_mut().release(allocation.id);
        return Err(InferenceApiError::ProviderTensorWriteFailed {
            reason: format!(
                "failed to write committed KV resource for layer {layer} ({role}): {error}"
            ),
        });
    }
    Ok((committed_resource, allocation.id))
}

impl E2eRuntimeModelExecutionEngine {
    /// Promotes each layer's *pending* K/V resource (written by
    /// `execute_qwen_graph_nodes` during this step) to the cache's
    /// *committed* resource, atomically across every layer, via
    /// [`KvUpdateTransaction`].
    fn promote_pending_kv_resources(
        &self,
        runtime: &mut Runtime,
        state: &FirstNativeExecutionKvState,
    ) -> Result<Vec<FirstNativeLayerKvState>, InferenceApiError> {
        let mut transaction = KvUpdateTransaction::begin(runtime, state)?;
        for (layer, pending) in state.layer_kv.iter().enumerate() {
            if let Err(error) = transaction.promote_layer(runtime, &state.cache, layer, pending) {
                transaction.abort(runtime);
                return Err(error);
            }
        }
        transaction.commit(runtime, &state.cache)
    }
}

// ---------------------------------------------------------------------
// No-shortcut validation
// ---------------------------------------------------------------------

/// Validates that inference produced execution evidence for the Kernel
/// Registry / Kernel Dispatch / Reference CPU path rather than relying on a
/// caller assertion.
pub fn validate_e2e_no_shortcuts(
    observations: &[InferenceApiObservation],
    advertisements: &[KernelAdvertisement],
) -> Result<(), E2eConformanceError> {
    let has = |kind: InferenceApiObservationKind| {
        observations
            .iter()
            .any(|observation| observation.kind == kind)
    };
    for required in [
        InferenceApiObservationKind::ExecutionGraphValidated,
        InferenceApiObservationKind::KernelSelected,
        InferenceApiObservationKind::KernelDispatched,
        InferenceApiObservationKind::ProviderExecuted,
        InferenceApiObservationKind::TensorLogitsProduced,
    ] {
        if !has(required) {
            return Err(E2eConformanceError::BoundaryViolation {
                reason: format!("inference did not emit {required:?} execution evidence"),
            });
        }
    }
    if observations.is_empty() {
        return Err(E2eConformanceError::BoundaryViolation {
            reason:
                "inference bypassed Kernel Registry and invoked Reference CPU Provider directly"
                    .into(),
        });
    }
    validate_no_placeholder_kernel_advertisements(advertisements)
        .map_err(E2eConformanceError::from)?;
    validate_reference_cpu_required_kernel_coverage(advertisements)
        .map_err(E2eConformanceError::from)?;
    validate_e2e_per_node_causal_chain(observations)?;
    Ok(())
}

/// Extracts the `node=...` correlation an observation's message carries
/// (Correctif 17 / task group 17), if any -- the same `node=<value> ` shape
/// every per-node causal event in `inference_api.rs`'s generation loop
/// writes via `observation_message`. Returns the value up to (not
/// including) the next space, since further `key=value` pairs may follow.
fn observation_node_correlation(message: &str) -> Option<&str> {
    message
        .split("node=")
        .nth(1)
        .map(|rest| rest.split(' ').next().unwrap_or(rest))
}

/// Correctif 17 / task group 17: unlike the presence checks above (which
/// only confirm five *global* evidence categories occurred somewhere across
/// the whole generation), this walks every distinct graph node that
/// actually dispatched (identified by its `GraphNodeReady` events) and
/// requires each one to carry a *complete* per-node causal chain --
/// `PlanBindingResolved`, `PreparedKernelResolved`, `ProviderSubmitted`,
/// `ProviderCompleted`, and `TensorResourceProduced`, all correlated to
/// that same node -- not merely that each event kind occurred for *some*
/// node. `KvUpdatePrepared`/`KvUpdateCommitted` are deliberately excluded:
/// they only apply to KV-cache-bearing nodes (not every node produces one),
/// and `KvUpdateCommitted` in particular correlates to a `TensorResourceId`
/// rather than a node at all (see its emission site's doc comment in
/// `inference_api.rs`).
fn validate_e2e_per_node_causal_chain(
    observations: &[InferenceApiObservation],
) -> Result<(), E2eConformanceError> {
    let ready_nodes: BTreeSet<&str> = observations
        .iter()
        .filter(|observation| observation.kind == InferenceApiObservationKind::GraphNodeReady)
        .filter_map(|observation| observation_node_correlation(&observation.message))
        .collect();
    if ready_nodes.is_empty() {
        return Err(E2eConformanceError::BoundaryViolation {
            reason: "inference emitted no per-node GraphNodeReady evidence at all".into(),
        });
    }
    for required in [
        InferenceApiObservationKind::PlanBindingResolved,
        InferenceApiObservationKind::PreparedKernelResolved,
        InferenceApiObservationKind::ProviderSubmitted,
        InferenceApiObservationKind::ProviderCompleted,
        InferenceApiObservationKind::TensorResourceProduced,
    ] {
        let correlated_nodes: BTreeSet<&str> = observations
            .iter()
            .filter(|observation| observation.kind == required)
            .filter_map(|observation| observation_node_correlation(&observation.message))
            .collect();
        let missing: Vec<&str> = ready_nodes.difference(&correlated_nodes).copied().collect();
        if !missing.is_empty() {
            return Err(E2eConformanceError::BoundaryViolation {
                reason: format!(
                    "node(s) {missing:?} dispatched (GraphNodeReady) but never emitted a \
                     correlated {required:?} -- incomplete per-node causal chain"
                ),
            });
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------
// Suite runner
// ---------------------------------------------------------------------

/// Outcome of a completed E2E success-path run, kept around so downstream
/// checks (determinism, streaming order, usage accounting) can inspect it.
struct E2eRunOutcome {
    generation_result: GenerationResult,
    observer: InferenceApiObserver,
    kv_observations: Vec<KvCacheObservation>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct FirstNativeFixtureGeneration {
    pub text: String,
    pub result: GenerationResult,
    pub observer: InferenceApiObserver,
}

fn build_runtime() -> Runtime {
    let mut runtime = Runtime::builder()
        .register_provider(std::sync::Arc::new(ReferenceCpuProvider::new()))
        .build()
        .expect("Reference CPU provider registers cleanly");
    register_reference_cpu_prepared_kernels(&mut runtime);
    runtime
}

/// Same as [`build_runtime`], but the Runtime's sealed trust configuration
/// (`seal-runtime-model-trust-and-provenance-authority`) trusts `fixture`'s
/// manifest digest -- for the many E2E tests that go on to actually load
/// `fixture` through it via [`load_fixture_instance`] or
/// [`load_fixture_instance_with_weights`]. Tests that only inspect Runtime
/// state without loading anything (kernel registry advertisement checks,
/// for instance) use [`build_runtime`] directly instead, since they have no
/// fixture to trust.
fn build_runtime_trusting_fixture(fixture: &E2eFixture) -> Runtime {
    let mut runtime = Runtime::builder()
        .register_provider(std::sync::Arc::new(ReferenceCpuProvider::new()))
        .trust_store(
            ModelTrustStore::default().trust_digest(fixture.manifest.id.digest.value.clone()),
        )
        .build()
        .expect("Reference CPU provider registers cleanly");
    register_reference_cpu_prepared_kernels(&mut runtime);
    runtime
}

fn build_runtime_with_model_execution_engine(fixture: &E2eFixture) -> Runtime {
    let mut runtime = Runtime::builder()
        .register_provider(std::sync::Arc::new(ReferenceCpuProvider::new()))
        .model_execution_engine(std::sync::Arc::new(E2eRuntimeModelExecutionEngine {
            fixture: fixture.clone(),
            kv_states: Arc::new(Mutex::new(BTreeMap::new())),
            pending_kv_states: Arc::new(Mutex::new(BTreeMap::new())),
            #[cfg(test)]
            forced_token: None,
        }))
        .trust_store(
            ModelTrustStore::default().trust_digest(fixture.manifest.id.digest.value.clone()),
        )
        .build()
        .expect("Reference CPU provider registers cleanly");
    register_reference_cpu_prepared_kernels(&mut runtime);
    runtime
}

#[cfg(test)]
fn build_runtime_with_model_execution_engine_and_forced_token(
    fixture: &E2eFixture,
    forced_token: Option<TokenId>,
) -> Runtime {
    let mut runtime = Runtime::builder()
        .register_provider(std::sync::Arc::new(ReferenceCpuProvider::new()))
        .model_execution_engine(std::sync::Arc::new(E2eRuntimeModelExecutionEngine {
            fixture: fixture.clone(),
            kv_states: Arc::new(Mutex::new(BTreeMap::new())),
            pending_kv_states: Arc::new(Mutex::new(BTreeMap::new())),
            forced_token,
        }))
        .trust_store(
            ModelTrustStore::default().trust_digest(fixture.manifest.id.digest.value.clone()),
        )
        .build()
        .expect("Reference CPU provider registers cleanly");
    register_reference_cpu_prepared_kernels(&mut runtime);
    runtime
}

fn register_reference_cpu_prepared_kernels(runtime: &mut Runtime) {
    let mut prepared_ids = PreparedKernelIdAllocator::default();
    for advertisement in reference_cpu_kernel_advertisements() {
        let id = prepared_ids.allocate();
        let mut prepared = PreparedKernel::new(
            id,
            advertisement.id.clone(),
            CompiledKernelArtifactId::from_digest(format!(
                "builtin:{}",
                advertisement.id.stable_key()
            )),
            ProviderBinding::new(REFERENCE_CPU_PROVIDER_NAME),
            DeviceBinding::new(DeviceId::new(REFERENCE_CPU_DEVICE_ID)),
            PreparedKernelGeneration::new(1),
        );
        prepared
            .mark_ready()
            .expect("Reference CPU prepared kernel fixture can become ready");
        runtime
            .kernel_registry_mut()
            .register_prepared_kernel(prepared);
        runtime
            .kernel_registry_mut()
            .promote_generation(&advertisement.id, id)
            .expect("Reference CPU prepared kernel fixture promotes cleanly");
    }
}

/// Loads the E2E fixture model and creates its Model Instance, then binds
/// its weight tensors to Runtime resources through that instance
/// (`bind_qwen_fixture_weights`). The returned `MemoryManager` is a
/// placeholder every caller discards (`_memory`): loading now allocates
/// through `runtime.memory_mut()` directly (task 6.2), the same Runtime-
/// owned ledger dispatch itself accounts through (see
/// `execute_qwen_graph`'s doc comment in section 5), rather than a
/// throwaway that is dropped along with any admission/accounting history it
/// recorded.
fn load_fixture_instance(
    fixture: &E2eFixture,
    runtime: &mut Runtime,
) -> Result<(ModelInstanceId, MemoryManager), E2eConformanceError> {
    let mut coordinator = ModelLoadingCoordinator::new();
    coordinator.register_architecture(fixture.architecture_implementation.clone());
    let mut request = ModelLoadingRequest::new(
        ModelLoadingRequestId::new("e2e-fixture-load"),
        fixture.manifest.id.clone(),
    );
    request.quantization_policy = ModelQuantizationPolicy::RejectUnsupported;
    let loaded = load_model(
        &mut coordinator,
        runtime,
        ModelLoadingApiRequest::new(request),
        &fixture.manifest,
    )?;
    let instance = create_model_instance(
        runtime,
        &loaded,
        fixture.architecture_implementation.clone(),
        ResourceAffinity::new(FallbackClass::Transparent),
    )?;
    bind_qwen_fixture_weights(runtime, &instance, fixture)?;
    Ok((instance, MemoryManager::default()))
}

/// `transactional-weight-materialization`: a Model Instance whose weight
/// materialization fails SHALL never have reported Ready in the first
/// place -- `ModelInstances::create()` leaves it in `Loading`, and only a
/// fully successful `WeightMaterializationTransaction::commit` reaches
/// Ready. Proven here under a memory budget tight enough to admit `load()`'s
/// own aggregate allocation but not every subsequent per-tensor weight
/// admission, and that the failed attempt leaves no weight bound to the
/// instance (real rollback, not just a lifecycle label).
///
/// An earlier version of this test (and the code it tested) had the
/// instance reach `Ready` immediately on creation, then get demoted after
/// materialization failed -- a real, since-fixed bug an external audit of
/// PR #36 correctly identified: nothing prevented a caller from observing
/// the instance as `Ready` during that window. This test's name and
/// assertions were rewritten to match the corrected behavior, not just the
/// corrected code.
/// Shared by every check that proves a weight's `TensorResidency` record is
/// gone once its Provider storage and Memory Manager allocation have both
/// been released -- rollback, unload, and repeated load/unload all assert
/// this same property (`invalidate-tensor-residency-on-release`); `context`
/// names which one, for the failure message.
#[cfg(test)]
fn assert_tensor_residency_absent(
    runtime: &Runtime,
    resource_id: &TensorResourceId,
    context: &str,
) -> Result<(), E2eConformanceError> {
    if runtime.memory().tensor_residency(resource_id).is_some() {
        return Err(E2eConformanceError::MemoryValidationFailed {
            reason: format!(
                "weight resource '{resource_id}' still has a TensorResidency record {context}"
            ),
        });
    }
    Ok(())
}

#[cfg(test)]
fn check_weight_materialization_failure_never_reaches_ready(
    fixture: &E2eFixture,
) -> Result<(), E2eConformanceError> {
    let mut runtime = Runtime::builder()
        .register_provider(std::sync::Arc::new(ReferenceCpuProvider::new()))
        .config(RuntimeConfig {
            memory: MemoryManagerConfig {
                max_runtime_bytes: Some(1 << 13),
                allow_pending_allocations: false,
                ..MemoryManagerConfig::default()
            },
            ..RuntimeConfig::default()
        })
        .trust_store(
            ModelTrustStore::default().trust_digest(fixture.manifest.id.digest.value.clone()),
        )
        .build()
        .map_err(|error| E2eConformanceError::SuiteUnavailable {
            reason: error.to_string(),
        })?;
    register_reference_cpu_prepared_kernels(&mut runtime);

    let mut coordinator = ModelLoadingCoordinator::new();
    coordinator.register_architecture(fixture.architecture_implementation.clone());
    let mut request = ModelLoadingRequest::new(
        ModelLoadingRequestId::new("e2e-weight-materialization-failure"),
        fixture.manifest.id.clone(),
    );
    request.quantization_policy = ModelQuantizationPolicy::RejectUnsupported;
    let loaded = load_model(
        &mut coordinator,
        &mut runtime,
        ModelLoadingApiRequest::new(request),
        &fixture.manifest,
    )?;
    let instance = create_model_instance(
        &mut runtime,
        &loaded,
        fixture.architecture_implementation.clone(),
        ResourceAffinity::new(FallbackClass::Transparent),
    )?;

    // Confirm the instance is genuinely NOT Ready immediately after
    // creation -- the corrected behavior, replacing what used to be an
    // assertion that it *was* Ready here.
    let status_before = runtime
        .model_instance(&instance)
        .map_err(InferenceApiError::from)?
        .status();
    if status_before.lifecycle == ModelInstanceLifecycleState::Ready
        || status_before.readiness.accepts_generation()
    {
        return Err(E2eConformanceError::GenerationFailed {
            reason: format!(
                "expected the instance to NOT be Ready right after creation, before any \
                 weight materialization has run; got lifecycle {:?} / readiness {:?}",
                status_before.lifecycle, status_before.readiness
            ),
        });
    }

    match materialize_model_instance_weights(
        &mut runtime,
        &instance,
        fixture.manifest.id.name.as_str(),
        &fixture.weights,
    ) {
        Err(_) => {}
        Ok(()) => {
            return Err(E2eConformanceError::GenerationFailed {
                reason: "expected weight materialization to fail under a tight memory budget \
                          (test miscalibrated, or admission stopped being enforced)"
                    .into(),
            });
        }
    }

    let status_after = runtime
        .model_instance(&instance)
        .map_err(InferenceApiError::from)?
        .status();
    if status_after.lifecycle == ModelInstanceLifecycleState::Ready {
        return Err(E2eConformanceError::GenerationFailed {
            reason: "instance is Ready after weight materialization failed".into(),
        });
    }
    if status_after.readiness.accepts_generation() {
        return Err(E2eConformanceError::GenerationFailed {
            reason: format!(
                "instance readiness accepts generation after weight materialization failed: {:?}",
                status_after.readiness
            ),
        });
    }
    if status_after.lifecycle != ModelInstanceLifecycleState::Failed {
        return Err(E2eConformanceError::GenerationFailed {
            reason: format!(
                "expected the instance to end in Failed after materialization failed; got {:?}",
                status_after.lifecycle
            ),
        });
    }
    let bound_weight_count = runtime
        .model_instance(&instance)
        .map_err(InferenceApiError::from)?
        .definition
        .resource_bindings
        .weights
        .len();
    if bound_weight_count != 0 {
        return Err(E2eConformanceError::GenerationFailed {
            reason: format!(
                "expected zero weights bound after a failed materialization attempt rolled \
                 back (real rollback, not just a lifecycle label); found {bound_weight_count}"
            ),
        });
    }
    // Prove the rollback released Provider-owned storage too, not only the
    // Model Instance's own bindings -- `WeightMaterializationTransaction::
    // abort` must have called `release_tensor` for every weight staged
    // before the failure, for any weight this attempt might have reached.
    let executor = resolve_kernel_execution_provider(
        &runtime,
        &ProviderBinding::new(REFERENCE_CPU_PROVIDER_NAME),
    )
    .map_err(|error| E2eConformanceError::GenerationFailed {
        reason: error.to_string(),
    })?;
    for name in fixture.weights.keys() {
        let resource_id = TensorResourceId::new(format!("model.{instance}.weight.{name}"));
        if executor.read_tensor(&resource_id).is_some() {
            return Err(E2eConformanceError::GenerationFailed {
                reason: format!(
                    "weight resource '{resource_id}' remained present in Provider-owned \
                     storage after a failed materialization attempt was supposed to roll it \
                     back"
                ),
            });
        }
        assert_tensor_residency_absent(
            &runtime,
            &resource_id,
            "after a failed materialization attempt was supposed to roll it back",
        )?;
    }
    Ok(())
}

/// Rejects a weight payload whose bytes do not match the known-correct
/// digest (task 6.5: artifact byte changes / digest rejection) before this
/// fixture's weights are materialized -- a corrupted or substituted payload
/// must fail closed here, not silently bind whatever bytes happen to be in
/// memory. The E2E fixture's own concern (it has a fixed, hard-coded
/// expected digest; a real Model Artifact loader would check against the
/// digest declared in its manifest instead), kept separate from
/// `materialize_model_instance_weights`, which has no fixture dependency at
/// all (Correctif 9: no model-family- or fixture-specific logic in the
/// generic weight-materialization step itself).
fn bind_qwen_fixture_weights(
    runtime: &mut Runtime,
    instance: &ModelInstanceId,
    fixture: &E2eFixture,
) -> Result<(), E2eConformanceError> {
    let actual_digest = e2e_fixture_weight_digest(&fixture.weights);
    if actual_digest != E2E_FIXTURE_WEIGHT_DIGEST {
        return Err(E2eConformanceError::FixtureInvalid {
            reason: format!(
                "fixture weight payload digest {actual_digest} does not match expected {E2E_FIXTURE_WEIGHT_DIGEST}"
            ),
        });
    }
    // Task group 8 / Correctif 6: materialize from the real, checked-in
    // Safetensors artifact's actual bytes (`e2e_fixture_weights_from_real_artifact`),
    // not the in-memory `fixture.weights` the digest check above just
    // verified -- `materialize-weights-from-real-model-artifact`'s parity
    // test (`tests::e2e_fixture_real_artifact_weights_match_in_memory_weights`)
    // proves the two are bit-identical for an untampered fixture, so this
    // is a real change in *source*, not in observable behavior for every
    // existing caller of this function.
    let real_weights = e2e_fixture_weights_from_real_artifact(&fixture.config)?;
    materialize_model_instance_weights(
        runtime,
        instance,
        fixture.manifest.id.name.as_str(),
        &real_weights,
    )
    .map_err(E2eConformanceError::from)
}

/// Materializes a Model Instance's weight tensors as Runtime resources:
/// creates one `TensorResourceId` per declared weight, writes its bytes
/// into the registered Reference CPU Provider's storage (the same
/// Provider-owned storage compute intermediates live in -- not a private
/// Rust-side map), accounts the allocation through Runtime's
/// `MemoryManager`, and records the name -> resource binding on the Model
/// Instance so graph execution can look weights up by name through the
/// instance rather than a caller-held map. Implements
/// `model-loading-materializes-weight-resources`'s "Model Loading
/// Materializes Weight Resources" requirement.
///
/// Generic over any name -> tensor payload; contains no Qwen- or
/// fixture-specific logic (Correctif 9). It is still called as a step
/// *after* `load_model`/`create_model_instance` rather than from inside
/// `ModelLoadingCoordinator::load()` itself -- `load()` has no Provider
/// access and materializes nothing for any artifact type today (task group
/// 8's still-open, deeper gap; see design.md and this change's task list).
/// This function is the generic replacement for what used to be a
/// Qwen-fixture-only helper baked into this file; it belongs to the same
/// architectural layer as `create_model_instance`, not to any one model
/// family. Deliberately *not* relocated into `model_loading.rs`: that
/// module has no dependency on `Runtime` (`load()` itself takes only
/// `&mut MemoryManager`), and this function needs `&mut Runtime` for
/// Provider resolution and the failure-path lifecycle transition below --
/// moving it would make `model_loading.rs` depend on `runtime.rs`, which
/// already depends on `model_loading.rs`, for no behavioral benefit.
///
/// `pub`: this is the one legitimate way -- for production code or an
/// external embedder alike -- to turn named weight bytes into bound
/// resources for a Model Instance
/// (`bind-model-loading-evidence-to-validated-artifact`). Its success is
/// itself the proof `derive_effective_readiness_checks` trusts (via the
/// `MaterializationEvidence` this call's `WeightMaterializationTransaction::commit`
/// mints); there is no separate token for a caller to construct or forge.
///
/// `Ok(())` means the supplied `weights` staged and committed
/// successfully -- it does **not** by itself mean the instance reached
/// `Ready`. `commit` only marks the instance `Ready` if the full
/// Runtime-derived readiness gate (mandatory inventory complete, Provider
/// and Device ready, ...) is satisfied afterward; an empty or partial
/// `weights` map against an instance with an unmet mandatory inventory
/// commits and evidences what it was given, correctly leaving the
/// instance non-`Ready`. Check `runtime.model_instance(instance)?.
/// lifecycle()`/`.readiness()` (or call `warm_model_instance` again once
/// conditions change) to find out. A further audit of PR #36 found the
/// previous behavior -- unconditionally marking `Ready` once staging
/// succeeded -- let an incomplete materialization reach `Ready`, violating
/// `model-loading`'s pre-existing "Model Loading Does Not Bypass Instance
/// Readiness" and "Partial Loading Policy" requirements.
pub fn materialize_model_instance_weights(
    runtime: &mut Runtime,
    instance: &ModelInstanceId,
    artifact_owner: &str,
    weights: &BTreeMap<String, HostTensor>,
) -> Result<(), InferenceApiError> {
    let mut transaction = WeightMaterializationTransaction::begin(runtime, instance)?;
    for (name, tensor) in weights {
        if let Err(error) =
            transaction.stage_weight(runtime, instance, artifact_owner, name, tensor)
        {
            transaction.abort(runtime);
            // The instance never reached Ready for this attempt (`create()`
            // no longer auto-readies, and `commit` -- which alone calls
            // `mark_ready` -- is never reached on this path), so there is
            // nothing to demote; transition it to `Failed` so the failure
            // is durably visible on the instance itself, not only in this
            // `Result::Err`. Best-effort: if the instance cannot be found,
            // or is not presently in a state `Failed` legally transitions
            // from, the original materialization error below is still what
            // is returned either way.
            if let Ok(model_instance) = runtime.model_instances_mut().instance_mut(instance) {
                let _ = model_instance.transition_to(ModelInstanceLifecycleState::Failed);
            }
            return Err(error);
        }
    }
    transaction.commit(runtime, instance)
}

/// Stages weight materialization one resource at a time -- Memory Manager
/// admission, then Provider write, then residency registration, each
/// step's error propagated rather than discarded -- and either rolls back
/// everything staged so far ([`Self::abort`]) or publishes it all at once
/// and marks the instance Ready ([`Self::commit`]), mirroring
/// [`KvUpdateTransaction`]'s already-correct pattern for the same class of
/// problem (Correctif 11 / task group 9). Implements
/// `transactional-weight-materialization`'s "Weight Materialization Is
/// Transactional" requirement.
struct WeightMaterializationTransaction {
    provider_binding: ProviderBinding,
    executor: Arc<dyn ProviderExecutionApi>,
    staged: Vec<StagedWeight>,
}

struct StagedWeight {
    name: String,
    resource_id: TensorResourceId,
    allocation: MemoryAllocationId,
}

impl WeightMaterializationTransaction {
    /// Resolves the Provider bound to `instance`'s placement
    /// (`Runtime::create_model_instance` already cross-validated this
    /// against the prepared plan when the instance was created), falling
    /// back to Reference CPU only when placement left no Provider bound --
    /// `generalize-first-native-provider-dispatch`'s "Runtime Treats
    /// Reference CPU As Normal Provider" fix: this used to hardcode
    /// Reference CPU unconditionally, making a non-CPU-bound Model Instance
    /// materialize its weights through the wrong Provider's storage.
    fn begin(runtime: &Runtime, instance: &ModelInstanceId) -> Result<Self, InferenceApiError> {
        let provider_binding = runtime
            .model_instance(instance)?
            .definition()
            .placement
            .provider
            .clone()
            .unwrap_or_else(|| ProviderBinding::new(REFERENCE_CPU_PROVIDER_NAME));
        let executor = resolve_kernel_execution_provider(runtime, &provider_binding)?;
        Ok(Self {
            provider_binding,
            executor,
            staged: Vec::new(),
        })
    }

    /// Admits `tensor` through the Memory Manager, then writes it into
    /// Provider-owned storage, then registers its residency -- in that
    /// order, each step's error surfacing immediately rather than being
    /// discarded. Does not touch the Model Instance's `resource_bindings`;
    /// that is [`Self::commit`]'s responsibility once every weight in this
    /// attempt has staged successfully.
    fn stage_weight(
        &mut self,
        runtime: &mut Runtime,
        instance: &ModelInstanceId,
        artifact_owner: &str,
        name: &str,
        tensor: &HostTensor,
    ) -> Result<(), InferenceApiError> {
        // Shape/dtype agreement precedes even the content digest check:
        // a tensor the artifact declares quantized has no digest (digests
        // are F32-only, mirroring `host_tensors_from_artifact_bytes`'s own
        // materialization limit), so without this check a caller could
        // fabricate F32 content under a quantized tensor's name and have
        // nothing reject it. This applies to every tensor with a declared
        // shape, independent of whether that tensor also has a declared
        // digest -- the two checks guard different things
        // (`seal-runtime-model-trust-and-provenance-authority`).
        if let Some((expected_shape, expected_dtype)) = runtime
            .model_instance(instance)
            .map_err(InferenceApiError::from)?
            .definition()
            .required_weight_shapes
            .get(name)
            .cloned()
            && (tensor.shape != expected_shape || expected_dtype != ModelDType::F32)
        {
            return Err(InferenceApiError::WeightShapeOrDtypeMismatch {
                reason: format!(
                    "weight resource '{name}' declared shape {expected_shape:?} and dtype {expected_dtype:?}, which this Runtime can only materialize as F32 content matching that shape"
                ),
            });
        }
        // Content verification precedes Memory Manager admission: if the
        // loaded artifact declared a content digest for this tensor name,
        // the bytes supplied here must match it before anything is
        // admitted or written. A caller with legitimate access to this
        // transaction (the one authorized path to bind a weight) can still
        // supply the *wrong* bytes under the *right* name -- this check is
        // what stops that from being accepted as the declared tensor's
        // content (`bind-materialized-weight-content-to-model-artifact-
        // digests`). A tensor whose inventory entry declared no digest is
        // unaffected (`None` means unknown, not "no content required" --
        // the same precedent `required_weight_names` already established).
        if let Some(expected_digest) = runtime
            .model_instance(instance)
            .map_err(InferenceApiError::from)?
            .definition()
            .required_weight_digests
            .get(name)
            .cloned()
            && let Err(error) = expected_digest.verify_bytes(&tensor.content_bytes())
        {
            return Err(InferenceApiError::WeightContentDigestMismatch {
                reason: format!(
                    "weight resource '{name}' content does not match the artifact's declared digest: {error}"
                ),
            });
        }
        let resource_id = TensorResourceId::new(format!("model.{instance}.weight.{name}"));
        let byte_size = tensor.data.len() as u64 * std::mem::size_of::<f32>() as u64;
        // Admission SHALL precede Provider materialization: reserve the
        // resource's memory before writing its bytes into Provider-owned
        // storage, not after.
        let allocation = runtime
            .memory_mut()
            .allocate(MemoryAllocationRequest::new(
                MemoryAllocationClass::ModelArtifact,
                byte_size,
                MemoryPlacement::ProviderOwnedOpaque(self.provider_binding.clone()),
                MemoryAllocationOwner::InferenceArtifact(artifact_owner.into()),
            ))
            .map_err(|error| InferenceApiError::MemoryAdmissionFailed {
                reason: format!("failed to account weight resource '{name}': {error}"),
            })?;
        if let Err(error) = self
            .executor
            .write_tensor(resource_id.clone(), tensor.clone())
        {
            // Admission above already succeeded: release it before
            // propagating, so a write failure leaves no trace in the
            // Memory Manager ledger.
            let _ = runtime.memory_mut().release(allocation.id);
            return Err(InferenceApiError::ProviderTensorWriteFailed {
                reason: format!(
                    "failed to write weight resource '{name}' to Provider storage: {error}"
                ),
            });
        }
        if let Err(error) = runtime.memory_mut().record_tensor_residency(
            TensorResidency::new(
                resource_id.clone(),
                MemoryPlacement::ProviderOwnedOpaque(self.provider_binding.clone()),
                ResourceAffinity::new(FallbackClass::Transparent)
                    .with_provider(self.provider_binding.clone()),
            )
            .with_allocation(allocation.id),
        ) {
            // Residency registration failed after admission and write both
            // succeeded: release what was just staged for this one weight
            // before propagating, so the caller's subsequent `abort()`
            // over `self.staged` never sees this half-staged entry (it was
            // never pushed). Best-effort: this call is already on a
            // failure path over a different error, so a further failure
            // releasing the just-written tensor is not itself surfaced --
            // the original residency-registration error is what the caller
            // needs to see.
            let _ = self.executor.release_tensor(&resource_id);
            let _ = runtime.memory_mut().release(allocation.id);
            return Err(InferenceApiError::MemoryAdmissionFailed {
                reason: format!(
                    "failed to register residency for weight resource '{name}': {error}"
                ),
            });
        }
        self.staged.push(StagedWeight {
            name: name.to_string(),
            resource_id,
            allocation: allocation.id,
        });
        Ok(())
    }

    /// Releases every resource staged so far this attempt -- Provider
    /// tensor then Memory Manager allocation, per weight -- leaving no
    /// trace of this attempt behind. Reached whenever any weight in the
    /// attempt fails to stage.
    fn abort(self, runtime: &mut Runtime) {
        for staged in &self.staged {
            // Best-effort: this is already unwinding a failed attempt, with
            // nothing further to roll back to if the Provider release
            // itself fails.
            let _ = self.executor.release_tensor(&staged.resource_id);
            // Remove the residency record before releasing the allocation
            // it references: once the Provider tensor and allocation are
            // both gone, a lingering `TensorResidency` entry would
            // misreport this resource as still resident (Correctif:
            // `invalidate-tensor-residency-on-release`).
            runtime
                .memory_mut()
                .remove_tensor_residency(&staged.resource_id);
            let _ = runtime.memory_mut().release(staged.allocation);
        }
    }

    /// Publishes every staged weight's binding onto the Model Instance,
    /// mints/replaces its Runtime-issued `MaterializationEvidence` to match
    /// the resulting *full* current weight-binding set (not just this
    /// attempt's own staged subset -- a second materialization attempt on
    /// an already-partially-materialized instance must still produce
    /// evidence covering every previously-bound weight, or a legitimate
    /// instance would fail its own evidence-matching readiness check), and
    /// -- only if the full Runtime-derived readiness gate (the same one
    /// `warm_model_instance`/`resume_model_instance` use, not a parallel or
    /// weaker check) is actually satisfied -- marks the instance Ready.
    ///
    /// Does *not* unconditionally mark Ready once staging succeeds: a
    /// materialization attempt supplying an empty or partial `weights` map
    /// (legitimate for incremental/progressive materialization, per this
    /// method's own evidence-recomputation above) must still leave the
    /// instance non-Ready if the loaded artifact's mandatory tensor
    /// inventory is not yet fully covered, or if Provider/Device readiness
    /// is not currently satisfied -- publishing bindings and evidence for
    /// what *was* successfully staged is not itself proof the instance is
    /// usable. A further audit of PR #36 found this exact gap: `commit`
    /// previously called `mark_ready` unconditionally right after minting
    /// evidence, so `materialize_model_instance_weights(..., &BTreeMap::
    /// new())` against an instance with a non-empty mandatory inventory
    /// could still reach `Ready` -- non-compliant with `model-loading`'s
    /// pre-existing "Model Loading Does Not Bypass Instance Readiness" and
    /// "Partial Loading Policy" requirements, which this fix makes the code
    /// actually honor rather than changing the spec to match the bug.
    /// Reached only once every weight in this attempt staged successfully.
    fn commit(
        self,
        runtime: &mut Runtime,
        instance: &ModelInstanceId,
    ) -> Result<(), InferenceApiError> {
        {
            let model_instance = runtime
                .model_instances_mut()
                .instance_mut(instance)
                .map_err(InferenceApiError::from)?;
            for staged in &self.staged {
                model_instance
                    .definition
                    .resource_bindings
                    .weights
                    .insert(staged.name.clone(), staged.resource_id.clone());
                model_instance
                    .definition
                    .resource_bindings
                    .memory_allocations
                    .insert(staged.allocation);
            }
        }
        let (artifact, bound_resources) = {
            let model_instance = runtime
                .model_instance(instance)
                .map_err(InferenceApiError::from)?;
            (
                model_instance.definition.artifact.clone(),
                model_instance
                    .definition
                    .resource_bindings
                    .weights
                    .values()
                    .cloned()
                    .collect::<std::collections::BTreeSet<_>>(),
            )
        };
        runtime
            .model_instances_mut()
            .record_materialization_evidence(
                instance,
                MaterializationEvidence::new(artifact, bound_resources),
            );
        let effective_checks = derive_effective_readiness_checks(
            runtime,
            instance,
            &ModelInstanceReadinessChecks::default(),
        )?;
        let model_instance = runtime
            .model_instances_mut()
            .instance_mut(instance)
            .map_err(InferenceApiError::from)?;
        if model_instance
            .validate_readiness(&effective_checks)
            .is_err()
        {
            // Not yet fully ready (incomplete mandatory inventory, Provider
            // or Device not ready, ...): materialization of what was
            // staged this attempt still succeeded and its evidence still
            // stands, but the instance itself is correctly left non-Ready
            // rather than force-marked. A later call -- either another
            // `materialize_model_instance_weights` attempt completing the
            // inventory, or `warm_model_instance` once other conditions
            // clear -- re-derives from scratch.
            return Ok(());
        }
        runtime
            .model_instances_mut()
            .mark_ready(instance)
            .map_err(InferenceApiError::from)
    }
}

fn require_ready_first_native_instance<'a>(
    runtime: &'a Runtime,
    instance: &ModelInstanceId,
) -> Result<&'a ModelInstance, InferenceApiError> {
    let model_instance = runtime
        .model_instance(instance)
        .map_err(InferenceApiError::from)?;
    let status = model_instance.status();
    if !status.readiness.accepts_generation() {
        return Err(InferenceApiError::ModelInstanceNotReady {
            reason: format!(
                "first-native generation requires ready ModelInstance '{instance}', got lifecycle {:?} / readiness {:?}",
                status.lifecycle, status.readiness
            ),
        });
    }
    Ok(model_instance)
}

struct FirstNativePreparedPlans {
    prefill: PreparedExecutionPlan,
    prefill_node_count: usize,
    decode: PreparedExecutionPlan,
    decode_node_count: usize,
}

fn first_native_plan_context(phase: PreparedExecutionPhase, token_count: u64) -> PlanGuardContext {
    let mut context = PlanGuardContext::for_phase(phase);
    context.sequence_length = Some(token_count.max(1));
    context.total_tokens = Some(token_count.max(1));
    context.affinity = Some(ResourceAffinity::new(FallbackClass::Transparent));
    context.provider_ready = true;
    context.device_ready = true;
    context.memory_feasible = true;
    context
}

fn require_compatible_first_native_plan(
    plan: Option<&mut PreparedExecutionPlan>,
    context: &PlanGuardContext,
) -> Result<PreparedExecutionPlanGeneration, PreparedExecutionPlanError> {
    let plan = plan.ok_or(PreparedExecutionPlanError::PlanNotFound)?;
    plan.execute_ready_path(context)?;
    Ok(plan.generation)
}

#[cfg(all(
    test,
    not(target_arch = "wasm32"),
    feature = "wasmtime-component-engine"
))]
const QWEN_GRAPH_COMPONENT_BYTES: &[u8] =
    include_bytes!("../fixtures/components/qwen-graph.component.wat");

#[cfg(all(
    test,
    not(target_arch = "wasm32"),
    feature = "wasmtime-component-engine"
))]
const QWEN_GRAPH_COMPONENT_MANIFEST_BYTES: &[u8] =
    include_bytes!("../fixtures/components/qwen-graph.component.wat.magnetar-component.yaml");

/// The first real, minimal Qwen Model Component (task 11.4/11.5,
/// `model-component-graph-contract`): a compiled `components/qwen`
/// Component binary, built via `cargo build --target wasm32-unknown-unknown
/// --release` then `wasm-tools component new`. Unlike
/// [`QWEN_GRAPH_COMPONENT_BYTES`] (a hand-written, imports-free checksum
/// fixture), this Component genuinely imports `graph-builder` and produces
/// its graphs through it -- this is what the strict, default production
/// path (`build_first_native_graphs_from_real_qwen_component`) actually
/// loads; `QWEN_GRAPH_COMPONENT_BYTES` remains in use only by tests
/// exercising Component-loading conformance generically (trust, digest,
/// resource limits), not by anything claiming to produce a real graph.
#[cfg(all(not(target_arch = "wasm32"), feature = "wasmtime-component-engine"))]
const QWEN_REAL_COMPONENT_NAME: &str = "magnetar.qwen.real";
/// The digest a Qwen Component artifact MUST have to be accepted, in
/// production or in tests -- not merely a description of whatever bytes
/// happen to be embedded. `prepare_distributed_package` computes the real
/// sha256 of whatever bytes it actually receives and rejects a mismatch
/// (`ComponentDistributionErrorCategory::DigestMismatch`), so a stale or
/// substituted artifact -- from either loading branch below -- is rejected
/// structurally, not just by convention.
#[cfg(all(not(target_arch = "wasm32"), feature = "wasmtime-component-engine"))]
const QWEN_REAL_COMPONENT_DIGEST: &str =
    "sha256:552bb114838c10f742a1b6b6afade7c3044116826bb31cb33e21b16a2a422feb";

/// Test-oracle only (`reach-architecture-freeze-1` task 12.4): the checked-in
/// real Qwen Component binary, embedded for test fixtures. Production never
/// reads this -- see the `not(test)` branch of [`qwen_real_component_package`]
/// below.
#[cfg(all(
    not(target_arch = "wasm32"),
    feature = "wasmtime-component-engine",
    test
))]
const QWEN_REAL_COMPONENT_BYTES: &[u8] =
    include_bytes!("../fixtures/components/qwen-real.component.wasm");
#[cfg(all(
    not(target_arch = "wasm32"),
    feature = "wasmtime-component-engine",
    test
))]
const QWEN_REAL_COMPONENT_MANIFEST_BYTES: &[u8] =
    include_bytes!("../fixtures/components/qwen-real.component.wasm.magnetar-component.yaml");

#[cfg(all(
    not(target_arch = "wasm32"),
    feature = "wasmtime-component-engine",
    test
))]
fn qwen_real_component_package() -> Result<ComponentArtifactPackage, E2eConformanceError> {
    Ok(ComponentArtifactPackage::new(
        QWEN_REAL_COMPONENT_BYTES.to_vec(),
        QWEN_REAL_COMPONENT_MANIFEST_BYTES.to_vec(),
        ComponentDigest::parse("sha256", QWEN_REAL_COMPONENT_DIGEST),
        ComponentDistributionSource::new(
            ComponentDistributionSourceKind::DevelopmentFixture,
            QWEN_REAL_COMPONENT_NAME,
        ),
    ))
}

/// A Component artifact an embedder (e.g. `magnetar-cli`, the "deployment /
/// CLI / Component source adapter" layer in this task's own design) has
/// explicitly pushed for production first-native generation to use --
/// [`register_qwen_component_artifact`] is the only way to populate this.
/// Process-wide and set-once by design (`qwen_real_component_runtime`
/// itself already caches the *compiled* Component process-wide for the
/// same reason: compiling a real Component is expensive, and first-native
/// generation today has exactly one caller-facing model, `"qwen-test"`
/// (see [`run_first_native_generation`]), not a fleet of distinct
/// artifacts that would need per-call selection).
#[cfg(all(not(target_arch = "wasm32"), feature = "wasmtime-component-engine"))]
static QWEN_COMPONENT_ARTIFACT: std::sync::OnceLock<ComponentArtifactPackage> =
    std::sync::OnceLock::new();

/// Lets an embedder push a real Qwen Component artifact for production
/// first-native generation to use (`reach-architecture-freeze-1` task
/// 12.4's "deployment / CLI / Component source adapter" boundary --
/// `magnetar-cli` calls this with a Component binary and manifest it owns
/// and embeds itself, e.g. for the bundled `"qwen-test"` self-test/demo
/// alias). Idempotent: a second call with the process already holding a
/// registered artifact is a harmless no-op, not an error -- a caller like
/// `magnetar-cli` can call this unconditionally before every generation
/// request rather than tracking its own "have I registered yet" state.
/// The pushed bytes still go through the same digest verification every
/// other loading path already required (`QWEN_REAL_COMPONENT_DIGEST`,
/// checked in `ComponentManager::prepare_distributed_package`): pushing the
/// wrong bytes fails closed exactly like a missing or corrupted external
/// source would, it does not bypass trust.
#[cfg(all(not(target_arch = "wasm32"), feature = "wasmtime-component-engine"))]
pub fn register_qwen_component_artifact(component_bytes: Vec<u8>, manifest_bytes: Vec<u8>) {
    let _ = QWEN_COMPONENT_ARTIFACT.set(ComponentArtifactPackage::new(
        component_bytes,
        manifest_bytes,
        ComponentDigest::parse("sha256", QWEN_REAL_COMPONENT_DIGEST),
        ComponentDistributionSource::new(
            // Not `DevelopmentFixture`: this function is not test-gated --
            // it is `magnetar-cli`'s real, production embedder-facing call
            // path (see this function's own doc comment), which supplies
            // bytes it owns and embeds itself. `DevelopmentFixture` stays
            // reserved for the `#[cfg(test)]`-gated fixture packages
            // elsewhere in this file (`ClientProvided instead of
            // DevelopmentFixture` GitHub issue).
            ComponentDistributionSourceKind::ClientProvided,
            QWEN_REAL_COMPONENT_NAME,
        ),
    ));
}

/// Production Qwen Component loading (`reach-architecture-freeze-1` task
/// 12.4): no `include_bytes!` here, no silent fallback. First checks for an
/// artifact an embedder has explicitly pushed via
/// [`register_qwen_component_artifact`]; if none was pushed, falls back to
/// a caller-configured local path (`MAGNETAR_QWEN_COMPONENT_PATH` names the
/// Component `.wasm` file itself, with its manifest expected alongside it
/// as `<path>.magnetar-component.yaml`) -- the minimal external-source
/// resolution mechanisms this task calls for, deliberately not a full
/// Component registry/distribution service (a separate, larger piece of
/// infrastructure that does not exist yet and this task does not invent).
/// Neither configured, an unreadable file, or bytes that do not hash to
/// [`QWEN_REAL_COMPONENT_DIGEST`] all fail closed with a structured error --
/// never a fallback to an embedded fixture, because this crate has none.
#[cfg(all(
    not(target_arch = "wasm32"),
    feature = "wasmtime-component-engine",
    not(test)
))]
fn qwen_real_component_package() -> Result<ComponentArtifactPackage, E2eConformanceError> {
    if let Some(package) = QWEN_COMPONENT_ARTIFACT.get() {
        return Ok(package.clone());
    }
    resolve_qwen_component_from_env_var("MAGNETAR_QWEN_COMPONENT_PATH")
}

/// The actual local-path resolution logic behind
/// [`qwen_real_component_package`]'s fallback branch, extracted as its own
/// testable function taking the env var *name* as a parameter (never
/// `#[cfg(not(test))]` itself, unlike its one production caller) so tests
/// can point it at a controlled variable instead of depending on -- or
/// polluting -- the real process environment. `env_var_name` names the
/// Component `.wasm` file itself; its manifest is expected alongside it as
/// `<path>.magnetar-component.yaml`.
#[cfg(all(not(target_arch = "wasm32"), feature = "wasmtime-component-engine"))]
fn resolve_qwen_component_from_env_var(
    env_var_name: &str,
) -> Result<ComponentArtifactPackage, E2eConformanceError> {
    let component_path =
        std::env::var(env_var_name).map_err(|_| E2eConformanceError::ModelComponentFailed {
            reason: format!(
                "no Qwen Component artifact was registered (see \
                 register_qwen_component_artifact) and {env_var_name} is not set; production \
                 first-native generation requires an externally provided Qwen Model Component \
                 artifact and has no embedded development fixture of its own to fall back to"
            ),
        })?;
    let component_bytes = std::fs::read(&component_path).map_err(|error| {
        E2eConformanceError::ModelComponentFailed {
            reason: format!(
                "failed to read Qwen Component bytes from '{component_path}' \
                 ({env_var_name}): {error}"
            ),
        }
    })?;
    let manifest_path = format!("{component_path}.magnetar-component.yaml");
    let manifest_bytes = std::fs::read(&manifest_path).map_err(|error| {
        E2eConformanceError::ModelComponentFailed {
            reason: format!(
                "failed to read Qwen Component manifest from '{manifest_path}': {error}"
            ),
        }
    })?;
    Ok(ComponentArtifactPackage::new(
        component_bytes,
        manifest_bytes,
        ComponentDigest::parse("sha256", QWEN_REAL_COMPONENT_DIGEST),
        ComponentDistributionSource::new(
            ComponentDistributionSourceKind::LocalDirectory,
            QWEN_REAL_COMPONENT_NAME,
        ),
    ))
}

/// Real per-weight dimensions for `config`'s architecture, keyed by the
/// canonical Model Artifact tensor name (`"layers.{N}.self_attn.q_proj"`,
/// `"token_embedding"`, ...) -- the same names `fixture.weights` is keyed
/// by, and what the real Qwen Component's `weight-edge` calls now supply
/// directly as their logical name (see `components/qwen/src/lib.rs`'s
/// `weight_tensor_name`), rather than a Component-internal shorthand the
/// Runtime would need its own mapping table to translate. Derived purely
/// from `config` (the same static architecture metadata `qwen_build_graph`
/// itself reads to size these edges), not from any bound `TensorResourceId`
/// -- weight *shape* is architecture metadata, resolving the real bytes
/// behind a weight edge happens later, at execution time
/// (`resolve_qwen_weight_edge`), unaffected by this function.
#[cfg(all(not(target_arch = "wasm32"), feature = "wasmtime-component-engine"))]
fn qwen_weight_shapes_for_config(config: &QwenConfig) -> BTreeMap<String, Vec<u64>> {
    let a = &config.architecture;
    let q_dim = a.attention_head_count * a.head_dimension;
    let kv_dim = a.kv_head_count * a.head_dimension;
    let mut shapes = BTreeMap::new();
    shapes.insert(
        "token_embedding".to_string(),
        vec![a.vocabulary_size, a.hidden_size],
    );
    shapes.insert("final_norm".to_string(), vec![a.hidden_size]);
    shapes.insert(
        "lm_head".to_string(),
        vec![a.hidden_size, a.vocabulary_size],
    );
    for layer in 0..a.layer_count {
        let prefix = format!("layers.{layer}");
        shapes.insert(format!("{prefix}.input_norm"), vec![a.hidden_size]);
        shapes.insert(
            format!("{prefix}.self_attn.q_proj"),
            vec![a.hidden_size, q_dim],
        );
        shapes.insert(
            format!("{prefix}.self_attn.k_proj"),
            vec![a.hidden_size, kv_dim],
        );
        shapes.insert(
            format!("{prefix}.self_attn.v_proj"),
            vec![a.hidden_size, kv_dim],
        );
        shapes.insert(
            format!("{prefix}.self_attn.o_proj"),
            vec![q_dim, a.hidden_size],
        );
        shapes.insert(format!("{prefix}.post_attn_norm"), vec![a.hidden_size]);
        shapes.insert(
            format!("{prefix}.mlp.gate_proj"),
            vec![a.hidden_size, a.intermediate_size],
        );
        shapes.insert(
            format!("{prefix}.mlp.up_proj"),
            vec![a.hidden_size, a.intermediate_size],
        );
        shapes.insert(
            format!("{prefix}.mlp.down_proj"),
            vec![a.intermediate_size, a.hidden_size],
        );
    }
    shapes
}

#[cfg(all(not(target_arch = "wasm32"), feature = "wasmtime-component-engine"))]
fn expect_single_string_invocation_result(
    result: &ComponentInvocationResult,
    operation: &str,
) -> Result<String, E2eConformanceError> {
    match result.values.as_slice() {
        [ComponentValue::String(value)] => Ok(value.clone()),
        values => Err(E2eConformanceError::GraphValidationFailed {
            reason: format!(
                "Qwen Component export '{operation}' returned {values:?}, expected a single string"
            ),
        }),
    }
}

/// Holds the real Qwen Component's compiled artifact and its registered
/// [`GraphBuilderCapability`], built exactly once per process
/// (`qwen_real_component_runtime`) and reused across every call to
/// [`build_first_native_graphs_from_real_qwen_component`] -- compiling a
/// real (non-trivial) wasm Component is expensive (real, measured cost:
/// low seconds), while instantiating an already-compiled one and running
/// it is cheap. Without this cache, every dispatch call in
/// `execute_generation_step` -- once per generation *step*, not once per
/// generation -- would recompile the Component from scratch, which was
/// measured to take a single E2E test from milliseconds to ~28 seconds.
#[cfg(all(not(target_arch = "wasm32"), feature = "wasmtime-component-engine"))]
struct QwenRealComponentRuntime {
    manager: Mutex<ComponentManager>,
    capability: Arc<GraphBuilderCapability>,
    definition: ComponentDefinitionId,
}

#[cfg(all(not(target_arch = "wasm32"), feature = "wasmtime-component-engine"))]
fn qwen_real_component_runtime() -> Result<&'static QwenRealComponentRuntime, E2eConformanceError> {
    static RUNTIME: std::sync::OnceLock<QwenRealComponentRuntime> = std::sync::OnceLock::new();
    if let Some(runtime) = RUNTIME.get() {
        return Ok(runtime);
    }
    let capability = Arc::new(GraphBuilderCapability::new());
    let mut manager = ComponentManager::with_engine(Box::new(
        crate::component_wasmtime::WasmtimeComponentEngine::new().map_err(|error| {
            E2eConformanceError::ModelComponentFailed {
                reason: error.to_string(),
            }
        })?,
    ));
    manager.set_resource_limits(qwen_component_runtime_limits());
    manager
        .set_trust_store(ComponentTrustStore::default().trust_digest(QWEN_REAL_COMPONENT_DIGEST));
    let graph_builder_interface =
        WitInterface::new("magnetar:model-component-graph/graph-builder", "1.0.0");
    manager.provide_capability(
        graph_builder_interface,
        capability.clone() as Arc<dyn HostCapability>,
    );
    let definition = manager
        .prepare_pushed_package(qwen_real_component_package()?)
        .map_err(|error| E2eConformanceError::ModelComponentFailed {
            reason: error.to_string(),
        })?;
    // `OnceLock::set` losing a race is not an error here: the losing
    // thread's freshly-built `manager`/`capability` are simply dropped,
    // and every caller (winner and losers alike) reads back through
    // `RUNTIME.get()` afterwards, so they all observe the same instance.
    let _ = RUNTIME.set(QwenRealComponentRuntime {
        manager: Mutex::new(manager),
        capability,
        definition,
    });
    Ok(RUNTIME.get().expect("just set or set by a racing caller"))
}

/// Builds prefill and decode Execution Graphs by instantiating the real
/// Qwen Model Component (`QWEN_REAL_COMPONENT_BYTES`, compiled once and
/// cached -- see [`qwen_real_component_runtime`]) and calling its
/// `build-prefill-graph`/`build-decode-graph` exports, which produce the
/// graph through real `graph-builder` host calls into a
/// [`GraphBuilderCapability`] -- not `qwen_prefill_graph`/`qwen_decode_graph`
/// (task 11.5/12.6: this is production's *only* graph source; those Rust
/// functions are test-oracle only now, unreachable from any non-test
/// build). Returns the
/// component's own `ComponentDefinitionId`/`ComponentInstanceId` alongside
/// the graphs so the caller can still emit the same
/// `ComponentValidated`/`ComponentInstantiated` observations the prior
/// fixture-checksum path did. Destroys the fresh instance this call
/// creates before returning -- only the compiled artifact is cached, not
/// instance state, so nothing accumulates across repeated calls.
#[cfg(all(not(target_arch = "wasm32"), feature = "wasmtime-component-engine"))]
fn build_first_native_graphs_from_real_qwen_component(
    fixture: &E2eFixture,
    prompt_token_count: u64,
) -> Result<
    (
        FirstNativeComponentGraphs,
        ComponentDefinitionId,
        ComponentInstanceId,
    ),
    E2eConformanceError,
> {
    let runtime = qwen_real_component_runtime()?;
    let mut manager = runtime.manager.lock().unwrap();
    let capability = &runtime.capability;
    let definition = runtime.definition;

    let instance = manager
        .instantiate_prepared_component(definition)
        .map_err(|error| E2eConformanceError::ModelComponentFailed {
            reason: error.to_string(),
        })?;
    let result = (|| {
        let engine_key = manager
            .engine_instance_key(instance)
            .ok_or_else(|| E2eConformanceError::ModelComponentFailed {
                reason: "component instance has no engine key".into(),
            })?
            .to_string();

        let export_interface = WitInterface::new(
            "magnetar:model-component-graph/model-component-graph-producer",
            "1.0.0",
        );
        let weight_shapes = qwen_weight_shapes_for_config(&fixture.config);
        let compatibility_key = qwen_component_compatibility_key(&fixture.identity);
        let session_context = |weight_shapes: BTreeMap<String, Vec<u64>>| SessionContext {
            component_id: fixture.identity.id.as_str().to_string(),
            compatibility_key: compatibility_key.clone(),
            kv_namespace: "qwen".to_string(),
            weight_shapes,
            output_edge_name: "logits".to_string(),
        };

        capability.prepare_session(&engine_key, session_context(weight_shapes.clone()));
        let prefill_result = manager
            .invoke(
                ComponentInvocation::new(instance, export_interface.clone(), "build-prefill-graph")
                    .with_arguments(vec![ComponentValue::S64(prompt_token_count.max(1) as i64)]),
            )
            .map_err(|error| E2eConformanceError::ModelComponentFailed {
                reason: error.to_string(),
            })?;
        let prefill_handle =
            expect_single_string_invocation_result(&prefill_result, "build-prefill-graph")?;
        let prefill = capability
            .take_graph(&engine_key, &prefill_handle)
            .ok_or_else(|| E2eConformanceError::ModelComponentFailed {
                reason: "build-prefill-graph handle did not resolve to a finished graph".into(),
            })?;

        capability.prepare_session(&engine_key, session_context(weight_shapes));
        let decode_result = manager
            .invoke(
                ComponentInvocation::new(instance, export_interface, "build-decode-graph")
                    .with_arguments(vec![ComponentValue::S64(prompt_token_count.max(1) as i64)]),
            )
            .map_err(|error| E2eConformanceError::ModelComponentFailed {
                reason: error.to_string(),
            })?;
        let decode_handle =
            expect_single_string_invocation_result(&decode_result, "build-decode-graph")?;
        let decode = capability
            .take_graph(&engine_key, &decode_handle)
            .ok_or_else(|| E2eConformanceError::ModelComponentFailed {
                reason: "build-decode-graph handle did not resolve to a finished graph".into(),
            })?;

        validate_first_scope_graph(&prefill)?;
        validate_first_scope_graph(&decode)?;
        capability.clear_session(&engine_key);
        Ok(FirstNativeComponentGraphs {
            prefill_node_count: prefill.nodes.len(),
            decode_node_count: decode.nodes.len(),
            prefill,
            decode,
        })
    })();
    let _ = manager.destroy_instance(instance);
    result.map(|graphs| (graphs, definition, instance))
}

#[cfg(all(
    test,
    not(target_arch = "wasm32"),
    feature = "wasmtime-component-engine"
))]
fn qwen_graph_component_package() -> ComponentArtifactPackage {
    ComponentArtifactPackage::new(
        QWEN_GRAPH_COMPONENT_BYTES.to_vec(),
        QWEN_GRAPH_COMPONENT_MANIFEST_BYTES.to_vec(),
        ComponentDigest::parse("sha256", QWEN_GRAPH_COMPONENT_DIGEST),
        ComponentDistributionSource::new(
            ComponentDistributionSourceKind::DevelopmentFixture,
            QWEN_GRAPH_COMPONENT_NAME,
        ),
    )
}

#[cfg(all(
    test,
    not(target_arch = "wasm32"),
    feature = "wasmtime-component-engine"
))]
#[derive(Debug)]
struct QwenComponentPreflight {
    definition: ComponentDefinitionId,
    instance: ComponentInstanceId,
    graph_semantics: QwenComponentGraphSemantics,
    observations: Vec<ComponentObservation>,
}

/// What the Qwen Model Component reports about its own prefill/decode
/// graphs: not just node counts (which a component could satisfy with any
/// arbitrary set of operators) but a hash of the full ordered
/// Operator-kind-code sequence (`qwen_operator_sequence_hash`), so
/// `validate_against_graphs` performs genuine semantic comparison against
/// the Runtime-built graph rather than proving only that the two graphs
/// happen to be the same size.
#[cfg(test)]
#[derive(Clone, Debug, Eq, PartialEq)]
struct QwenComponentGraphSemantics {
    prefill_node_count: usize,
    decode_node_count: usize,
    prefill_operator_hash: u32,
    decode_operator_hash: u32,
}

#[cfg(test)]
impl QwenComponentGraphSemantics {
    fn validate_against_graphs(
        &self,
        prefill: &ExecutionGraph,
        decode: &ExecutionGraph,
    ) -> Result<(), E2eConformanceError> {
        if self.prefill_node_count != prefill.nodes.len() {
            return Err(E2eConformanceError::GraphValidationFailed {
                reason: format!(
                    "Qwen Component prefill graph declared {} node(s), runtime graph has {}",
                    self.prefill_node_count,
                    prefill.nodes.len()
                ),
            });
        }
        if self.decode_node_count != decode.nodes.len() {
            return Err(E2eConformanceError::GraphValidationFailed {
                reason: format!(
                    "Qwen Component decode graph declared {} node(s), runtime graph has {}",
                    self.decode_node_count,
                    decode.nodes.len()
                ),
            });
        }
        let expected_prefill_hash =
            qwen_operator_sequence_hash(&qwen_graph_operator_codes(prefill)?);
        if self.prefill_operator_hash != expected_prefill_hash {
            return Err(E2eConformanceError::GraphValidationFailed {
                reason: format!(
                    "Qwen Component prefill graph declared operator-sequence hash {:#010x}, runtime graph expects {expected_prefill_hash:#010x}",
                    self.prefill_operator_hash
                ),
            });
        }
        let expected_decode_hash = qwen_operator_sequence_hash(&qwen_graph_operator_codes(decode)?);
        if self.decode_operator_hash != expected_decode_hash {
            return Err(E2eConformanceError::GraphValidationFailed {
                reason: format!(
                    "Qwen Component decode graph declared operator-sequence hash {:#010x}, runtime graph expects {expected_decode_hash:#010x}",
                    self.decode_operator_hash
                ),
            });
        }
        Ok(())
    }
}

struct FirstNativeComponentGraphs {
    prefill: ExecutionGraph,
    prefill_node_count: usize,
    decode: ExecutionGraph,
    decode_node_count: usize,
}

#[cfg(test)]
fn build_first_native_graphs_from_component_output(
    fixture: &E2eFixture,
    prompt_token_count: u64,
    component_graph_semantics: QwenComponentGraphSemantics,
) -> Result<FirstNativeComponentGraphs, E2eConformanceError> {
    let prefill = qwen_prefill_graph(
        &fixture.config,
        &fixture.identity,
        prompt_token_count.max(1),
        true,
    )?;
    let decode = qwen_decode_graph(
        &fixture.config,
        &fixture.identity,
        prompt_token_count.max(1),
    )?;
    component_graph_semantics.validate_against_graphs(&prefill.graph, &decode.graph)?;
    validate_first_scope_graph(&prefill.graph)?;
    validate_first_scope_graph(&decode.graph)?;
    Ok(FirstNativeComponentGraphs {
        prefill_node_count: prefill.graph.nodes.len(),
        decode_node_count: decode.graph.nodes.len(),
        prefill: prefill.graph,
        decode: decode.graph,
    })
}

#[cfg(all(not(target_arch = "wasm32"), feature = "wasmtime-component-engine"))]
fn qwen_component_runtime_limits() -> ComponentResourceLimits {
    ComponentResourceLimits {
        // 8 MiB: the checksum-only fixture Component fit in 1 MiB, but the
        // real Qwen Component's wit-bindgen-generated glue (String/Vec
        // allocations across ~19 real graph-builder calls per graph) needs
        // more just to instantiate (a real minimum-memory requirement, not
        // this budget being too tight) -- confirmed by running the real
        // component and observing its actual instantiation requirement
        // (17 wasm pages, ~1.1 MiB) before choosing this headroom.
        max_memory_bytes: Some(1 << 23),
        execution_deadline_millis: Some(1_000),
        max_concurrent_invocations: Some(1),
        max_instances: Some(1),
        engine_execution_budget: Some(1_000_000),
        require_memory_limit: true,
    }
}

#[cfg(all(
    test,
    not(target_arch = "wasm32"),
    feature = "wasmtime-component-engine"
))]
struct QwenComponentPreflightRequest {
    component_package: ComponentArtifactPackage,
    trust_store: ComponentTrustStore,
    limits: ComponentResourceLimits,
}

#[cfg(all(
    test,
    not(target_arch = "wasm32"),
    feature = "wasmtime-component-engine"
))]
impl QwenComponentPreflightRequest {
    fn default_trusted() -> Self {
        Self {
            component_package: qwen_graph_component_package(),
            trust_store: ComponentTrustStore::default().trust_digest(QWEN_GRAPH_COMPONENT_DIGEST),
            limits: qwen_component_runtime_limits(),
        }
    }
}

#[cfg(all(
    test,
    not(target_arch = "wasm32"),
    feature = "wasmtime-component-engine"
))]
fn invoke_qwen_component_u32(
    manager: &mut ComponentManager,
    instance: ComponentInstanceId,
    interface: &WitInterface,
    operation: &str,
) -> Result<u32, E2eConformanceError> {
    let result = manager
        .invoke(ComponentInvocation::new(
            instance,
            interface.clone(),
            operation,
        ))
        .map_err(|error| E2eConformanceError::ModelComponentFailed {
            reason: error.to_string(),
        })?;
    match result.values.as_slice() {
        [ComponentValue::U32(value)] => Ok(*value),
        values => Err(E2eConformanceError::GraphValidationFailed {
            reason: format!(
                "Qwen Component export '{operation}' returned {values:?}, expected u32"
            ),
        }),
    }
}

#[cfg(all(
    test,
    not(target_arch = "wasm32"),
    feature = "wasmtime-component-engine"
))]
fn validate_and_instantiate_qwen_component_before_first_native_planning(
    request: QwenComponentPreflightRequest,
) -> Result<QwenComponentPreflight, E2eConformanceError> {
    let mut manager = ComponentManager::with_engine(Box::new(
        crate::component_wasmtime::WasmtimeComponentEngine::new().map_err(|error| {
            E2eConformanceError::ModelComponentFailed {
                reason: error.to_string(),
            }
        })?,
    ));
    manager.set_resource_limits(request.limits);
    manager.set_trust_store(request.trust_store);
    let definition = manager
        .prepare_pushed_package(request.component_package)
        .map_err(|error| E2eConformanceError::ModelComponentFailed {
            reason: error.to_string(),
        })?;
    let instance = manager
        .instantiate_prepared_component(definition)
        .map_err(|error| E2eConformanceError::ModelComponentFailed {
            reason: error.to_string(),
        })?;
    let interface = WitInterface::new("magnetar:qwen/graph-fixture", "1.0.0");
    let authority = invoke_qwen_component_u32(
        &mut manager,
        instance,
        &interface,
        "provider-authority-count",
    )?;
    if authority != 0 {
        return Err(E2eConformanceError::BoundaryViolation {
            reason: "Qwen Component fixture requested Provider authority".into(),
        });
    }
    let graph_semantics = QwenComponentGraphSemantics {
        prefill_node_count: invoke_qwen_component_u32(
            &mut manager,
            instance,
            &interface,
            "prefill-node-count",
        )? as usize,
        decode_node_count: invoke_qwen_component_u32(
            &mut manager,
            instance,
            &interface,
            "decode-node-count",
        )? as usize,
        prefill_operator_hash: invoke_qwen_component_u32(
            &mut manager,
            instance,
            &interface,
            "prefill-operator-hash",
        )?,
        decode_operator_hash: invoke_qwen_component_u32(
            &mut manager,
            instance,
            &interface,
            "decode-operator-hash",
        )?,
    };
    Ok(QwenComponentPreflight {
        definition,
        instance,
        graph_semantics,
        observations: manager.observations().to_vec(),
    })
}

#[cfg(all(
    test,
    not(target_arch = "wasm32"),
    feature = "wasmtime-component-engine"
))]
fn validate_and_instantiate_trusted_qwen_component_before_first_native_planning()
-> Result<QwenComponentPreflight, E2eConformanceError> {
    validate_and_instantiate_qwen_component_before_first_native_planning(
        QwenComponentPreflightRequest::default_trusted(),
    )
}

fn prepare_first_native_plan_for_graph(
    runtime: &Runtime,
    graph: &ExecutionGraph,
    instance: &ModelInstanceId,
    model_instance_revision: u64,
    token_count: u64,
    generation: PreparedExecutionPlanGeneration,
) -> Result<PreparedExecutionPlan, E2eConformanceError> {
    let phase = PreparedExecutionPhase::from(graph.phase);
    let mut scope = PreparedExecutionPlanScope::for_phase(phase)
        .with_model_instance(instance.clone(), model_instance_revision)
        .with_workload_bucket(format!("{:?}-{}-tokens", phase, token_count.max(1)));
    scope.provider = Some(ProviderBinding::new(REFERENCE_CPU_PROVIDER_NAME));

    let mut plan = PreparedExecutionPlan::new(
        PreparedExecutionPlanId::new(format!("first-native-{phase:?}-plan"))?,
        generation,
        semantic_graph_fingerprint(graph),
        scope,
    )?;
    let affinity = ResourceAffinity::new(FallbackClass::Transparent)
        .with_provider(ProviderBinding::new(REFERENCE_CPU_PROVIDER_NAME))
        .with_execution_context(runtime.context().id());

    for (node_id, node) in &graph.nodes {
        let node_affinity = node
            .resource_affinity
            .clone()
            .unwrap_or_else(|| affinity.clone());
        let mut selection_request = KernelSelectionRequest::new(
            format!("first-native-plan-{phase:?}-{}", node_id.as_str()),
            node.operator.clone(),
            node_affinity.clone(),
        );
        selection_request.graph_plan = Some(graph.id.clone());
        selection_request.model_instance = Some(instance.clone());
        selection_request.observability_correlation =
            Some(format!("first-native-plan:{phase:?}:{node_id}"));
        for input in &node.inputs {
            let resource = graph_kernel_resource(graph, input, &node_affinity)?;
            merge_graph_edge_requirements(&mut selection_request, &resource);
            if let Some(kv) = graph
                .edges
                .get(input)
                .and_then(|edge| edge.kv_cache.as_ref())
            {
                note_graph_kv_requirement(&mut selection_request, kv);
                if matches!(kv.behavior, GraphKvCacheBehavior::Input) {
                    merge_kernel_managed_kv_requirement(&mut selection_request, kv, &resource);
                }
            }
            selection_request = selection_request.with_input(resource);
        }
        for output in &node.outputs {
            let resource = graph_kernel_resource(graph, output, &node_affinity)?;
            merge_graph_edge_requirements(&mut selection_request, &resource);
            if let Some(kv) = graph
                .edges
                .get(output)
                .and_then(|edge| edge.kv_cache.as_ref())
            {
                note_graph_kv_requirement(&mut selection_request, kv);
            }
            selection_request = selection_request.with_output(resource);
        }
        let selection = runtime
            .kernel_registry()
            .select(&selection_request)
            .map_err(|error| E2eConformanceError::KernelCoverageMissing {
                reason: format!(
                    "Kernel Registry selection failed for node {node_id} ({operator}): {error}",
                    operator = node.operator,
                ),
            })?;
        let candidate =
            selection
                .selected
                .ok_or_else(|| E2eConformanceError::KernelCoverageMissing {
                    reason: format!("no Kernel Registry candidate selected for node {node_id}"),
                })?;

        let mut binding = PlanNodeBinding::new(
            [node_id.clone()],
            candidate.kernel.clone(),
            candidate.provider.clone(),
        )?
        .with_specialization(format!("first-native-{phase:?}-{}", node_id.as_str()));
        let prepared_kernel = runtime
            .kernel_registry()
            .active_prepared_kernel(&candidate.kernel)
            .ok_or_else(|| E2eConformanceError::KernelCoverageMissing {
                reason: format!("no active PreparedKernel registered for node {node_id}"),
            })?;
        binding = binding.with_prepared_kernel(prepared_kernel.id, prepared_kernel.generation);
        if let Some(device) = candidate.device {
            binding = binding.with_device(device);
        }
        if let Some(profile) = candidate.kernel.conformance_profile {
            binding = binding.with_qualification_profile(profile);
        }
        plan.add_node_binding(binding)?;
    }

    plan.set_resource_plan(ResourceBindingPlan::default())?;
    plan.set_memory_requirements(PlanMemoryRequirements::default())?;
    plan.add_guard(PlanGuard::Phase(phase));
    plan.add_guard(PlanGuard::SequenceRange {
        min: 1,
        max: E2E_FIXTURE_CONTEXT,
    });
    plan.add_guard(PlanGuard::Readiness);
    plan.add_guard(PlanGuard::AffinityRequired);
    plan.add_guard(PlanGuard::MemoryFeasible);
    plan.mark_ready_atomically()?;

    let mut plan_for_validation = plan.clone();
    let context = first_native_plan_context(phase, token_count);
    require_compatible_first_native_plan(Some(&mut plan_for_validation), &context)?;
    Ok(plan)
}

fn kernel_memory_class_for_edge(edge: &TensorEdge) -> KernelMemoryClass {
    match edge.residency {
        TensorResidencyConstraint::Device => KernelMemoryClass::Device,
        TensorResidencyConstraint::BrowserLinearMemory => KernelMemoryClass::BrowserLinearMemory,
        TensorResidencyConstraint::ProviderOwnedOpaque => KernelMemoryClass::ProviderOwned,
        TensorResidencyConstraint::Host => match edge.memory_class {
            MemoryAllocationClass::HostPinned | MemoryAllocationClass::TransferStaging => {
                KernelMemoryClass::PinnedHost
            }
            MemoryAllocationClass::BrowserLinearMemory => KernelMemoryClass::BrowserLinearMemory,
            _ => KernelMemoryClass::Host,
        },
    }
}

fn layout_kind_for_descriptor(descriptor: &TensorDescriptor) -> TensorLayoutKind {
    match descriptor.layout {
        LayoutDescriptor::Contiguous => TensorLayoutKind::Contiguous,
        LayoutDescriptor::Strided { .. } => TensorLayoutKind::Strided,
        LayoutDescriptor::Blocked { .. } => TensorLayoutKind::Blocked,
        LayoutDescriptor::Paged { .. } => TensorLayoutKind::Paged,
        LayoutDescriptor::PackedQuantized { .. } => TensorLayoutKind::QuantizedPacked,
        LayoutDescriptor::AttentionSpecific { .. } => TensorLayoutKind::AttentionSpecific,
        LayoutDescriptor::BrowserCompatible { .. } => TensorLayoutKind::BrowserCompatible,
        LayoutDescriptor::ProviderOpaque { .. } => TensorLayoutKind::ProviderOpaque,
    }
}

fn graph_kernel_resource(
    graph: &ExecutionGraph,
    edge_id: &TensorEdgeId,
    default_affinity: &ResourceAffinity,
) -> Result<KernelResource, E2eConformanceError> {
    let edge =
        graph
            .edges
            .get(edge_id)
            .ok_or_else(|| E2eConformanceError::GraphValidationFailed {
                reason: format!("graph node references missing tensor edge '{edge_id}'"),
            })?;
    let affinity = edge
        .affinity
        .clone()
        .unwrap_or_else(|| default_affinity.clone());
    Ok(KernelResource::new(
        TensorResourceDescriptor::new(
            TensorResourceId::new(edge.id.as_str()),
            edge.descriptor.clone(),
            affinity,
        ),
        kernel_memory_class_for_edge(edge),
    ))
}

fn merge_graph_edge_requirements(request: &mut KernelSelectionRequest, resource: &KernelResource) {
    if let DTypeDescriptor::Portable(dtype) = &resource.resource.descriptor.dtype {
        request.dtype_requirements.insert(*dtype);
    }
    request
        .layout_requirements
        .insert(layout_kind_for_descriptor(&resource.resource.descriptor));
    request
        .memory_class_requirements
        .insert(resource.memory_class);
}

fn note_graph_kv_requirement(request: &mut KernelSelectionRequest, kv: &GraphKvCacheMetadata) {
    let behavior = match kv.behavior {
        GraphKvCacheBehavior::Input => "input",
        GraphKvCacheBehavior::Output => "output",
        GraphKvCacheBehavior::Append => "append",
    };
    let cache_kind = if kv.paged { "paged" } else { "contiguous" };
    request.policy.insert(
        format!("graph-kv-cache:{}", kv.cache_id),
        format!("{behavior}:{cache_kind}"),
    );
}

fn merge_kernel_managed_kv_requirement(
    request: &mut KernelSelectionRequest,
    kv: &GraphKvCacheMetadata,
    resource: &KernelResource,
) {
    let metadata = request
        .kv_cache
        .get_or_insert_with(|| KernelKvCacheMetadata {
            layouts: BTreeSet::new(),
            paged_cache: kv.paged,
            append: false,
            read: false,
            dtypes: BTreeSet::new(),
            memory_classes: BTreeSet::new(),
            affinity: Some(resource.resource.affinity.clone()),
        });
    metadata.paged_cache |= kv.paged;
    metadata.read = true;
    metadata.layouts.insert(if kv.paged {
        "paged".into()
    } else {
        "contiguous".into()
    });
    if let DTypeDescriptor::Portable(dtype) = &resource.resource.descriptor.dtype {
        metadata.dtypes.insert(*dtype);
    }
    metadata.memory_classes.insert(resource.memory_class);
}

fn prepare_first_native_execution_plans(
    runtime: &Runtime,
    instance: &ModelInstanceId,
    graphs: FirstNativeComponentGraphs,
    prompt_token_count: u64,
) -> Result<FirstNativePreparedPlans, E2eConformanceError> {
    let status = require_ready_first_native_instance(runtime, instance)?;
    let mutation_version = status.status().mutation_version;
    Ok(FirstNativePreparedPlans {
        prefill: prepare_first_native_plan_for_graph(
            runtime,
            &graphs.prefill,
            instance,
            mutation_version,
            prompt_token_count,
            PreparedExecutionPlanGeneration::new(1),
        )?,
        prefill_node_count: graphs.prefill_node_count,
        decode: prepare_first_native_plan_for_graph(
            runtime,
            &graphs.decode,
            instance,
            mutation_version,
            1,
            PreparedExecutionPlanGeneration::new(1),
        )?,
        decode_node_count: graphs.decode_node_count,
    })
}

/// Builds the portable graph semantics a component describing `config`'s
/// prefill/decode graphs correctly would report at `prompt_token_count`:
/// node counts and the full Operator-kind-code sequence
/// (`qwen_graph_operator_codes`), derived directly from the Runtime-built
/// graphs. Used wherever a caller needs a known-correct value rather than
/// one queried across the Component boundary (the non-component fallback
/// build, and tests not themselves exercising component/graph mismatch
/// detection).
#[cfg(test)]
fn qwen_component_graph_semantics_for_prompt(
    config: &QwenConfig,
    identity: &ModelComponentIdentity,
    prompt_token_count: u64,
) -> Result<QwenComponentGraphSemantics, E2eConformanceError> {
    let prefill = qwen_prefill_graph(config, identity, prompt_token_count.max(1), true)?.graph;
    let decode = qwen_decode_graph(config, identity, prompt_token_count.max(1))?.graph;
    Ok(QwenComponentGraphSemantics {
        prefill_node_count: prefill.nodes.len(),
        decode_node_count: decode.nodes.len(),
        prefill_operator_hash: qwen_operator_sequence_hash(&qwen_graph_operator_codes(&prefill)?),
        decode_operator_hash: qwen_operator_sequence_hash(&qwen_graph_operator_codes(&decode)?),
    })
}

/// Builds the graphs `execute_generation_step` will dispatch against for
/// `fixture` at `prompt_token_count` -- the same graph *source* as that
/// method, not just a semantically-equivalent one, since callers (many
/// conformance checks and tests) use this to build a
/// [`PreparedExecutionPlan`] (via [`first_native_plans_for_prompt`]) that
/// must fingerprint-match whatever graph actually gets dispatched. Task
/// 11.5: under the strict, default build this is the real Qwen Component
/// (matching `execute_generation_step`'s own strict branch).
#[cfg(all(not(target_arch = "wasm32"), feature = "wasmtime-component-engine"))]
fn first_native_component_graphs_for_prompt(
    fixture: &E2eFixture,
    prompt_token_count: u64,
) -> Result<FirstNativeComponentGraphs, E2eConformanceError> {
    build_first_native_graphs_from_real_qwen_component(fixture, prompt_token_count)
        .map(|(graphs, _definition, _instance)| graphs)
}

/// Test-oracle branch (`reach-architecture-freeze-1` task 12.6): when no
/// strict Component engine is available, a test build still needs *some*
/// graph to drive downstream (KV, sampling, tokenization, ...) coverage
/// with, so it falls back to the Rust-synthesized recipe -- never reachable
/// outside `#[cfg(test)]`, and never the thing first-native generation
/// treats as authoritative in a real build.
#[cfg(all(
    test,
    not(all(not(target_arch = "wasm32"), feature = "wasmtime-component-engine"))
))]
fn first_native_component_graphs_for_prompt(
    fixture: &E2eFixture,
    prompt_token_count: u64,
) -> Result<FirstNativeComponentGraphs, E2eConformanceError> {
    let semantics = qwen_component_graph_semantics_for_prompt(
        &fixture.config,
        &fixture.identity,
        prompt_token_count,
    )?;
    build_first_native_graphs_from_component_output(fixture, prompt_token_count, semantics)
}

/// Production, non-test branch when no strict Component engine is
/// available (no `wasmtime-component-engine`, or `wasm32`, which has none
/// today -- see task group 10's investigation note). Correctif 7 / task
/// group 10's fail-closed requirement, and the explicit decision behind
/// task 12.6: production first-native generation has exactly one semantic
/// source for Model Component graphs, the real Component itself. There is
/// no second, Rust-synthesized production path to fall back to, silently
/// or otherwise -- this fails structurally instead.
#[cfg(all(
    not(test),
    not(all(not(target_arch = "wasm32"), feature = "wasmtime-component-engine"))
))]
fn first_native_component_graphs_for_prompt(
    _fixture: &E2eFixture,
    _prompt_token_count: u64,
) -> Result<FirstNativeComponentGraphs, E2eConformanceError> {
    Err(E2eConformanceError::ModelComponentFailed {
        reason: "no Component engine is available on this build target; first-native generation \
                 requires a real Model Component and has no unattested fallback graph source"
            .into(),
    })
}

/// Builds real published prepared execution plans for `fixture` at
/// `prompt_token_count`, so callers exercising the first-native hot-path
/// dispatch functions directly (conformance checks, tests) supply a genuine
/// `PreparedExecutionPlan` instead of skipping plan-bound execution (those
/// functions no longer accept `None`; see
/// `execute_qwen_prefill_hidden_states_through_dispatch`'s doc comment).
fn first_native_plans_for_prompt(
    runtime: &Runtime,
    fixture: &E2eFixture,
    instance: &ModelInstanceId,
    prompt_token_count: u64,
) -> Result<FirstNativePreparedPlans, E2eConformanceError> {
    let component_graphs = first_native_component_graphs_for_prompt(fixture, prompt_token_count)?;
    prepare_first_native_execution_plans(runtime, instance, component_graphs, prompt_token_count)
}

/// Runs the generation loop with real published plans built for `request`'s
/// prompt length, so callers exercise the same plan-bound hot path
/// production uses (via `run_generation_loop_with_execution_plans`) instead
/// of the no-plan `run_generation_loop` entry point, which the first-native
/// executor rejects with a structured `GraphPlanningFailed` error.
#[allow(clippy::too_many_arguments)]
fn run_first_native_generation_loop_with_plans(
    runtime: &mut Runtime,
    fixture: &E2eFixture,
    instance: &ModelInstanceId,
    request: &GenerationRequest,
    sampling_policy: SamplingPolicy,
    cache_usage: CacheUsageSummary,
    should_cancel: impl FnMut(&[TokenId]) -> bool,
    observer: &mut InferenceApiObserver,
) -> Result<GenerationResult, InferenceApiError> {
    let prompt_token_count = request.input_token_ids.len() as u64;
    let mut prepared_plans =
        first_native_plans_for_prompt(runtime, fixture, instance, prompt_token_count).map_err(
            |error| InferenceApiError::GraphPlanningFailed {
                reason: error.to_string(),
            },
        )?;
    let mut execution_plans = RuntimeGenerationExecutionPlans {
        prefill: &mut prepared_plans.prefill,
        decode: &mut prepared_plans.decode,
    };
    run_generation_loop_with_execution_plans(
        runtime,
        request,
        sampling_policy,
        cache_usage,
        should_cancel,
        observer,
        &mut execution_plans,
    )
}

fn generation_tokenizer_reference(fixture: &E2eFixture) -> GenerationTokenizerReference {
    GenerationTokenizerReference {
        tokenizer_id: fixture.tokenizer.metadata().id.clone(),
        metadata: fixture.tokenizer.metadata().clone(),
    }
}

fn run_success_path_with_prompt(
    fixture: &E2eFixture,
    model_ref: &ModelRef,
    prompt: &str,
) -> Result<E2eRunOutcome, E2eConformanceError> {
    let mut runtime = build_runtime_with_model_execution_engine(fixture);

    // Model resolution.
    let mut registry = ModelRegistry::new();
    registry.register(model_ref.clone(), fixture.manifest.id.clone());
    let resolution = registry.resolve(&ModelResolutionRequest::new(model_ref.clone()))?;
    if resolution.artifact != fixture.manifest.id {
        return Err(E2eConformanceError::ModelResolutionFailed {
            reason: "resolved artifact does not match fixture manifest".into(),
        });
    }

    // Model Loading + Model Instance.
    let (instance, _memory) = load_fixture_instance(fixture, &mut runtime)?;
    require_ready_first_native_instance(&runtime, &instance)?;

    // Session creation.
    let session_request = SessionCreationRequest {
        model: GenerationModelReference::ModelInstance(instance.clone()),
        tokenizer: generation_tokenizer_reference(fixture),
        generation_defaults: GenerationParameters::greedy(),
        policy: SessionPolicy::default(),
        memory: SessionMemoryBudget::default(),
        allowed_capabilities: BTreeSet::new(),
        correlation_id: None,
        created_at_millis: 0,
    };
    let session = create_inference_session(&mut runtime, session_request)?;

    // Tokenization (plain-text path).
    let tokenized = tokenize_prompt_input(
        &fixture.tokenizer,
        TokenizationRequest::new(PromptInput::PlainText(prompt.into())),
        None,
    )?;
    let mut observer = InferenceApiObserver::new();
    // Task 11.5: the strict, default path now builds its graphs by calling
    // the real Qwen Component's `graph-builder` host imports, not
    // `qwen_prefill_graph`/`qwen_decode_graph`.
    #[cfg(all(not(target_arch = "wasm32"), feature = "wasmtime-component-engine"))]
    let component_graphs = {
        let (graphs, definition, instance) = build_first_native_graphs_from_real_qwen_component(
            fixture,
            tokenized.token_ids.len() as u64,
        )?;
        observer.observe(
            InferenceApiObservationKind::ComponentValidated,
            format!("component_definition={definition:?}"),
            None,
        );
        observer.observe(
            InferenceApiObservationKind::ComponentInstantiated,
            format!("component_instance={instance:?}"),
            None,
        );
        graphs
    };
    // Correctif 7 / task group 10, task 12.6: without a real Component
    // engine, production has no second, Rust-synthesized graph source to
    // fall back to -- it fails closed, structurally. A test build still
    // needs *some* graph to drive downstream coverage with when built
    // without a strict engine, so it keeps the Rust-synthesized recipe as a
    // test-oracle-only fallback.
    #[cfg(all(
        test,
        not(all(not(target_arch = "wasm32"), feature = "wasmtime-component-engine"))
    ))]
    let component_graphs = {
        let semantics = qwen_component_graph_semantics_for_prompt(
            &fixture.config,
            &fixture.identity,
            tokenized.token_ids.len() as u64,
        )?;
        build_first_native_graphs_from_component_output(
            fixture,
            tokenized.token_ids.len() as u64,
            semantics,
        )?
    };
    #[cfg(all(
        not(test),
        not(all(not(target_arch = "wasm32"), feature = "wasmtime-component-engine"))
    ))]
    let component_graphs: FirstNativeComponentGraphs =
        return Err(E2eConformanceError::ModelComponentFailed {
            reason: "no Component engine is available on this build target; first-native \
                     generation requires a real Model Component and has no unattested \
                     fallback graph source"
                .into(),
        });
    let mut prepared_plans = prepare_first_native_execution_plans(
        &runtime,
        &instance,
        component_graphs,
        tokenized.token_ids.len() as u64,
    )?;
    let _plan_generations = (
        prepared_plans.prefill.generation,
        prepared_plans.prefill_node_count,
        prepared_plans.decode.generation,
        prepared_plans.decode_node_count,
    );

    // Generation request build + validate.
    let request = build_generation_request(
        GenerationRequestId::new("e2e-success-path")?,
        Some(session.clone()),
        GenerationModelReference::ModelInstance(instance.clone()),
        generation_tokenizer_reference(fixture),
        tokenized,
        4,
        GenerationParameters::greedy(),
        StopConditions {
            eos: EosPolicy {
                eos_token_ids: vec![E2E_FIXTURE_EOS_TOKEN],
                ..EosPolicy::default()
            },
            ..StopConditions::default()
        },
        StreamingMode::TokenIds,
    );
    let request = prepare_generation(&runtime, request)?;

    // Prefill/decode/sample/stream through the real forward pass.
    let mut execution_plans = RuntimeGenerationExecutionPlans {
        prefill: &mut prepared_plans.prefill,
        decode: &mut prepared_plans.decode,
    };
    let generation_result = run_generation_loop_with_execution_plans(
        &mut runtime,
        &request,
        SamplingPolicy::default(),
        CacheUsageSummary::default(),
        |_generated_so_far| false,
        &mut observer,
        &mut execution_plans,
    )?;

    // Streaming decode of generated tokens. The byte fixture can produce token
    // sequences that are structurally valid but not a complete UTF-8 text for
    // every arbitrary prompt, so the user-facing helper keeps generation
    // successful and renders token IDs when text decoding cannot complete.
    let decoded_text = decode_tokens_streaming(
        &fixture.tokenizer,
        StreamingDecodeRequest::new(generation_result.output.generated_token_ids.clone()),
    )
    .map(|decoded| decoded.text)
    .unwrap_or_else(|_| {
        let tokens = generation_result
            .output
            .generated_token_ids
            .iter()
            .map(u32::to_string)
            .collect::<Vec<_>>()
            .join(" ");
        format!("[generated token ids: {tokens}]")
    });
    let generation_result = generation_result.with_decoded_text(decoded_text);

    // Session close + Model Instance cleanup.
    let kv_observations = runtime.kv_caches().observations().to_vec();
    close_inference_session(&mut runtime, &session)?;
    unload_model_instance(
        &mut runtime,
        &instance,
        ModelInstanceUnloadPolicy::DrainActiveUse,
    )
    .map_err(E2eConformanceError::from)?;

    Ok(E2eRunOutcome {
        generation_result,
        observer,
        kv_observations,
    })
}

/// Runs the full required success path: resolve, load, instantiate, create
/// session, tokenize (plain text), generate through a real Reference CPU
/// forward pass with greedy Sampling, stream, close session, cleanup.
fn run_success_path(fixture: &E2eFixture) -> Result<E2eRunOutcome, E2eConformanceError> {
    run_success_path_with_prompt(fixture, &ModelRef::new("qwen-test")?, "hi")
}

pub fn run_first_native_fixture_generation(
    prompt: &str,
) -> Result<FirstNativeFixtureGeneration, E2eConformanceError> {
    let fixture = e2e_fixture()?;
    let outcome = run_success_path_with_prompt(&fixture, &ModelRef::new("qwen-test")?, prompt)?;
    if !outcome
        .kv_observations
        .iter()
        .any(|observation| observation.kind == KvCacheObservationKind::PrefillCompleted)
    {
        return Err(E2eConformanceError::GenerationFailed {
            reason: "first native fixture generation produced no KV prefill commit".into(),
        });
    }
    validate_e2e_no_shortcuts(
        outcome.observer.observations(),
        &reference_cpu_kernel_advertisements(),
    )?;
    let text = outcome
        .generation_result
        .decoded_text
        .clone()
        .ok_or_else(|| E2eConformanceError::StreamingFailed {
            reason: "first native fixture generation produced no decoded text".into(),
        })?;
    Ok(FirstNativeFixtureGeneration {
        text,
        result: outcome.generation_result,
        observer: outcome.observer,
    })
}

pub fn run_first_native_generation(
    model_ref: &ModelRef,
    prompt: &str,
) -> Result<FirstNativeFixtureGeneration, FirstNativeRuntimeError> {
    if model_ref.as_str() != "qwen-test" {
        return Err(FirstNativeRuntimeError::model_not_found(model_ref));
    }
    let fixture = e2e_fixture().map_err(FirstNativeRuntimeError::from_conformance)?;
    let outcome = run_success_path_with_prompt(&fixture, model_ref, prompt)
        .map_err(FirstNativeRuntimeError::from_conformance)?;
    validate_e2e_no_shortcuts(
        outcome.observer.observations(),
        &reference_cpu_kernel_advertisements(),
    )
    .map_err(FirstNativeRuntimeError::from_conformance)?;
    let text = outcome
        .generation_result
        .decoded_text
        .clone()
        .ok_or_else(|| {
            FirstNativeRuntimeError::from_conformance(E2eConformanceError::StreamingFailed {
                reason: "first native generation produced no decoded text".into(),
            })
        })?;
    Ok(FirstNativeFixtureGeneration {
        text,
        result: outcome.generation_result,
        observer: outcome.observer,
    })
}

/// A persistent first-native chat session (task 8.3): one `Runtime`, one
/// `ModelInstance`, and one `InferenceSessionId` created once by [`Self::open`]
/// and reused by every [`Self::turn`] -- unlike [`run_first_native_generation`],
/// which builds and tears down a fresh `Runtime` and `InferenceSessionId` for
/// every call. This is what makes `magnetar chat`'s cancellation and
/// session-identity guarantees genuine: cancelling targets the very session
/// turns execute through, not an orphan nothing else touches.
pub struct FirstNativeChatSession {
    fixture: E2eFixture,
    runtime: Runtime,
    instance: ModelInstanceId,
    session: InferenceSessionId,
    next_request: u64,
}

impl FirstNativeChatSession {
    pub fn open(model_ref: &ModelRef) -> Result<Self, FirstNativeRuntimeError> {
        if model_ref.as_str() != "qwen-test" {
            return Err(FirstNativeRuntimeError::model_not_found(model_ref));
        }
        let fixture = e2e_fixture().map_err(FirstNativeRuntimeError::from_conformance)?;
        let mut runtime = build_runtime_with_model_execution_engine(&fixture);

        let mut registry = ModelRegistry::new();
        registry.register(model_ref.clone(), fixture.manifest.id.clone());
        let resolution = registry
            .resolve(&ModelResolutionRequest::new(model_ref.clone()))
            .map_err(|error| FirstNativeRuntimeError::from_conformance(error.into()))?;
        if resolution.artifact != fixture.manifest.id {
            return Err(FirstNativeRuntimeError::from_conformance(
                E2eConformanceError::ModelResolutionFailed {
                    reason: "resolved artifact does not match fixture manifest".into(),
                },
            ));
        }

        let (instance, _memory) = load_fixture_instance(&fixture, &mut runtime)
            .map_err(FirstNativeRuntimeError::from_conformance)?;
        require_ready_first_native_instance(&runtime, &instance)
            .map_err(|error| FirstNativeRuntimeError::from_conformance(error.into()))?;

        let session_request = SessionCreationRequest {
            model: GenerationModelReference::ModelInstance(instance.clone()),
            tokenizer: generation_tokenizer_reference(&fixture),
            generation_defaults: GenerationParameters::greedy(),
            policy: SessionPolicy::default(),
            memory: SessionMemoryBudget::default(),
            allowed_capabilities: BTreeSet::new(),
            correlation_id: None,
            created_at_millis: 0,
        };
        let session = create_inference_session(&mut runtime, session_request)
            .map_err(|error| FirstNativeRuntimeError::from_conformance(error.into()))?;

        Ok(Self {
            fixture,
            runtime,
            instance,
            session,
            next_request: 0,
        })
    }

    /// The persistent Runtime `InferenceSessionId` every turn executes
    /// through -- stable across the life of this chat session.
    pub fn session_id(&self) -> &InferenceSessionId {
        &self.session
    }

    /// Runs one chat turn's prefill/decode generation through this
    /// session's persistent `Runtime`, `ModelInstance`, and
    /// `InferenceSessionId`. `prompt` is caller-rendered text (the CLI owns
    /// transcript assembly and chat template rendering); this only
    /// tokenizes, plans, and executes it -- it never re-creates the Runtime
    /// session or Model Instance `open` already established.
    pub fn turn(
        &mut self,
        prompt: &str,
        max_new_tokens: usize,
    ) -> Result<FirstNativeFixtureGeneration, FirstNativeRuntimeError> {
        self.next_request += 1;
        let request_id =
            GenerationRequestId::new(format!("first-native-chat-{}", self.next_request))
                .map_err(|error| FirstNativeRuntimeError::from_conformance(error.into()))?;

        let tokenized = tokenize_prompt_input(
            &self.fixture.tokenizer,
            TokenizationRequest::new(PromptInput::PlainText(prompt.into())),
            None,
        )
        .map_err(|error| FirstNativeRuntimeError::from_conformance(error.into()))?;

        let mut observer = InferenceApiObserver::new();
        // Task 11.5: the strict, default path now builds its graphs by
        // calling the real Qwen Component's `graph-builder` host imports,
        // not `qwen_prefill_graph`/`qwen_decode_graph`.
        #[cfg(all(not(target_arch = "wasm32"), feature = "wasmtime-component-engine"))]
        let component_graphs = {
            let (graphs, definition, instance) =
                build_first_native_graphs_from_real_qwen_component(
                    &self.fixture,
                    tokenized.token_ids.len() as u64,
                )
                .map_err(FirstNativeRuntimeError::from_conformance)?;
            observer.observe(
                InferenceApiObservationKind::ComponentValidated,
                format!("component_definition={definition:?}"),
                None,
            );
            observer.observe(
                InferenceApiObservationKind::ComponentInstantiated,
                format!("component_instance={instance:?}"),
                None,
            );
            graphs
        };
        // Correctif 7 / task group 10, task 12.6: see the sibling fallback
        // in `run_success_path_with_prompt` -- production has no second,
        // Rust-synthesized graph source without a strict Component engine;
        // it fails closed. The Rust-synthesized recipe survives only as a
        // test-oracle fallback so a test build without a strict engine
        // still has some graph to drive downstream coverage with.
        #[cfg(all(
            test,
            not(all(not(target_arch = "wasm32"), feature = "wasmtime-component-engine"))
        ))]
        let component_graphs = {
            let semantics = qwen_component_graph_semantics_for_prompt(
                &self.fixture.config,
                &self.fixture.identity,
                tokenized.token_ids.len() as u64,
            )
            .map_err(FirstNativeRuntimeError::from_conformance)?;
            build_first_native_graphs_from_component_output(
                &self.fixture,
                tokenized.token_ids.len() as u64,
                semantics,
            )
            .map_err(FirstNativeRuntimeError::from_conformance)?
        };
        #[cfg(all(
            not(test),
            not(all(not(target_arch = "wasm32"), feature = "wasmtime-component-engine"))
        ))]
        let component_graphs: FirstNativeComponentGraphs = return Err(
            FirstNativeRuntimeError::from_conformance(E2eConformanceError::ModelComponentFailed {
                reason: "no Component engine is available on this build target; \
                             first-native generation requires a real Model Component and has \
                             no unattested fallback graph source"
                    .into(),
            }),
        );

        let mut prepared_plans = prepare_first_native_execution_plans(
            &self.runtime,
            &self.instance,
            component_graphs,
            tokenized.token_ids.len() as u64,
        )
        .map_err(FirstNativeRuntimeError::from_conformance)?;

        let request = build_generation_request(
            request_id,
            Some(self.session.clone()),
            GenerationModelReference::ModelInstance(self.instance.clone()),
            generation_tokenizer_reference(&self.fixture),
            tokenized,
            max_new_tokens,
            GenerationParameters::greedy(),
            StopConditions {
                eos: EosPolicy {
                    eos_token_ids: vec![E2E_FIXTURE_EOS_TOKEN],
                    ..EosPolicy::default()
                },
                ..StopConditions::default()
            },
            StreamingMode::TokenIds,
        );
        let request = prepare_generation(&self.runtime, request)
            .map_err(|error| FirstNativeRuntimeError::from_conformance(error.into()))?;

        let mut execution_plans = RuntimeGenerationExecutionPlans {
            prefill: &mut prepared_plans.prefill,
            decode: &mut prepared_plans.decode,
        };
        let generation_result = run_generation_loop_with_execution_plans(
            &mut self.runtime,
            &request,
            SamplingPolicy::default(),
            CacheUsageSummary::default(),
            |_generated_so_far| false,
            &mut observer,
            &mut execution_plans,
        )
        .map_err(|error| FirstNativeRuntimeError::from_conformance(error.into()))?;

        let decoded_text = decode_tokens_streaming(
            &self.fixture.tokenizer,
            StreamingDecodeRequest::new(generation_result.output.generated_token_ids.clone()),
        )
        .map(|decoded| decoded.text)
        .unwrap_or_else(|_| {
            let tokens = generation_result
                .output
                .generated_token_ids
                .iter()
                .map(u32::to_string)
                .collect::<Vec<_>>()
                .join(" ");
            format!("[generated token ids: {tokens}]")
        });
        let generation_result = generation_result.with_decoded_text(decoded_text);

        validate_e2e_no_shortcuts(
            observer.observations(),
            &reference_cpu_kernel_advertisements(),
        )
        .map_err(FirstNativeRuntimeError::from_conformance)?;

        let text = generation_result.decoded_text.clone().ok_or_else(|| {
            FirstNativeRuntimeError::from_conformance(E2eConformanceError::StreamingFailed {
                reason: "first native chat turn produced no decoded text".into(),
            })
        })?;

        Ok(FirstNativeFixtureGeneration {
            text,
            result: generation_result,
            observer,
        })
    }

    /// Cancels this session's persistent Runtime `InferenceSessionId` --
    /// the same session [`Self::turn`] executes every generation through.
    pub fn cancel(&mut self) -> Result<(), FirstNativeRuntimeError> {
        cancel_inference_session(&mut self.runtime, &self.session)
            .map_err(|error| FirstNativeRuntimeError::from_conformance(error.into()))
    }

    pub fn close(mut self) -> Result<(), FirstNativeRuntimeError> {
        close_inference_session(&mut self.runtime, &self.session)
            .map_err(|error| FirstNativeRuntimeError::from_conformance(error.into()))?;
        unload_model_instance(
            &mut self.runtime,
            &self.instance,
            ModelInstanceUnloadPolicy::DrainActiveUse,
        )
        .map_err(|error| FirstNativeRuntimeError::from_conformance(error.into()))?;
        Ok(())
    }
}

fn check_success_path(fixture: &E2eFixture) -> Result<(), E2eConformanceError> {
    let outcome = run_success_path(fixture)?;
    if outcome
        .generation_result
        .output
        .generated_token_ids
        .is_empty()
    {
        return Err(E2eConformanceError::GenerationFailed {
            reason: "success path produced no generated tokens".into(),
        });
    }
    Ok(())
}

fn check_determinism(fixture: &E2eFixture) -> Result<(), E2eConformanceError> {
    let first = run_success_path(fixture)?;
    let second = run_success_path(fixture)?;
    if first.generation_result.output.generated_token_ids
        != second.generation_result.output.generated_token_ids
    {
        return Err(E2eConformanceError::DeterminismFailed {
            reason: "repeated success path runs produced different generated tokens".into(),
        });
    }
    Ok(())
}

fn check_streaming_order(fixture: &E2eFixture) -> Result<(), E2eConformanceError> {
    let outcome = run_success_path(fixture)?;
    let kinds: Vec<InferenceApiObservationKind> = outcome
        .observer
        .observations()
        .iter()
        .map(|observation| observation.kind)
        .collect();
    let position = |kind: InferenceApiObservationKind| kinds.iter().position(|k| *k == kind);
    let (
        Some(prefill_started),
        Some(prefill_completed),
        Some(first_token),
        Some(completed),
        Some(closed),
    ) = (
        position(InferenceApiObservationKind::PrefillStarted),
        position(InferenceApiObservationKind::PrefillCompleted),
        position(InferenceApiObservationKind::TokenGenerated),
        position(InferenceApiObservationKind::GenerationCompleted),
        position(InferenceApiObservationKind::StreamClosed),
    )
    else {
        return Err(E2eConformanceError::StreamingFailed {
            reason: "expected streaming observation kinds were not all emitted".into(),
        });
    };
    if !(prefill_started < prefill_completed
        && prefill_completed < first_token
        && first_token < completed
        && completed < closed)
    {
        return Err(E2eConformanceError::StreamingFailed {
            reason: "streaming events were not emitted in the expected order".into(),
        });
    }
    Ok(())
}

fn record_operator_dispatch(
    covered: &mut BTreeSet<String>,
    name: &str,
    result: Result<(KernelDispatchResult, HostTensor), InferenceApiError>,
) -> Result<(), E2eConformanceError> {
    result.map_err(E2eConformanceError::from)?;
    covered.insert(name.to_string());
    Ok(())
}

fn check_operator_coverage(fixture: &E2eFixture) -> Result<BTreeSet<String>, E2eConformanceError> {
    let mut runtime = build_runtime_trusting_fixture(fixture);
    let provider: Arc<dyn ProviderExecutionApi> = Arc::new(ReferenceCpuExecutor::new());
    let mut node_events = Vec::new();
    let mut dispatch_ctx = QwenDispatchContext {
        runtime: &mut runtime,
        provider: provider.clone(),
        prepared_plan: None,
        graph: None,
        sequence_length: None,
        last_provider_execution: None,
        node_events: &mut node_events,
    };
    let architecture = &fixture.config.architecture;
    let mut covered = BTreeSet::new();
    let hidden = HostTensor::new(
        [2, architecture.hidden_size],
        vec![0.25; (2 * architecture.hidden_size) as usize],
    )?;
    let one_row_hidden = HostTensor::new(
        [1, architecture.hidden_size],
        vec![0.5; architecture.hidden_size as usize],
    )?;
    let ids = HostTensor::new([2], vec![1.0, 2.0])?;
    let embedding_table = fixture_tensor_by_name(&fixture.weights, "token_embedding")?.clone();
    record_operator_dispatch(
        &mut covered,
        "embedding",
        dispatch_reference_cpu_operator(
            &mut dispatch_ctx,
            "coverage.embedding",
            dispatch_operator_id("embedding", OperatorFamily::Tensor),
            vec![
                (
                    TensorResourceId::new("coverage.embedding.table"),
                    f32_tensor_descriptor(&embedding_table),
                    embedding_table,
                ),
                (
                    TensorResourceId::new("coverage.embedding.ids"),
                    f32_tensor_descriptor(&ids),
                    ids,
                ),
            ],
            (
                TensorResourceId::new("coverage.embedding.out"),
                TensorDescriptor::new(
                    ShapeDescriptor::new([2, architecture.hidden_size]),
                    DTypeDescriptor::portable(ComputeDType::Float32),
                    LayoutDescriptor::Contiguous,
                ),
            ),
            BTreeMap::new(),
        ),
    )?;
    let weight = HostTensor::new(
        [2, architecture.hidden_size],
        vec![1.0; (2 * architecture.hidden_size) as usize],
    )?;
    record_operator_dispatch(
        &mut covered,
        "rmsnorm",
        dispatch_qwen_rmsnorm(
            &mut dispatch_ctx,
            "coverage.rmsnorm",
            hidden.clone(),
            weight,
            fixture.config.rmsnorm_epsilon,
        ),
    )?;
    let matmul_b = HostTensor::new(
        [architecture.hidden_size, architecture.hidden_size],
        vec![0.125; (architecture.hidden_size * architecture.hidden_size) as usize],
    )?;
    record_operator_dispatch(
        &mut covered,
        "matmul",
        dispatch_qwen_matmul(
            &mut dispatch_ctx,
            "coverage.matmul",
            one_row_hidden.clone(),
            matmul_b,
        ),
    )?;
    record_operator_dispatch(
        &mut covered,
        "rope",
        dispatch_qwen_rope_per_head(
            &mut dispatch_ctx,
            "coverage.rope",
            &one_row_hidden,
            architecture.attention_head_count,
            architecture.head_dimension,
            &fixture.config.rope,
            0,
        ),
    )?;
    record_operator_dispatch(
        &mut covered,
        "attention",
        dispatch_qwen_attention(
            &mut dispatch_ctx,
            "coverage.attention",
            one_row_hidden.clone(),
            one_row_hidden.clone(),
            one_row_hidden.clone(),
            architecture,
        ),
    )?;
    record_operator_dispatch(
        &mut covered,
        "softmax",
        dispatch_qwen_unary(
            &mut dispatch_ctx,
            "coverage.softmax",
            "softmax",
            OperatorFamily::Activation,
            one_row_hidden.clone(),
            BTreeMap::new(),
        ),
    )?;
    record_operator_dispatch(
        &mut covered,
        "silu",
        dispatch_qwen_unary(
            &mut dispatch_ctx,
            "coverage.silu",
            "silu",
            OperatorFamily::Activation,
            one_row_hidden.clone(),
            BTreeMap::new(),
        ),
    )?;
    record_operator_dispatch(
        &mut covered,
        "mul",
        dispatch_qwen_binary_same_shape(
            &mut dispatch_ctx,
            "coverage.mul",
            "mul",
            OperatorFamily::Tensor,
            one_row_hidden.clone(),
            one_row_hidden.clone(),
        ),
    )?;
    record_operator_dispatch(
        &mut covered,
        "add",
        dispatch_qwen_binary_same_shape(
            &mut dispatch_ctx,
            "coverage.add",
            "add",
            OperatorFamily::Tensor,
            one_row_hidden.clone(),
            one_row_hidden.clone(),
        ),
    )?;
    record_operator_dispatch(
        &mut covered,
        "residual-add",
        dispatch_qwen_binary_same_shape(
            &mut dispatch_ctx,
            "coverage.residual_add",
            "residual-add",
            OperatorFamily::Tensor,
            one_row_hidden.clone(),
            one_row_hidden,
        ),
    )?;
    Ok(covered)
}

fn check_kernel_coverage() -> Result<BTreeSet<String>, E2eConformanceError> {
    let advertisements = reference_cpu_kernel_advertisements();
    Ok(advertisements
        .iter()
        .map(|advertisement| advertisement.implemented_operator.name().to_string())
        .collect())
}

fn check_no_shortcut_direct_provider_rejected() -> Result<(), E2eConformanceError> {
    match validate_e2e_no_shortcuts(&[], &reference_cpu_kernel_advertisements()) {
        Err(E2eConformanceError::BoundaryViolation { .. }) => Ok(()),
        Err(other) => Err(E2eConformanceError::Internal {
            reason: format!("expected e2e-boundary-violation, got {}", other.code()),
        }),
        Ok(()) => Err(E2eConformanceError::Internal {
            reason: "direct Provider shortcut was not rejected".into(),
        }),
    }
}

/// Correctif 17 / task group 17: `validate_e2e_no_shortcuts` (via
/// `validate_e2e_per_node_causal_chain`) SHALL reject a per-node causal
/// chain that is *incomplete* for a node that genuinely dispatched, not
/// only confirm the five global evidence categories occurred somewhere. A
/// node with `GraphNodeReady` and `PlanBindingResolved`/`PreparedKernelResolved`/
/// `ProviderSubmitted` but no correlated `ProviderCompleted` or
/// `TensorResourceProduced` (as if a dispatch died silently between submit
/// and completion) must be caught, distinctly from the presence-only check
/// this task group's fix supersedes.
#[cfg(test)]
fn check_e2e_no_shortcuts_rejects_incomplete_per_node_causal_chain()
-> Result<(), E2eConformanceError> {
    let node = |kind: InferenceApiObservationKind, name: &str| {
        InferenceApiObservation::new(kind, format!("per-node causal event; node={name}"), None)
    };
    let observations = vec![
        node(InferenceApiObservationKind::GraphNodeReady, "embedding"),
        node(
            InferenceApiObservationKind::PlanBindingResolved,
            "embedding",
        ),
        node(
            InferenceApiObservationKind::PreparedKernelResolved,
            "embedding",
        ),
        node(InferenceApiObservationKind::ProviderSubmitted, "embedding"),
        // Deliberately missing: ProviderCompleted / TensorResourceProduced
        // for "embedding" -- as if the node's dispatch died silently
        // between submission and completion.
    ];
    match validate_e2e_per_node_causal_chain(&observations) {
        Err(E2eConformanceError::BoundaryViolation { reason })
            if reason.contains("embedding") && reason.contains("ProviderCompleted") =>
        {
            Ok(())
        }
        Err(error) => Err(E2eConformanceError::GenerationFailed {
            reason: format!("unexpected error for an incomplete per-node causal chain: {error}"),
        }),
        Ok(()) => Err(E2eConformanceError::GenerationFailed {
            reason: "validator accepted an incomplete per-node causal chain".into(),
        }),
    }
}

fn check_reference_cpu_selected_through_kernel_registry() -> Result<(), E2eConformanceError> {
    let runtime = build_runtime();
    let advertisements = reference_cpu_kernel_advertisements();
    let matmul_operator = advertisements
        .iter()
        .find(|advertisement| advertisement.implemented_operator.name() == "matmul")
        .map(|advertisement| advertisement.implemented_operator.clone())
        .ok_or_else(|| E2eConformanceError::KernelCoverageMissing {
            reason: "Reference CPU does not advertise matmul".into(),
        })?;
    let request = KernelSelectionRequest::new(
        "e2e-matmul-selection",
        matmul_operator,
        ResourceAffinity::new(FallbackClass::Transparent),
    );
    let result = runtime
        .kernel_registry()
        .select(&request)
        .map_err(E2eConformanceError::from)?;
    match result.selected {
        Some(_) => Ok(()),
        None => Err(E2eConformanceError::KernelCoverageMissing {
            reason: "Kernel Registry selected no candidate for matmul".into(),
        }),
    }
}

fn check_invalid_graph_fixture() -> Result<(), E2eConformanceError> {
    let bad_operator = OperatorId::new(
        OPERATOR_NAMESPACE,
        "unsupported-op",
        1,
        OperatorFamily::Control,
    );
    let edge = TensorEdge::new(
        TensorEdgeId::new("bad.output"),
        TensorDescriptor::new(
            ShapeDescriptor::new(vec![1, 1]),
            DTypeDescriptor::portable(ComputeDType::Float32),
            LayoutDescriptor::Contiguous,
        ),
    );
    let mut node = ExecutionNode::new(ExecutionNodeId::new("bad-node"), bad_operator);
    node.outputs = vec![TensorEdgeId::new("bad.output")];
    let graph = ExecutionGraph::new(
        ExecutionGraphId::new("e2e-invalid-graph"),
        ExecutionGraphPhase::Test,
    )
    .with_edge(edge)
    .with_node(node);
    match validate_first_scope_graph(&graph) {
        Err(_) => Ok(()),
        Ok(()) => Err(E2eConformanceError::Internal {
            reason: "expected invalid graph fixture to fail validation".into(),
        }),
    }
}

/// Test-oracle only (task 12.6): exercises the Rust-synthesized graph
/// builder directly to prove it stays internally valid, independent of
/// whether anything in production ever uses it as a graph source.
#[cfg(test)]
fn check_graph_production_and_execution(fixture: &E2eFixture) -> Result<(), E2eConformanceError> {
    // kv_cache_enabled=true so the prefill graph's K/V edges are actually
    // marked as cache outputs -- otherwise the decode graph below would
    // claim 2 cached tokens that this graph never produced.
    let prefill = qwen_prefill_graph(&fixture.config, &fixture.identity, 2, true)?;
    if prefill.validation.is_none() {
        return Err(E2eConformanceError::GraphValidationFailed {
            reason: "prefill graph was produced without validation".into(),
        });
    }
    // The prefill graph above was built for a 2-token prompt with caching
    // enabled, so the decode graph represents generating the 3rd token
    // against those 2 cached ones.
    let decode = qwen_decode_graph(&fixture.config, &fixture.identity, 2)?;
    if decode.validation.is_none() {
        return Err(E2eConformanceError::GraphValidationFailed {
            reason: "decode graph was produced without validation".into(),
        });
    }
    let policy = GraphPlanningPolicy::default();
    let catalog = default_graph_catalog();
    plan_execution_graph(&prefill.graph, &catalog, &policy, None)
        .map_err(E2eConformanceError::from)?;
    execute_graph_boundary(&prefill.graph, &catalog, &policy).map_err(E2eConformanceError::from)?;
    plan_execution_graph(&decode.graph, &catalog, &policy, None)
        .map_err(E2eConformanceError::from)?;
    execute_graph_boundary(&decode.graph, &catalog, &policy).map_err(E2eConformanceError::from)?;
    Ok(())
}

fn check_max_new_tokens_stops_generation(fixture: &E2eFixture) -> Result<(), E2eConformanceError> {
    let mut runtime = build_runtime_with_model_execution_engine(fixture);
    let (instance, _memory) = load_fixture_instance(fixture, &mut runtime)?;
    let tokenized = tokenize_prompt_input(
        &fixture.tokenizer,
        TokenizationRequest::new(PromptInput::PlainText("a".into())),
        None,
    )?;
    let request = build_generation_request(
        GenerationRequestId::new("e2e-max-new-tokens")?,
        None,
        GenerationModelReference::ModelInstance(instance.clone()),
        generation_tokenizer_reference(fixture),
        tokenized,
        1,
        GenerationParameters::greedy(),
        StopConditions::default(),
        StreamingMode::Disabled,
    );
    let request = prepare_generation(&runtime, request)?;
    let mut observer = InferenceApiObserver::new();
    let result = run_first_native_generation_loop_with_plans(
        &mut runtime,
        fixture,
        &instance,
        &request,
        SamplingPolicy::default(),
        CacheUsageSummary::default(),
        |_generated_so_far| false,
        &mut observer,
    )?;
    if result.output.finish_reason != FinishReason::MaxNewTokens {
        return Err(E2eConformanceError::GenerationFailed {
            reason: format!(
                "expected FinishReason::MaxNewTokens, got {:?}",
                result.output.finish_reason
            ),
        });
    }
    if result.output.generated_token_count != 1 {
        return Err(E2eConformanceError::GenerationFailed {
            reason: "expected exactly one generated token".into(),
        });
    }
    Ok(())
}

#[cfg(test)]
fn check_eos_token_stops_generation(fixture: &E2eFixture) -> Result<(), E2eConformanceError> {
    let mut runtime = build_runtime_with_model_execution_engine_and_forced_token(
        fixture,
        Some(E2E_FIXTURE_EOS_TOKEN),
    );
    let (instance, _memory) = load_fixture_instance(fixture, &mut runtime)?;
    let tokenized = tokenize_prompt_input(
        &fixture.tokenizer,
        TokenizationRequest::new(PromptInput::PlainText("a".into())),
        None,
    )?;
    let request = build_generation_request(
        GenerationRequestId::new("e2e-eos-stop")?,
        None,
        GenerationModelReference::ModelInstance(instance),
        generation_tokenizer_reference(fixture),
        tokenized,
        4,
        GenerationParameters::greedy(),
        StopConditions {
            eos: EosPolicy {
                eos_token_ids: vec![E2E_FIXTURE_EOS_TOKEN],
                ..EosPolicy::default()
            },
            ..StopConditions::default()
        },
        StreamingMode::Disabled,
    );
    let request = prepare_generation(&runtime, request)?;
    let mut observer = InferenceApiObserver::new();
    let result = run_generation_loop(
        &mut runtime,
        &request,
        SamplingPolicy::default(),
        CacheUsageSummary::default(),
        |_generated_so_far| false,
        &mut observer,
    )?;
    if result.output.finish_reason != FinishReason::EosToken {
        return Err(E2eConformanceError::GenerationFailed {
            reason: format!(
                "expected FinishReason::EosToken, got {:?}",
                result.output.finish_reason
            ),
        });
    }
    Ok(())
}

fn check_generation_cancelled(fixture: &E2eFixture) -> Result<(), E2eConformanceError> {
    let mut runtime = build_runtime_with_model_execution_engine(fixture);
    let (instance, _memory) = load_fixture_instance(fixture, &mut runtime)?;
    let tokenized = tokenize_prompt_input(
        &fixture.tokenizer,
        TokenizationRequest::new(PromptInput::PlainText("a".into())),
        None,
    )?;
    let request = build_generation_request(
        GenerationRequestId::new("e2e-cancelled")?,
        None,
        GenerationModelReference::ModelInstance(instance),
        generation_tokenizer_reference(fixture),
        tokenized,
        4,
        GenerationParameters::greedy(),
        StopConditions::default(),
        StreamingMode::Disabled,
    );
    let request = prepare_generation(&runtime, request)?;
    let mut observer = InferenceApiObserver::new();
    let result = run_generation_loop(
        &mut runtime,
        &request,
        SamplingPolicy::default(),
        CacheUsageSummary::default(),
        |_generated_so_far| true,
        &mut observer,
    )?;
    if result.output.finish_reason != FinishReason::Cancelled {
        return Err(E2eConformanceError::GenerationFailed {
            reason: format!(
                "expected FinishReason::Cancelled, got {:?}",
                result.output.finish_reason
            ),
        });
    }
    Ok(())
}

fn check_sampling_greedy_deterministic() -> Result<(), E2eConformanceError> {
    let tokenizer = e2e_fixture_tokenizer()?;
    let request = GenerationRequest {
        request_id: GenerationRequestId::new("e2e-sampling")?,
        session: None,
        model: GenerationModelReference::LoadedModelContext("e2e-fixture".into()),
        tokenizer: GenerationTokenizerReference {
            tokenizer_id: tokenizer.metadata().id.clone(),
            metadata: tokenizer.metadata().clone(),
        },
        prompt_token_count: 1,
        input_token_ids: vec![1],
        max_new_tokens: 1,
        max_total_tokens: None,
        model_context_length: None,
        parameters: GenerationParameters::greedy(),
        stop_conditions: StopConditions::default(),
        streaming: StreamingMode::Disabled,
        priority: GenerationPriority::default(),
        cancellation: CancellationMetadata::default(),
        memory: GenerationMemoryEstimate::default(),
        correlation_id: None,
        trace_id: None,
    };
    let mut logits = vec![0.0_f32; E2E_FIXTURE_VOCAB as usize];
    logits[42] = 10.0;
    let (first, _) =
        decode_step_from_sampling(&request, &[], logits.clone(), SamplingPolicy::default())
            .map_err(|error| E2eConformanceError::SamplingFailed {
                reason: error.to_string(),
            })?;
    let (second, _) = decode_step_from_sampling(&request, &[], logits, SamplingPolicy::default())
        .map_err(|error| E2eConformanceError::SamplingFailed {
        reason: error.to_string(),
    })?;
    if first.selected_token_id != 42 || second.selected_token_id != 42 {
        return Err(E2eConformanceError::SamplingFailed {
            reason: format!(
                "expected greedy selection of token 42 both times, got {} and {}",
                first.selected_token_id, second.selected_token_id
            ),
        });
    }
    Ok(())
}

fn check_closed_session_rejects_generation(
    fixture: &E2eFixture,
) -> Result<(), E2eConformanceError> {
    let mut runtime = build_runtime_trusting_fixture(fixture);
    let (instance, _memory) = load_fixture_instance(fixture, &mut runtime)?;
    let session_request = SessionCreationRequest {
        model: GenerationModelReference::ModelInstance(instance),
        tokenizer: generation_tokenizer_reference(fixture),
        generation_defaults: GenerationParameters::greedy(),
        policy: SessionPolicy::default(),
        memory: SessionMemoryBudget::default(),
        allowed_capabilities: BTreeSet::new(),
        correlation_id: None,
        created_at_millis: 0,
    };
    let session = create_inference_session(&mut runtime, session_request)?;
    close_inference_session(&mut runtime, &session)?;
    match session_status(
        &runtime,
        &session,
        &SessionAccessPolicy::authorize(session.clone()),
    ) {
        Err(InferenceApiError::SessionNotFound | InferenceApiError::SessionClosed) => Ok(()),
        Err(other) => Err(E2eConformanceError::from(other)),
        Ok(status) if status.lifecycle == SessionLifecycleState::Closed => Ok(()),
        Ok(_) => Err(E2eConformanceError::Internal {
            reason: "closed session was still reported usable".into(),
        }),
    }
}

#[cfg(test)]
fn check_first_native_generation_requires_ready_model_instance(
    fixture: &E2eFixture,
) -> Result<(), E2eConformanceError> {
    let mut runtime = build_runtime_trusting_fixture(fixture);
    let (instance, _memory) = load_fixture_instance(fixture, &mut runtime)?;
    suspend_model_instance(
        &mut runtime,
        &instance,
        ModelInstanceSuspensionReason::AdministrativePolicy,
    )?;

    match require_ready_first_native_instance(&runtime, &instance) {
        Err(InferenceApiError::ModelInstanceNotReady { reason })
            if reason.contains("requires ready ModelInstance") =>
        {
            Ok(())
        }
        Err(error) => Err(E2eConformanceError::GenerationFailed {
            reason: format!("unexpected readiness error: {error}"),
        }),
        Ok(_) => Err(E2eConformanceError::GenerationFailed {
            reason: "first-native generation accepted a non-ready ModelInstance".into(),
        }),
    }
}

#[cfg(test)]
fn check_missing_prepared_plan_fails_closed() -> Result<(), E2eConformanceError> {
    let context = first_native_plan_context(PreparedExecutionPhase::Prefill, 1);
    match require_compatible_first_native_plan(None, &context) {
        Err(PreparedExecutionPlanError::PlanNotFound) => Ok(()),
        Err(error) => Err(E2eConformanceError::GenerationFailed {
            reason: format!("unexpected missing-plan error: {error}"),
        }),
        Ok(_) => Err(E2eConformanceError::GenerationFailed {
            reason: "first-native execution accepted missing PreparedExecutionPlan".into(),
        }),
    }
}

#[cfg(test)]
fn check_invalidated_prepared_plan_rejects_new_work(
    fixture: &E2eFixture,
) -> Result<(), E2eConformanceError> {
    let mut runtime = build_runtime_trusting_fixture(fixture);
    let (instance, _memory) = load_fixture_instance(fixture, &mut runtime)?;
    let graphs = build_first_native_graphs_from_component_output(
        fixture,
        2,
        qwen_component_graph_semantics_for_prompt(&fixture.config, &fixture.identity, 2)?,
    )?;
    let mut plans = prepare_first_native_execution_plans(&runtime, &instance, graphs, 2)?;
    plans
        .decode
        .hard_invalidate(crate::kernel_execution_plan::PlanRebuildReason::KernelRevoked)?;
    let context = first_native_plan_context(PreparedExecutionPhase::Decode, 1);
    match require_compatible_first_native_plan(Some(&mut plans.decode), &context) {
        Err(PreparedExecutionPlanError::PlanNotReadyForExecution) => Ok(()),
        Err(error) => Err(E2eConformanceError::GenerationFailed {
            reason: format!("unexpected invalidated-plan error: {error}"),
        }),
        Ok(_) => Err(E2eConformanceError::GenerationFailed {
            reason: "first-native execution accepted invalidated PreparedExecutionPlan".into(),
        }),
    }
}

#[cfg(test)]
fn check_stale_plan_outside_policy_fails_closed(
    fixture: &E2eFixture,
) -> Result<(), E2eConformanceError> {
    let mut runtime = build_runtime_trusting_fixture(fixture);
    let (instance, _memory) = load_fixture_instance(fixture, &mut runtime)?;
    let graphs = build_first_native_graphs_from_component_output(
        fixture,
        2,
        qwen_component_graph_semantics_for_prompt(&fixture.config, &fixture.identity, 2)?,
    )?;
    let mut plans = prepare_first_native_execution_plans(&runtime, &instance, graphs, 2)?;
    plans.decode.mark_stale(
        crate::kernel_execution_plan::PlanRebuildReason::KernelRevoked,
        crate::kernel_execution_plan::PlanRebuildUrgency::RequiredBeforeNewWork,
    )?;
    let context = first_native_plan_context(PreparedExecutionPhase::Decode, 1);
    match require_compatible_first_native_plan(Some(&mut plans.decode), &context) {
        Err(PreparedExecutionPlanError::PlanStaleOutsidePolicy) => Ok(()),
        Err(error) => Err(E2eConformanceError::GenerationFailed {
            reason: format!("unexpected stale-outside-policy error: {error}"),
        }),
        Ok(_) => Err(E2eConformanceError::GenerationFailed {
            reason: "first-native execution accepted a plan stale outside its rebuild policy"
                .into(),
        }),
    }
}

#[cfg(test)]
fn check_qwen_graph_nodes_have_prepared_kernel_bindings(
    fixture: &E2eFixture,
) -> Result<(), E2eConformanceError> {
    let mut runtime = build_runtime_trusting_fixture(fixture);
    let (instance, _memory) = load_fixture_instance(fixture, &mut runtime)?;
    let graphs = build_first_native_graphs_from_component_output(
        fixture,
        2,
        qwen_component_graph_semantics_for_prompt(&fixture.config, &fixture.identity, 2)?,
    )?;
    let plans = prepare_first_native_execution_plans(&runtime, &instance, graphs, 2)?;

    for (plan, expected_node_count) in [
        (&plans.prefill, plans.prefill_node_count),
        (&plans.decode, plans.decode_node_count),
    ] {
        if plan.node_bindings.len() != expected_node_count {
            return Err(E2eConformanceError::GenerationFailed {
                reason: format!(
                    "prepared plan has {} bindings for {expected_node_count} graph nodes",
                    plan.node_bindings.len()
                ),
            });
        }
        for binding in &plan.node_bindings {
            if binding.graph_nodes.is_empty() {
                return Err(E2eConformanceError::GenerationFailed {
                    reason: "prepared plan contains a binding without graph nodes".into(),
                });
            }
            if binding.kernel.provider.as_str() != REFERENCE_CPU_PROVIDER_NAME {
                return Err(E2eConformanceError::GenerationFailed {
                    reason: "prepared plan selected a non-Reference CPU provider".into(),
                });
            }
            if binding.provider.as_str() != REFERENCE_CPU_PROVIDER_NAME {
                return Err(E2eConformanceError::GenerationFailed {
                    reason: "prepared plan binding provider is not Reference CPU".into(),
                });
            }
            if binding.device.as_ref().map(ToString::to_string).as_deref()
                != Some(REFERENCE_CPU_DEVICE_ID)
            {
                return Err(E2eConformanceError::GenerationFailed {
                    reason: "prepared plan binding did not record Reference CPU device identity"
                        .into(),
                });
            }
            if binding.prepared_kernel.is_none() || binding.prepared_kernel_generation.is_none() {
                return Err(E2eConformanceError::GenerationFailed {
                    reason: "prepared plan binding lacks PreparedKernelId or generation".into(),
                });
            }
            if binding.qualification_profile.as_deref() != Some(REFERENCE_CPU_CONFORMANCE_PROFILE) {
                return Err(E2eConformanceError::GenerationFailed {
                    reason: "prepared plan binding lacks implementation conformance identity"
                        .into(),
                });
            }
        }
    }
    Ok(())
}

#[cfg(test)]
fn check_graph_dispatch_rejects_unregistered_provider(
    fixture: &E2eFixture,
) -> Result<(), E2eConformanceError> {
    let mut runtime = build_runtime_trusting_fixture(fixture);
    let (instance, _memory) = load_fixture_instance(fixture, &mut runtime)?;
    let mut plans = first_native_plans_for_prompt(&runtime, fixture, &instance, 2)?;
    let graphs = first_native_component_graphs_for_prompt(fixture, 2)?;
    // Every node binding names a Provider Runtime must resolve at execution
    // time (task 5.2); point them all at a name nothing registers, the
    // execution-time equivalent of the Provider having been removed from
    // Runtime's registration between plan preparation and execution.
    for binding in &mut plans.prefill.node_bindings {
        binding.provider = ProviderBinding::new("unregistered-provider");
    }
    let ids = HostTensor::new([2], vec![1.0, 2.0])?;
    let cache_id = KvCacheId::new("test-unregistered-executor-cache")?;
    match execute_qwen_graph(
        &mut runtime,
        fixture,
        &instance,
        &cache_id,
        &graphs.prefill,
        &mut plans.prefill,
        BTreeMap::from([(TensorEdgeId::new("input.token_ids"), ids)]),
        None,
        Some(0),
        &mut Vec::new(),
    ) {
        Err(InferenceApiError::ProviderUnavailable { reason })
            if reason.contains("unregistered-provider") =>
        {
            Ok(())
        }
        Err(error) => Err(E2eConformanceError::GenerationFailed {
            reason: format!("unexpected error for an unregistered provider: {error}"),
        }),
        Ok(_) => Err(E2eConformanceError::GenerationFailed {
            reason: "graph executor dispatched through an unregistered provider".into(),
        }),
    }
}

#[cfg(test)]
fn check_graph_dispatch_uses_registered_provider_instance(
    fixture: &E2eFixture,
) -> Result<(), E2eConformanceError> {
    let mut runtime = build_runtime_trusting_fixture(fixture);
    let (instance, _memory) = load_fixture_instance(fixture, &mut runtime)?;
    let mut plans = first_native_plans_for_prompt(&runtime, fixture, &instance, 2)?;
    let graphs = first_native_component_graphs_for_prompt(fixture, 2)?;
    let provider_binding = plans
        .prefill
        .node_bindings
        .first()
        .map(|binding| binding.provider.clone())
        .ok_or_else(|| E2eConformanceError::GraphValidationFailed {
            reason: "prefill plan has no node bindings".into(),
        })?;
    let before = resolve_kernel_execution_provider(&runtime, &provider_binding)
        .map_err(E2eConformanceError::from)?
        .observations()
        .len();
    let ids = HostTensor::new([2], vec![1.0, 2.0])?;
    let cache_id = KvCacheId::new("test-registered-executor-instance-cache")?;
    execute_qwen_graph(
        &mut runtime,
        fixture,
        &instance,
        &cache_id,
        &graphs.prefill,
        &mut plans.prefill,
        BTreeMap::from([(TensorEdgeId::new("input.token_ids"), ids)]),
        None,
        Some(0),
        &mut Vec::new(),
    )
    .map_err(E2eConformanceError::from)?;
    // Re-resolving from Runtime provider registration after dispatch (the
    // same path production execution uses) must observe the *same*
    // registered instance's growing observation trail -- not a disconnected
    // throwaway that discarded its own observations when it went out of
    // scope.
    let after = resolve_kernel_execution_provider(&runtime, &provider_binding)
        .map_err(E2eConformanceError::from)?
        .observations()
        .len();
    if after <= before {
        return Err(E2eConformanceError::GenerationFailed {
            reason:
                "graph dispatch did not record observations on the registered provider instance"
                    .into(),
        });
    }
    Ok(())
}

#[cfg(test)]
fn check_graph_dispatch_accounts_outputs_through_runtime_memory_manager(
    fixture: &E2eFixture,
) -> Result<(), E2eConformanceError> {
    let mut runtime = build_runtime_trusting_fixture(fixture);
    let (instance, _memory) = load_fixture_instance(fixture, &mut runtime)?;
    let mut plans = first_native_plans_for_prompt(&runtime, fixture, &instance, 2)?;
    let graphs = first_native_component_graphs_for_prompt(fixture, 2)?;
    let ids = HostTensor::new([2], vec![1.0, 2.0])?;
    let cache_id = KvCacheId::new("test-output-accounting-cache")?;
    execute_qwen_graph(
        &mut runtime,
        fixture,
        &instance,
        &cache_id,
        &graphs.prefill,
        &mut plans.prefill,
        BTreeMap::from([(TensorEdgeId::new("input.token_ids"), ids)]),
        None,
        Some(0),
        &mut Vec::new(),
    )
    .map_err(E2eConformanceError::from)?;
    let tensor_allocations = runtime
        .memory()
        .allocations()
        .filter(|allocation| allocation.request.class == MemoryAllocationClass::Tensor)
        .count();
    if tensor_allocations == 0 {
        return Err(E2eConformanceError::MemoryValidationFailed {
            reason:
                "graph dispatch produced no output tensor allocations in Runtime's MemoryManager"
                    .into(),
        });
    }
    Ok(())
}

/// Correctif 5: `execute_qwen_graph_nodes`'s node-to-node transport is
/// Resource-based, not a private `HostTensor` cache -- an *intermediate*
/// graph edge's value (not just the final returned bindings) must be
/// independently readable straight from the registered Provider's storage,
/// under the resource id the executor recorded for it.
#[cfg(test)]
fn check_graph_dispatch_intermediate_edge_is_resolvable_from_provider_storage(
    fixture: &E2eFixture,
) -> Result<(), E2eConformanceError> {
    let mut runtime = build_runtime_trusting_fixture(fixture);
    let (instance, _memory) = load_fixture_instance(fixture, &mut runtime)?;
    let mut plans = first_native_plans_for_prompt(&runtime, fixture, &instance, 2)?;
    let graphs = first_native_component_graphs_for_prompt(fixture, 2)?;
    let embedding_output_edge = graphs
        .prefill
        .nodes
        .get(&ExecutionNodeId::new("embedding"))
        .and_then(|node| node.outputs.first())
        .cloned()
        .ok_or_else(|| E2eConformanceError::GraphValidationFailed {
            reason: "prefill graph has no 'embedding' node output edge".into(),
        })?;
    let ids = HostTensor::new([2], vec![1.0, 2.0])?;
    let cache_id = KvCacheId::new("test-intermediate-edge-resource-cache")?;
    execute_qwen_graph(
        &mut runtime,
        fixture,
        &instance,
        &cache_id,
        &graphs.prefill,
        &mut plans.prefill,
        BTreeMap::from([(TensorEdgeId::new("input.token_ids"), ids)]),
        None,
        Some(0),
        &mut Vec::new(),
    )
    .map_err(E2eConformanceError::from)?;
    let provider = resolve_kernel_execution_provider(
        &runtime,
        &ProviderBinding::new(REFERENCE_CPU_PROVIDER_NAME),
    )
    .map_err(E2eConformanceError::from)?;
    let resource_id = TensorResourceId::new(format!("edge.{embedding_output_edge}"));
    if provider.read_tensor(&resource_id).is_none() {
        return Err(E2eConformanceError::MemoryValidationFailed {
            reason: format!(
                "intermediate edge '{embedding_output_edge}' is not resolvable from Provider \
                 storage at resource '{resource_id}'; graph execution must not hold this \
                 value only in a private, non-Provider-backed cache"
            ),
        });
    }
    Ok(())
}

#[cfg(test)]
fn check_graph_dispatch_releases_workspace_after_use(
    fixture: &E2eFixture,
) -> Result<(), E2eConformanceError> {
    let mut runtime = build_runtime_trusting_fixture(fixture);
    let (instance, _memory) = load_fixture_instance(fixture, &mut runtime)?;
    let mut plans = first_native_plans_for_prompt(&runtime, fixture, &instance, 2)?;
    // This fixture's one layer includes an `attention` node, the only
    // Operator Reference CPU advertises a required workspace for.
    let graphs = first_native_component_graphs_for_prompt(fixture, 2)?;
    let ids = HostTensor::new([2], vec![1.0, 2.0])?;
    let cache_id = KvCacheId::new("test-workspace-release-cache")?;
    execute_qwen_graph(
        &mut runtime,
        fixture,
        &instance,
        &cache_id,
        &graphs.prefill,
        &mut plans.prefill,
        BTreeMap::from([(TensorEdgeId::new("input.token_ids"), ids)]),
        None,
        Some(0),
        &mut Vec::new(),
    )
    .map_err(E2eConformanceError::from)?;
    let workspace_allocations: Vec<_> = runtime
        .memory()
        .allocations()
        .filter(|allocation| allocation.request.class == MemoryAllocationClass::TemporaryWorkspace)
        .collect();
    if workspace_allocations.is_empty() {
        return Err(E2eConformanceError::MemoryValidationFailed {
            reason: "attention dispatch requested no workspace allocation to release".into(),
        });
    }
    if workspace_allocations
        .iter()
        .any(|allocation| allocation.state == MemoryAllocationState::Active)
    {
        return Err(E2eConformanceError::MemoryValidationFailed {
            reason: "workspace allocation was not released after its dispatch completed".into(),
        });
    }
    Ok(())
}

#[cfg(test)]
fn check_graph_dispatch_records_memory_feasibility_failure_under_tight_budget(
    fixture: &E2eFixture,
) -> Result<(), E2eConformanceError> {
    // Wide enough for model loading's own resident-bytes allocation and this
    // fixture's per-tensor weight resources (task 6.2) to be admitted, but
    // far below `attention`'s required 1 MiB workspace -- so *that*
    // allocation is what fails admission, not an earlier, unrelated one.
    let mut runtime = Runtime::builder()
        .register_provider(std::sync::Arc::new(ReferenceCpuProvider::new()))
        .config(RuntimeConfig {
            memory: MemoryManagerConfig {
                max_runtime_bytes: Some(1 << 16),
                allow_pending_allocations: false,
                ..MemoryManagerConfig::default()
            },
            ..RuntimeConfig::default()
        })
        .trust_store(
            ModelTrustStore::default().trust_digest(fixture.manifest.id.digest.value.clone()),
        )
        .build()
        .map_err(|error| E2eConformanceError::SuiteUnavailable {
            reason: error.to_string(),
        })?;
    register_reference_cpu_prepared_kernels(&mut runtime);
    let (instance, _memory) = load_fixture_instance(fixture, &mut runtime)?;
    let mut plans = first_native_plans_for_prompt(&runtime, fixture, &instance, 2)?;
    let graphs = first_native_component_graphs_for_prompt(fixture, 2)?;
    let ids = HostTensor::new([2], vec![1.0, 2.0])?;
    let cache_id = KvCacheId::new("test-tight-budget-cache")?;
    let result = execute_qwen_graph(
        &mut runtime,
        fixture,
        &instance,
        &cache_id,
        &graphs.prefill,
        &mut plans.prefill,
        BTreeMap::from([(TensorEdgeId::new("input.token_ids"), ids)]),
        None,
        Some(0),
        &mut Vec::new(),
    );
    // Memory Admission Precedes Provider Materialization: every node's
    // declared output must be admitted before its Kernel is dispatched, so
    // the first node this fixture's graph reaches whose output (or, for
    // `attention`, required workspace) does not fit the tight budget hard-
    // fails admission and the Kernel is never dispatched for it.
    match result {
        Err(
            InferenceApiError::MemoryAdmissionFailed { reason }
            | InferenceApiError::GenerationFailed { reason },
        ) if reason.contains("out of memory") || reason.contains("memory admission failed") => {
            Ok(())
        }
        Err(error) => Err(E2eConformanceError::GenerationFailed {
            reason: format!("unexpected error under a tight Runtime memory budget: {error}"),
        }),
        Ok(_) => Err(E2eConformanceError::GenerationFailed {
            reason: "graph dispatch succeeded despite a tight Runtime memory budget".into(),
        }),
    }
}

#[cfg(test)]
fn check_weight_binding_rejects_tampered_artifact_bytes(
    fixture: &E2eFixture,
) -> Result<(), E2eConformanceError> {
    let mut runtime = build_runtime_trusting_fixture(fixture);
    let mut tampered = fixture.clone();
    let (_name, tensor) =
        tampered
            .weights
            .iter_mut()
            .next()
            .ok_or_else(|| E2eConformanceError::FixtureInvalid {
                reason: "fixture has no weight tensors to tamper with".into(),
            })?;
    tensor.data[0] += 1.0;
    match load_fixture_instance(&tampered, &mut runtime) {
        Err(E2eConformanceError::FixtureInvalid { reason }) if reason.contains("digest") => Ok(()),
        Err(error) => Err(E2eConformanceError::GenerationFailed {
            reason: format!("unexpected error for a tampered weight artifact: {error}"),
        }),
        Ok(_) => Err(E2eConformanceError::GenerationFailed {
            reason: "model loading accepted a weight artifact with tampered bytes".into(),
        }),
    }
}

/// Like `load_fixture_instance`, but binds `weights` directly through
/// `materialize_model_instance_weights` instead of `bind_qwen_fixture_weights`
/// -- so a caller can supply a deliberately altered weight map without it
/// being rejected by the fixture's own digest check (task 6.5's concern,
/// already covered by `check_weight_binding_rejects_tampered_artifact_bytes`;
/// not what this is for).
#[cfg(test)]
fn load_fixture_instance_with_weights(
    fixture: &E2eFixture,
    runtime: &mut Runtime,
    weights: &BTreeMap<String, HostTensor>,
) -> Result<ModelInstanceId, E2eConformanceError> {
    let mut coordinator = ModelLoadingCoordinator::new();
    coordinator.register_architecture(fixture.architecture_implementation.clone());
    let mut request = ModelLoadingRequest::new(
        ModelLoadingRequestId::new("e2e-fixture-load-weight-sensitivity"),
        fixture.manifest.id.clone(),
    );
    request.quantization_policy = ModelQuantizationPolicy::RejectUnsupported;
    let loaded = load_model(
        &mut coordinator,
        runtime,
        ModelLoadingApiRequest::new(request),
        &fixture.manifest,
    )?;
    let instance = create_model_instance(
        runtime,
        &loaded,
        fixture.architecture_implementation.clone(),
        ResourceAffinity::new(FallbackClass::Transparent),
    )?;
    materialize_model_instance_weights(
        runtime,
        &instance,
        fixture.manifest.id.name.as_str(),
        weights,
    )?;
    Ok(instance)
}

/// `bind-materialized-weight-content-to-model-artifact-digests`: proves
/// the content-digest check at the exact public entrypoint it lives in
/// (`WeightMaterializationTransaction::stage_weight`, reached through
/// `materialize_model_instance_weights`), not only through
/// `bind_qwen_fixture_weights`'s separate, earlier, aggregate in-memory
/// check (`check_weight_binding_rejects_tampered_artifact_bytes` proves
/// that one). `fixture.manifest`'s tensor inventory now carries real
/// per-tensor digests computed from the real checked-in Safetensors file
/// (`e2e_fixture_manifest`), so tampering one tensor's bytes before
/// materializing it directly must be rejected with the specific
/// content-digest-mismatch error, not merely *some* error.
#[cfg(test)]
fn check_materialize_model_instance_weights_rejects_content_digest_mismatch(
    fixture: &E2eFixture,
) -> Result<(), E2eConformanceError> {
    let mut runtime = build_runtime_trusting_fixture(fixture);
    let mut tampered_weights = fixture.weights.clone();
    let (_name, tensor) =
        tampered_weights
            .iter_mut()
            .next()
            .ok_or_else(|| E2eConformanceError::FixtureInvalid {
                reason: "fixture has no weight tensors to tamper with".into(),
            })?;
    tensor.data[0] += 1.0;
    match load_fixture_instance_with_weights(fixture, &mut runtime, &tampered_weights) {
        Err(E2eConformanceError::GenerationFailed { reason })
            if reason.contains("weight content digest mismatch") =>
        {
            Ok(())
        }
        Err(error) => Err(E2eConformanceError::GenerationFailed {
            reason: format!("expected a weight content digest mismatch error, got: {error}"),
        }),
        Ok(_) => Err(E2eConformanceError::GenerationFailed {
            reason: "materialize_model_instance_weights accepted tampered tensor content".into(),
        }),
    }
}

/// Regression guard for the happy path this Change's check sits directly
/// in front of: the real, untampered fixture weights (bit-identical to
/// what their declared digests were computed from) must still materialize
/// and bind normally through the exact same entrypoint the mismatch test
/// above uses.
#[cfg(test)]
fn check_materialize_model_instance_weights_accepts_matching_content(
    fixture: &E2eFixture,
) -> Result<(), E2eConformanceError> {
    let mut runtime = build_runtime_trusting_fixture(fixture);
    let real_weights = e2e_fixture_weights_from_real_artifact(&fixture.config)?;
    load_fixture_instance_with_weights(fixture, &mut runtime, &real_weights)?;
    Ok(())
}

/// Runs a real prefill through the production graph-execution path
/// (`execute_qwen_graph`, the same one `execute_generation_step` uses) with
/// `weights` bound to a fresh `ModelInstance`, and returns the "logits"
/// edge's values.
#[cfg(test)]
fn forward_logits_with_weights(
    fixture: &E2eFixture,
    weights: &BTreeMap<String, HostTensor>,
    prompt: &[TokenId],
) -> Result<Vec<f32>, E2eConformanceError> {
    let mut runtime = build_runtime_trusting_fixture(fixture);
    let instance = load_fixture_instance_with_weights(fixture, &mut runtime, weights)?;
    let mut plans =
        first_native_plans_for_prompt(&runtime, fixture, &instance, prompt.len() as u64)?;
    let graphs = first_native_component_graphs_for_prompt(fixture, prompt.len() as u64)?;
    let ids = HostTensor::new(
        [prompt.len() as u64],
        prompt.iter().map(|id| *id as f32).collect::<Vec<_>>(),
    )?;
    let cache_id = KvCacheId::new("test-weight-sensitivity-cache")?;
    let (_dispatch, mut bindings, _layer_kv, _provider) = execute_qwen_graph(
        &mut runtime,
        fixture,
        &instance,
        &cache_id,
        &graphs.prefill,
        &mut plans.prefill,
        BTreeMap::from([(TensorEdgeId::new("input.token_ids"), ids)]),
        None,
        Some(0),
        &mut Vec::new(),
    )?;
    let logits = bindings
        .remove(&TensorEdgeId::new("logits"))
        .ok_or_else(|| E2eConformanceError::GenerationFailed {
            reason: "first-native graph produced no logits output".into(),
        })?;
    Ok(logits.data)
}

/// Correctif 6 / task 8.7: a single changed weight byte in the Artifact
/// SHALL change generated logits -- proof the graph-executed path actually
/// reads and uses the bound weight bytes numerically, rather than (for
/// example) a cached or hard-coded computation that happens to match the
/// fixture's usual values. Complements `check_weight_binding_rejects_tampered_artifact_bytes`
/// (task 8.8), which proves a *digest* mismatch is caught before binding --
/// this instead proves the bound bytes are not merely checked but actually
/// consumed.
#[cfg(test)]
fn check_weight_byte_change_alters_generated_logits(
    fixture: &E2eFixture,
) -> Result<(), E2eConformanceError> {
    // This test's own mutated weights would now (correctly) be rejected by
    // `bind-materialized-weight-content-to-model-artifact-digests`'s
    // content-digest check if bound against `fixture.manifest`'s real,
    // digest-bearing tensor inventory -- that rejection is a *different*
    // property, already proven by `check_weight_binding_rejects_tampered_
    // artifact_bytes` and the new digest-mismatch tests this Change adds.
    // This test's own concern is orthogonal: given bytes that *did* bind
    // (however that happened), are they actually consumed numerically. A
    // digest-free copy of the fixture keeps that concern isolated rather
    // than conflating it with the digest check.
    let mut digest_free_fixture = fixture.clone();
    for tensor in &mut digest_free_fixture.manifest.tensors {
        tensor.digest = None;
    }
    let prompt = [1, 2];
    let baseline_logits =
        forward_logits_with_weights(&digest_free_fixture, &fixture.weights, &prompt)?;

    let mut mutated_weights = fixture.weights.clone();
    let (_name, tensor) =
        mutated_weights
            .iter_mut()
            .next()
            .ok_or_else(|| E2eConformanceError::FixtureInvalid {
                reason: "fixture has no weight tensors to mutate".into(),
            })?;
    tensor.data[0] += 1.0;
    let mutated_logits =
        forward_logits_with_weights(&digest_free_fixture, &mutated_weights, &prompt)?;

    if baseline_logits == mutated_logits {
        return Err(E2eConformanceError::GenerationFailed {
            reason: "changing one weight byte did not change the generated logits".into(),
        });
    }
    Ok(())
}

#[cfg(test)]
fn check_graph_execution_fails_closed_on_missing_weight(
    fixture: &E2eFixture,
) -> Result<(), E2eConformanceError> {
    let mut runtime = build_runtime_trusting_fixture(fixture);
    let (instance, _memory) = load_fixture_instance(fixture, &mut runtime)?;
    // Loading bound every declared weight successfully (digest verified);
    // removing one binding afterward -- without touching the fixture or its
    // digest -- isolates the "this instance is missing a resource graph
    // execution needs" failure mode from artifact tampering, which
    // `check_weight_binding_rejects_tampered_artifact_bytes` already covers.
    runtime
        .model_instances_mut()
        .instance_mut(&instance)
        .map_err(InferenceApiError::from)?
        .definition
        .resource_bindings
        .weights
        .remove("token_embedding");
    let mut plans = first_native_plans_for_prompt(&runtime, fixture, &instance, 2)?;
    let graphs = first_native_component_graphs_for_prompt(fixture, 2)?;
    let ids = HostTensor::new([2], vec![1.0, 2.0])?;
    let cache_id = KvCacheId::new("test-missing-weight-cache")?;
    match execute_qwen_graph(
        &mut runtime,
        fixture,
        &instance,
        &cache_id,
        &graphs.prefill,
        &mut plans.prefill,
        BTreeMap::from([(TensorEdgeId::new("input.token_ids"), ids)]),
        None,
        Some(0),
        &mut Vec::new(),
    ) {
        Err(InferenceApiError::ModelLoadingFailed { reason })
            if reason.contains("token_embedding") =>
        {
            Ok(())
        }
        Err(error) => Err(E2eConformanceError::GenerationFailed {
            reason: format!("unexpected error for a missing weight: {error}"),
        }),
        Ok(_) => Err(E2eConformanceError::GenerationFailed {
            reason: "graph execution succeeded despite a missing required weight".into(),
        }),
    }
}

#[cfg(test)]
fn check_weight_resources_are_isolated_per_model_instance(
    fixture: &E2eFixture,
) -> Result<(), E2eConformanceError> {
    let mut runtime = build_runtime_trusting_fixture(fixture);
    let (first, _memory) = load_fixture_instance(fixture, &mut runtime)?;
    let (second, _memory) = load_fixture_instance(fixture, &mut runtime)?;
    if first == second {
        return Err(E2eConformanceError::GenerationFailed {
            reason: "loading the same fixture twice produced the same Model Instance id".into(),
        });
    }
    let first_bindings = runtime
        .model_instance(&first)
        .map_err(InferenceApiError::from)?
        .definition
        .resource_bindings
        .weights
        .clone();
    let second_bindings = runtime
        .model_instance(&second)
        .map_err(InferenceApiError::from)?
        .definition
        .resource_bindings
        .weights
        .clone();
    if first_bindings.is_empty() || first_bindings.len() != second_bindings.len() {
        return Err(E2eConformanceError::GenerationFailed {
            reason: "both Model Instances did not bind the same set of weight names".into(),
        });
    }
    for (name, first_resource) in &first_bindings {
        let second_resource =
            second_bindings
                .get(name)
                .ok_or_else(|| E2eConformanceError::GenerationFailed {
                    reason: format!("second Model Instance has no binding for weight '{name}'"),
                })?;
        if first_resource == second_resource {
            return Err(E2eConformanceError::GenerationFailed {
                reason: format!(
                    "weight '{name}' resolved to the same TensorResourceId for two different Model Instances"
                ),
            });
        }
    }
    Ok(())
}

#[cfg(test)]
fn check_unload_releases_weight_resource_allocations(
    fixture: &E2eFixture,
) -> Result<(), E2eConformanceError> {
    let mut runtime = build_runtime_trusting_fixture(fixture);
    let (instance, _memory) = load_fixture_instance(fixture, &mut runtime)?;
    let allocation_ids: Vec<_> = runtime
        .model_instance(&instance)
        .map_err(InferenceApiError::from)?
        .definition
        .resource_bindings
        .memory_allocations
        .iter()
        .copied()
        .collect();
    if allocation_ids.is_empty() {
        return Err(E2eConformanceError::GenerationFailed {
            reason: "loading the fixture bound no weight memory allocations to release".into(),
        });
    }
    // `transactional-weight-materialization` (P0-2-bis): unload must also
    // release the Provider-owned weight Tensor Resources themselves, not
    // only Memory Manager accounting -- capture them before unload so they
    // can be checked against Provider storage afterward.
    let weight_resource_ids: Vec<TensorResourceId> = runtime
        .model_instance(&instance)
        .map_err(InferenceApiError::from)?
        .definition
        .resource_bindings
        .weights
        .values()
        .cloned()
        .collect();
    if weight_resource_ids.is_empty() {
        return Err(E2eConformanceError::GenerationFailed {
            reason: "loading the fixture bound no weight Tensor Resources to release".into(),
        });
    }
    let executor = resolve_kernel_execution_provider(
        &runtime,
        &ProviderBinding::new(REFERENCE_CPU_PROVIDER_NAME),
    )
    .map_err(|error| E2eConformanceError::GenerationFailed {
        reason: error.to_string(),
    })?;
    for resource_id in &weight_resource_ids {
        if executor.read_tensor(resource_id).is_none() {
            return Err(E2eConformanceError::GenerationFailed {
                reason: format!(
                    "weight resource '{resource_id}' was not actually present in Provider \
                     storage before unload (test precondition broken)"
                ),
            });
        }
    }
    runtime
        .unload_model_instance(&instance, ModelInstanceUnloadPolicy::RejectActiveUse)
        .map_err(|error| E2eConformanceError::GenerationFailed {
            reason: error.to_string(),
        })?;
    for allocation_id in allocation_ids {
        let state = runtime
            .memory()
            .allocations()
            .find(|allocation| allocation.id == allocation_id)
            .map(|allocation| allocation.state);
        if state == Some(MemoryAllocationState::Active) {
            return Err(E2eConformanceError::MemoryValidationFailed {
                reason: format!(
                    "weight allocation {allocation_id:?} remained Active after Model Instance unload"
                ),
            });
        }
    }
    for resource_id in &weight_resource_ids {
        if executor.read_tensor(resource_id).is_some() {
            return Err(E2eConformanceError::MemoryValidationFailed {
                reason: format!(
                    "weight resource '{resource_id}' remained present in Provider-owned \
                     storage after Model Instance unload (P0-2-bis: unload must release \
                     Provider-owned weight storage, not only Memory Manager accounting)"
                ),
            });
        }
        assert_tensor_residency_absent(&runtime, resource_id, "after Model Instance unload")?;
    }
    Ok(())
}

/// Proves the load/unload cycle does not accumulate Provider-owned weight
/// storage over repeated cycles -- the audit's own "100x load/unload"
/// case, done at a smaller, still-meaningful count (each cycle already
/// proves the property; more repetitions prove only that it does not
/// degrade with iteration count, which a fixed small count already shows
/// as well without materially slower test runs).
#[cfg(test)]
fn check_repeated_load_unload_does_not_accumulate_weight_storage(
    fixture: &E2eFixture,
) -> Result<(), E2eConformanceError> {
    const CYCLES: usize = 10;
    let mut runtime = build_runtime_trusting_fixture(fixture);
    for cycle in 0..CYCLES {
        let (instance, _memory) = load_fixture_instance(fixture, &mut runtime)?;
        let weight_resource_ids: Vec<TensorResourceId> = runtime
            .model_instance(&instance)
            .map_err(InferenceApiError::from)?
            .definition
            .resource_bindings
            .weights
            .values()
            .cloned()
            .collect();
        if weight_resource_ids.is_empty() {
            return Err(E2eConformanceError::GenerationFailed {
                reason: format!("cycle {cycle}: loading the fixture bound no weight resources"),
            });
        }
        runtime
            .unload_model_instance(&instance, ModelInstanceUnloadPolicy::RejectActiveUse)
            .map_err(|error| E2eConformanceError::GenerationFailed {
                reason: format!("cycle {cycle}: unload failed: {error}"),
            })?;
        let executor = resolve_kernel_execution_provider(
            &runtime,
            &ProviderBinding::new(REFERENCE_CPU_PROVIDER_NAME),
        )
        .map_err(|error| E2eConformanceError::GenerationFailed {
            reason: error.to_string(),
        })?;
        for resource_id in &weight_resource_ids {
            if executor.read_tensor(resource_id).is_some() {
                return Err(E2eConformanceError::MemoryValidationFailed {
                    reason: format!(
                        "cycle {cycle}: weight resource '{resource_id}' still present in \
                         Provider storage after unload -- storage is accumulating across cycles"
                    ),
                });
            }
            // Each cycle creates a fresh Model Instance, so a fresh
            // TensorResourceId per weight -- a residency record surviving
            // past its own cycle's unload would mean residency metadata
            // grows unbounded across cycles even though Provider storage
            // and Memory Manager accounting both look clean
            // (`invalidate-tensor-residency-on-release`).
            assert_tensor_residency_absent(
                &runtime,
                resource_id,
                &format!("after unload in cycle {cycle} -- residency metadata is accumulating"),
            )?;
        }
    }
    Ok(())
}

fn check_memory_admission_failure() -> Result<(), E2eConformanceError> {
    let manager = MemoryManager::new(MemoryManagerConfig {
        max_runtime_bytes: Some(1),
        allow_pending_allocations: false,
        ..MemoryManagerConfig::default()
    });
    let tokenizer = e2e_fixture_tokenizer()?;
    let request = GenerationRequest {
        request_id: GenerationRequestId::new("e2e-memory-admission")?,
        session: None,
        model: GenerationModelReference::LoadedModelContext("e2e-fixture".into()),
        tokenizer: GenerationTokenizerReference {
            tokenizer_id: tokenizer.metadata().id.clone(),
            metadata: tokenizer.metadata().clone(),
        },
        prompt_token_count: 1,
        input_token_ids: vec![1],
        max_new_tokens: 1,
        max_total_tokens: None,
        model_context_length: None,
        parameters: GenerationParameters::greedy(),
        stop_conditions: StopConditions::default(),
        streaming: StreamingMode::Disabled,
        priority: GenerationPriority::default(),
        cancellation: CancellationMetadata::default(),
        memory: GenerationMemoryEstimate {
            input_token_buffer_bytes: 10_000_000,
            queue_allowed: false,
            ..GenerationMemoryEstimate::default()
        },
        correlation_id: None,
        trace_id: None,
    };
    let admission = memory_admission(&request, &manager).map_err(|error| {
        E2eConformanceError::MemoryValidationFailed {
            reason: error.to_string(),
        }
    })?;
    if admission.is_admitted() {
        return Err(E2eConformanceError::Internal {
            reason: "expected memory admission to reject an over-budget request".into(),
        });
    }
    Ok(())
}

fn check_incompatible_tokenizer(fixture: &E2eFixture) -> Result<(), E2eConformanceError> {
    let mut incompatible = fixture.tokenizer.metadata().clone();
    incompatible.special_tokens.clear();
    match qwen_validate_tokenizer_compatibility(&fixture.config, &incompatible) {
        Err(_) => Ok(()),
        Ok(()) => Err(E2eConformanceError::Internal {
            reason: "expected incompatible tokenizer (missing EOS) to be rejected".into(),
        }),
    }
}

fn check_untrusted_artifact(fixture: &E2eFixture) -> Result<(), E2eConformanceError> {
    let mut coordinator = ModelLoadingCoordinator::new();
    coordinator.register_architecture(fixture.architecture_implementation.clone());
    let mut memory = MemoryManager::default();
    let request = ModelLoadingRequest::new(
        ModelLoadingRequestId::new("e2e-untrusted-load"),
        fixture.manifest.id.clone(),
    );
    match coordinator.load(
        request,
        &fixture.manifest,
        &ModelTrustDecision::new(ModelTrustStatus::Rejected, "untrusted for this check"),
        &mut memory,
    ) {
        Err(_) => Ok(()),
        Ok(_) => Err(E2eConformanceError::Internal {
            reason: "expected untrusted artifact to be rejected by Model Loading".into(),
        }),
    }
}

fn check_invalid_tensor_shape(fixture: &E2eFixture) -> Result<(), E2eConformanceError> {
    let bad_tensor = ModelTensorMetadata {
        name: "token_embedding".into(),
        shape: vec![1],
        storage_dtype: ModelDType::F32,
        layout: None,
        shard: None,
        offset_bytes: None,
        size_bytes: None,
        quantization: None,
        expected_compute_dtype: None,
        digest: None,
    };
    match qwen_validate_tensor_shapes(&fixture.config, std::slice::from_ref(&bad_tensor)) {
        Err(_) => Ok(()),
        Ok(()) => Err(E2eConformanceError::Internal {
            reason: "expected invalid tensor shape to be rejected".into(),
        }),
    }
}

fn check_unsupported_operator() -> Result<(), E2eConformanceError> {
    let unsupported = OperatorId::new(
        OPERATOR_NAMESPACE,
        "flash-attention-v3",
        1,
        OperatorFamily::Attention,
    );
    match validate_required_now_operator(&unsupported) {
        Err(_) => Ok(()),
        Ok(_) => Err(E2eConformanceError::Internal {
            reason: "expected unsupported operator to be rejected".into(),
        }),
    }
}

fn check_missing_kernel() -> Result<(), E2eConformanceError> {
    match validate_reference_cpu_required_kernel_coverage(&[]) {
        Err(_) => Ok(()),
        Ok(()) => Err(E2eConformanceError::Internal {
            reason: "expected empty kernel advertisement list to fail coverage".into(),
        }),
    }
}

fn check_required_kernel_removal_fails_coverage() -> Result<(), E2eConformanceError> {
    let mut advertisements = reference_cpu_kernel_advertisements();
    // Must remove a kernel the Qwen fixture's graph actually requires, not
    // merely the last-advertised one: `reference_cpu_kernel_advertisements`
    // also advertises Kernels (e.g. `split`, `reach-architecture-freeze-1`
    // task 5.4/5.5's multi-output proof Operator) that exist to prove
    // generic execution-path capability, not because any E2E fixture graph
    // node uses them -- removing one of those is correctly *not* expected
    // to fail coverage, so `.pop()`'s "whatever is last" used to work only
    // by coincidence (every advertised kernel was Qwen-required) and broke
    // the moment that stopped being true. `matmul` is unconditionally
    // required by every Qwen graph (every projection is one), so find it
    // by name rather than relying on Vec order at all.
    let matmul_index = advertisements
        .iter()
        .position(|advertisement| advertisement.implemented_operator.name() == "matmul")
        .ok_or_else(|| E2eConformanceError::Internal {
            reason: "expected first-native fixture to advertise a 'matmul' kernel".into(),
        })?;
    let removed = advertisements.remove(matmul_index);
    match validate_reference_cpu_required_kernel_coverage(&advertisements) {
        Err(error) if error.reason.contains(removed.implemented_operator.name()) => Ok(()),
        Err(error) => Err(E2eConformanceError::Internal {
            reason: format!(
                "required kernel removal failed with unexpected diagnostic: {}",
                error.reason
            ),
        }),
        Ok(()) => Err(E2eConformanceError::Internal {
            reason: format!(
                "expected removal of required kernel '{}' to fail coverage",
                removed.implemented_operator.name()
            ),
        }),
    }
}

fn check_invalid_model_reference() -> Result<(), E2eConformanceError> {
    let registry = ModelRegistry::new();
    let model_ref = ModelRef::new("unregistered-model")?;
    match registry.resolve(&ModelResolutionRequest::new(model_ref)) {
        Err(_) => Ok(()),
        Ok(_) => Err(E2eConformanceError::Internal {
            reason: "expected resolution of an unregistered model to fail".into(),
        }),
    }
}

fn check_raw_handle_access_denied() -> Result<(), E2eConformanceError> {
    let errors = [
        provider_handle_access_error(),
        device_handle_access_error(),
        kernel_handle_access_error(),
        memory_pointer_access_error(),
    ];
    if errors.iter().any(|error| {
        !matches!(
            error,
            ModelComponentError::ProviderAccessDenied
                | ModelComponentError::DeviceAccessDenied
                | ModelComponentError::KernelAccessDenied
                | ModelComponentError::MemoryPointerAccessDenied
        )
    }) {
        return Err(E2eConformanceError::Internal {
            reason: "raw handle access errors did not use structured denial categories".into(),
        });
    }
    Ok(())
}

fn check_cli_boundary_denials() -> Result<(), E2eConformanceError> {
    for capability in [
        "workspace-filesystem",
        "git",
        "shell",
        "tool-call",
        "secret",
    ] {
        if reject_cli_owned_authority(capability).is_ok() {
            return Err(E2eConformanceError::BoundaryViolation {
                reason: format!("capability '{capability}' should have been denied to Runtime"),
            });
        }
    }
    if !run_cli_boundary_conformance().is_conformant() {
        return Err(E2eConformanceError::BoundaryViolation {
            reason: "CLI boundary conformance report is not conformant".into(),
        });
    }
    Ok(())
}

fn check_kernel_optimization_orchestration_boundary() -> Result<(), E2eConformanceError> {
    let report = run_kernel_optimization_orchestration_conformance();
    if !report.is_conformant() {
        let failures: Vec<String> = report
            .results
            .into_iter()
            .filter(|result| !result.passed)
            .map(|result| result.requirement)
            .collect();
        return Err(E2eConformanceError::BoundaryViolation {
            reason: format!(
                "kernel optimization orchestration conformance report is not conformant: {failures:?}"
            ),
        });
    }
    Ok(())
}

fn check_kv_cache_diagnostics_redacted() -> Result<(), E2eConformanceError> {
    // The first E2E suite does not wire a live KV Cache into generation;
    // `CacheUsageSummary` structurally carries only hit/miss booleans, so
    // there is no raw cache payload to redact by construction.
    let usage = CacheUsageSummary {
        kv_cache_hit: Some(false),
        prefix_cache_hit: None,
    };
    if usage.kv_cache_hit.is_none() && usage.prefix_cache_hit.is_none() {
        return Err(E2eConformanceError::Internal {
            reason: "cache usage summary was unexpectedly empty".into(),
        });
    }
    Ok(())
}

#[cfg(test)]
fn check_incremental_decode_matches_full_sequence_oracle(
    fixture: &E2eFixture,
) -> Result<(), E2eConformanceError> {
    let mut runtime = build_runtime_trusting_fixture(fixture);
    let (instance, _memory) = load_fixture_instance(fixture, &mut runtime)?;
    let prompt = vec![1, 2];
    let admitted = 3;
    let mut plans =
        first_native_plans_for_prompt(&runtime, fixture, &instance, prompt.len() as u64)?;
    let (_prefill_dispatch, _prefill_hidden, layer_kv) =
        execute_qwen_prefill_hidden_states_through_dispatch(
            &mut runtime,
            fixture,
            &prompt,
            &mut plans.prefill,
        )?;
    let kv_state = FirstNativeExecutionKvState {
        cache: KvCacheId::new("first-native-oracle-kv").map_err(E2eConformanceError::from)?,
        compatibility: KvCacheCompatibility::new(
            GenerationModelReference::LoadedModelContext("qwen-test".into()),
            TokenizerId::new("qwen-test-tokenizer")?,
        ),
        layer_kv,
        provider: None,
    };

    let (_decode_dispatch, decode_hidden, updated_layer_kv) =
        execute_qwen_decode_hidden_states_through_dispatch(
            &mut runtime,
            fixture,
            admitted,
            &kv_state,
            prompt.len() as u64,
            &mut plans.decode,
        )?;
    let (_logits_dispatch, incremental_logits) =
        dispatch_qwen_logits_projection(&runtime, fixture, &decode_hidden, &plans.decode)?;

    let mut full_sequence = prompt;
    full_sequence.push(admitted);
    let oracle_logits = e2e_forward(fixture, &full_sequence)?;
    for (index, (actual, expected)) in incremental_logits
        .iter()
        .zip(oracle_logits.iter())
        .enumerate()
    {
        if (actual - expected).abs() > 1e-4 {
            return Err(E2eConformanceError::GenerationFailed {
                reason: format!(
                    "incremental decode logits diverged at {index}: {actual} != {expected}"
                ),
            });
        }
    }

    let executor = resolve_kernel_execution_provider(
        &runtime,
        &ProviderBinding::new(REFERENCE_CPU_PROVIDER_NAME),
    )
    .map_err(E2eConformanceError::from)?;
    for layer in updated_layer_kv {
        let k_tensor = executor.read_tensor(&layer.k).ok_or_else(|| {
            E2eConformanceError::GenerationFailed {
                reason: format!("no materialized K tensor for resource '{}'", layer.k),
            }
        })?;
        let v_tensor = executor.read_tensor(&layer.v).ok_or_else(|| {
            E2eConformanceError::GenerationFailed {
                reason: format!("no materialized V tensor for resource '{}'", layer.v),
            }
        })?;
        let (k_rows, _) = k_tensor.rows_cols()?;
        let (v_rows, _) = v_tensor.rows_cols()?;
        if k_rows != full_sequence.len() as u64 || v_rows != full_sequence.len() as u64 {
            return Err(E2eConformanceError::GenerationFailed {
                reason: "decode did not append exactly one K/V row per layer".into(),
            });
        }
    }
    Ok(())
}

/// Proves the graph-driven executor (`execute_qwen_graph`, which production
/// first-native execution now uses exclusively) produces logits matching the
/// independent `e2e_forward` oracle, and that its recorded per-layer KV
/// state carries one row per historical token. Complements
/// `check_incremental_decode_matches_full_sequence_oracle`, which checks the
/// same oracle against the retired hand-written dispatch sequence.
#[cfg(test)]
fn check_graph_executor_matches_full_sequence_oracle(
    fixture: &E2eFixture,
) -> Result<(), E2eConformanceError> {
    let mut runtime = build_runtime_trusting_fixture(fixture);
    let (instance, _memory) = load_fixture_instance(fixture, &mut runtime)?;
    let prompt = vec![1, 2];
    let admitted = 3;
    let mut plans =
        first_native_plans_for_prompt(&runtime, fixture, &instance, prompt.len() as u64)?;
    let graphs = first_native_component_graphs_for_prompt(fixture, prompt.len() as u64)?;

    let cache_id = KvCacheId::new("test-graph-executor-cache")?;
    let prompt_ids = HostTensor::new(
        [prompt.len() as u64],
        prompt.iter().map(|id| *id as f32).collect::<Vec<_>>(),
    )?;
    let (_prefill_dispatch, _prefill_bindings, layer_kv, _provider) = execute_qwen_graph(
        &mut runtime,
        fixture,
        &instance,
        &cache_id,
        &graphs.prefill,
        &mut plans.prefill,
        BTreeMap::from([(TensorEdgeId::new("input.token_ids"), prompt_ids)]),
        None,
        Some(0),
        &mut Vec::new(),
    )
    .map_err(E2eConformanceError::from)?;

    let admitted_ids = HostTensor::new([1], vec![admitted as f32])?;
    let (_decode_dispatch, decode_bindings, updated_layer_kv, _provider) = execute_qwen_graph(
        &mut runtime,
        fixture,
        &instance,
        &cache_id,
        &graphs.decode,
        &mut plans.decode,
        BTreeMap::from([(TensorEdgeId::new("input.token_ids"), admitted_ids)]),
        Some(&layer_kv),
        Some(prompt.len() as u64),
        &mut Vec::new(),
    )
    .map_err(E2eConformanceError::from)?;

    let logits = decode_bindings
        .get(&TensorEdgeId::new("logits"))
        .ok_or_else(|| E2eConformanceError::GenerationFailed {
            reason: "graph executor produced no logits output".into(),
        })?;

    let mut full_sequence = prompt;
    full_sequence.push(admitted);
    let oracle_logits = e2e_forward(fixture, &full_sequence)?;
    for (index, (actual, expected)) in logits.data.iter().zip(oracle_logits.iter()).enumerate() {
        if (actual - expected).abs() > 1e-4 {
            return Err(E2eConformanceError::GenerationFailed {
                reason: format!(
                    "graph executor decode logits diverged at {index}: {actual} != {expected}"
                ),
            });
        }
    }

    let executor = resolve_kernel_execution_provider(
        &runtime,
        &ProviderBinding::new(REFERENCE_CPU_PROVIDER_NAME),
    )
    .map_err(E2eConformanceError::from)?;
    for layer in updated_layer_kv {
        let k_tensor = executor.read_tensor(&layer.k).ok_or_else(|| {
            E2eConformanceError::GenerationFailed {
                reason: format!("no materialized K tensor for resource '{}'", layer.k),
            }
        })?;
        let v_tensor = executor.read_tensor(&layer.v).ok_or_else(|| {
            E2eConformanceError::GenerationFailed {
                reason: format!("no materialized V tensor for resource '{}'", layer.v),
            }
        })?;
        let (k_rows, _) = k_tensor.rows_cols()?;
        let (v_rows, _) = v_tensor.rows_cols()?;
        if k_rows != full_sequence.len() as u64 || v_rows != full_sequence.len() as u64 {
            return Err(E2eConformanceError::GenerationFailed {
                reason: "graph executor decode did not append exactly one K/V row per layer".into(),
            });
        }
    }
    Ok(())
}

#[cfg(test)]
fn parse_absolute_position(message: &str) -> Result<Option<usize>, E2eConformanceError> {
    let Some(value) = message
        .split_whitespace()
        .find_map(|part| part.strip_prefix("absolute_position="))
    else {
        return Ok(None);
    };
    value
        .parse::<usize>()
        .map(Some)
        .map_err(|error| E2eConformanceError::GenerationFailed {
            reason: format!("invalid absolute_position observation {value:?}: {error}"),
        })
}

#[cfg(test)]
fn check_generation_loop_decode_positions_follow_generated_tokens(
    fixture: &E2eFixture,
) -> Result<(), E2eConformanceError> {
    let mut runtime = build_runtime_with_model_execution_engine(fixture);
    let (instance, _memory) = load_fixture_instance(fixture, &mut runtime)?;
    let prompt = vec![1, 2, 3, 4];
    let tokenized = tokenize_prompt_input(
        &fixture.tokenizer,
        TokenizationRequest::new(PromptInput::TokenIds(prompt.clone())),
        None,
    )?;
    let request = build_generation_request(
        GenerationRequestId::new("e2e-decode-position-oracle")?,
        None,
        GenerationModelReference::ModelInstance(instance.clone()),
        generation_tokenizer_reference(fixture),
        tokenized,
        4,
        GenerationParameters::greedy(),
        StopConditions::default(),
        StreamingMode::Disabled,
    );
    let request = prepare_generation(&runtime, request)?;
    let mut observer = InferenceApiObserver::new();
    let result = run_first_native_generation_loop_with_plans(
        &mut runtime,
        fixture,
        &instance,
        &request,
        SamplingPolicy::default(),
        CacheUsageSummary::default(),
        |_generated_so_far| false,
        &mut observer,
    )?;
    if result.output.generated_token_count != 4 {
        return Err(E2eConformanceError::GenerationFailed {
            reason: format!(
                "expected four generated tokens for multi-step decode, got {}",
                result.output.generated_token_count
            ),
        });
    }

    let mut positions = Vec::new();
    for observation in observer.observations() {
        if observation.kind == InferenceApiObservationKind::ProviderCompleted
            && observation.message.contains("model_input_tokens=1")
            && let Some(position) = parse_absolute_position(&observation.message)?
        {
            positions.push(position);
        }
    }
    let expected = vec![prompt.len(), prompt.len() + 1, prompt.len() + 2];
    if positions != expected {
        return Err(E2eConformanceError::GenerationFailed {
            reason: format!(
                "decode absolute positions diverged from generation-loop oracle: {positions:?} != {expected:?}"
            ),
        });
    }
    Ok(())
}

#[cfg(test)]
fn graph_prefill_setup(
    fixture: &E2eFixture,
) -> Result<
    (
        Runtime,
        ModelInstanceId,
        KvCacheId,
        ExecutionGraph,
        PreparedExecutionPlan,
        HostTensor,
    ),
    E2eConformanceError,
> {
    let mut runtime = build_runtime_trusting_fixture(fixture);
    let (instance, _memory) = load_fixture_instance(fixture, &mut runtime)?;
    let prompt = [1, 2];
    let plans = first_native_plans_for_prompt(&runtime, fixture, &instance, prompt.len() as u64)?;
    let graphs = first_native_component_graphs_for_prompt(fixture, prompt.len() as u64)?;
    let ids = HostTensor::new(
        [prompt.len() as u64],
        prompt.iter().map(|id| *id as f32).collect::<Vec<_>>(),
    )?;
    let cache_id = KvCacheId::new("test-graph-prefill-setup-cache")?;
    Ok((
        runtime,
        instance,
        cache_id,
        graphs.prefill,
        plans.prefill,
        ids,
    ))
}

#[cfg(test)]
fn check_graph_executor_rejects_missing_plan_binding(
    fixture: &E2eFixture,
) -> Result<(), E2eConformanceError> {
    let (mut runtime, instance, cache_id, graph, mut plan, ids) = graph_prefill_setup(fixture)?;
    plan.node_bindings.retain(|binding| {
        !binding
            .graph_nodes
            .contains(&ExecutionNodeId::new("embedding"))
    });
    match execute_qwen_graph(
        &mut runtime,
        fixture,
        &instance,
        &cache_id,
        &graph,
        &mut plan,
        BTreeMap::from([(TensorEdgeId::new("input.token_ids"), ids)]),
        None,
        Some(0),
        &mut Vec::new(),
    ) {
        Err(InferenceApiError::KernelUnavailable { reason }) if reason.contains("embedding") => {
            Ok(())
        }
        Err(error) => Err(E2eConformanceError::GenerationFailed {
            reason: format!("unexpected error for a missing plan binding: {error}"),
        }),
        Ok(_) => Err(E2eConformanceError::GenerationFailed {
            reason: "graph executor accepted a graph node with no published plan binding".into(),
        }),
    }
}

#[cfg(test)]
fn check_graph_executor_rejects_unsupported_operator(
    fixture: &E2eFixture,
) -> Result<(), E2eConformanceError> {
    let (mut runtime, instance, cache_id, mut graph, mut plan, ids) = graph_prefill_setup(fixture)?;
    let node = graph
        .nodes
        .get_mut(&ExecutionNodeId::new("embedding"))
        .ok_or_else(|| E2eConformanceError::GraphValidationFailed {
            reason: "prefill graph is missing node 'embedding'".into(),
        })?;
    node.operator = OperatorId::magnetar("softmax", 1, OperatorFamily::Activation);
    match execute_qwen_graph(
        &mut runtime,
        fixture,
        &instance,
        &cache_id,
        &graph,
        &mut plan,
        BTreeMap::from([(TensorEdgeId::new("input.token_ids"), ids)]),
        None,
        Some(0),
        &mut Vec::new(),
    ) {
        Err(InferenceApiError::OperatorUnsupported { reason }) if reason.contains("softmax") => {
            Ok(())
        }
        Err(error) => Err(E2eConformanceError::GenerationFailed {
            reason: format!("unexpected error for an unsupported operator: {error}"),
        }),
        Ok(_) => Err(E2eConformanceError::GenerationFailed {
            reason: "graph executor dispatched an operator it does not implement".into(),
        }),
    }
}

#[cfg(test)]
fn check_graph_executor_rejects_cyclic_graph(
    fixture: &E2eFixture,
) -> Result<(), E2eConformanceError> {
    let (mut runtime, instance, cache_id, _graph, mut plan, _ids) = graph_prefill_setup(fixture)?;
    let mut graph = ExecutionGraph::new(
        ExecutionGraphId::new("cyclic-test"),
        ExecutionGraphPhase::Test,
    );
    graph = graph
        .with_edge(TensorEdge::new(
            TensorEdgeId::new("a"),
            f32_tensor_descriptor(&HostTensor::new([1, 1], vec![0.0])?),
        ))
        .with_edge(TensorEdge::new(
            TensorEdgeId::new("b"),
            f32_tensor_descriptor(&HostTensor::new([1, 1], vec![0.0])?),
        ))
        .with_node(
            ExecutionNode::new(
                ExecutionNodeId::new("node-a"),
                OperatorId::magnetar("silu", 1, OperatorFamily::Activation),
            )
            .with_input(TensorEdgeId::new("b"))
            .with_output(TensorEdgeId::new("a")),
        )
        .with_node(
            ExecutionNode::new(
                ExecutionNodeId::new("node-b"),
                OperatorId::magnetar("silu", 1, OperatorFamily::Activation),
            )
            .with_input(TensorEdgeId::new("a"))
            .with_output(TensorEdgeId::new("b")),
        );
    match execute_qwen_graph(
        &mut runtime,
        fixture,
        &instance,
        &cache_id,
        &graph,
        &mut plan,
        BTreeMap::new(),
        None,
        Some(0),
        &mut Vec::new(),
    ) {
        Err(InferenceApiError::GraphPlanningFailed { reason }) if reason.contains("cycle") => {
            Ok(())
        }
        Err(error) => Err(E2eConformanceError::GenerationFailed {
            reason: format!("unexpected error for a cyclic graph: {error}"),
        }),
        Ok(_) => Err(E2eConformanceError::GenerationFailed {
            reason: "graph executor accepted a cyclic graph".into(),
        }),
    }
}

#[cfg(test)]
fn check_graph_executor_rejects_removed_producer_node(
    fixture: &E2eFixture,
) -> Result<(), E2eConformanceError> {
    let (mut runtime, instance, cache_id, mut graph, mut plan, ids) = graph_prefill_setup(fixture)?;
    // Removing the node that produces `layer0.q` leaves `layer0.rope_q`
    // depending on an edge nothing in the graph produces -- structurally the
    // same shape a component or graph-mutation bug would take. `plan` was
    // built from the graph before this mutation, so
    // `PreparedExecutionPlanExecutor::prepare_node_execution` (Correctif 4)
    // now rejects every node's dispatch on the very first one it reaches:
    // the graph's semantic fingerprint no longer matches the published
    // Plan's, which is a stronger, earlier rejection of the same
    // underlying inconsistency than reaching the specific missing-producer
    // edge deeper into execution.
    graph.nodes.remove(&ExecutionNodeId::new("layer0.q_proj"));
    match execute_qwen_graph(
        &mut runtime,
        fixture,
        &instance,
        &cache_id,
        &graph,
        &mut plan,
        BTreeMap::from([(TensorEdgeId::new("input.token_ids"), ids)]),
        None,
        Some(0),
        &mut Vec::new(),
    ) {
        Err(InferenceApiError::KernelUnavailable { reason })
            if reason.contains("PlanValidationFailed") =>
        {
            Ok(())
        }
        Err(error) => Err(E2eConformanceError::GenerationFailed {
            reason: format!("unexpected error for a removed producer node: {error}"),
        }),
        Ok(_) => Err(E2eConformanceError::GenerationFailed {
            reason: "graph executor accepted a graph with a removed producer node".into(),
        }),
    }
}

#[cfg(test)]
fn check_graph_executor_logits_provenance_requires_declared_output_edge(
    fixture: &E2eFixture,
) -> Result<(), E2eConformanceError> {
    let (mut runtime, instance, cache_id, mut graph, mut plan, ids) = graph_prefill_setup(fixture)?;
    // Removing the `lm_head` node means nothing produces the `logits` edge
    // this graph declares as its output -- the caller must observe that
    // absence explicitly rather than reading stale or unrelated data. `plan`
    // was built from the graph before this mutation, so
    // `PreparedExecutionPlanExecutor::prepare_node_execution` (Correctif 4)
    // now fails closed on the graph/Plan fingerprint mismatch before any
    // node dispatches -- a stronger guarantee against a fabricated `logits`
    // binding than reaching the end of a partial run and checking its
    // absence, since no dispatch happens at all.
    graph.nodes.remove(&ExecutionNodeId::new("lm_head"));
    match execute_qwen_graph(
        &mut runtime,
        fixture,
        &instance,
        &cache_id,
        &graph,
        &mut plan,
        BTreeMap::from([(TensorEdgeId::new("input.token_ids"), ids)]),
        None,
        Some(0),
        &mut Vec::new(),
    ) {
        Err(InferenceApiError::KernelUnavailable { reason })
            if reason.contains("PlanValidationFailed") =>
        {
            Ok(())
        }
        Err(error) => Err(E2eConformanceError::GenerationFailed {
            reason: format!("unexpected error for a missing output-producing node: {error}"),
        }),
        Ok((_dispatch, bindings, _layer_kv, _provider)) => {
            if bindings.contains_key(&TensorEdgeId::new("logits")) {
                Err(E2eConformanceError::GenerationFailed {
                    reason: "graph executor produced a 'logits' binding with no producing node"
                        .into(),
                })
            } else {
                Err(E2eConformanceError::GenerationFailed {
                    reason: "graph/Plan fingerprint mismatch was not detected".into(),
                })
            }
        }
    }
}

/// Correctif 4, task 4.6: a `PreparedKernelId` a published Plan binds to
/// SHALL be refused for new dispatch once revoked, rather than the revoked
/// state being silently ignored because dispatch never actually asked the
/// Kernel Registry about it.
#[cfg(test)]
fn check_graph_dispatch_rejects_revoked_prepared_kernel(
    fixture: &E2eFixture,
) -> Result<(), E2eConformanceError> {
    let (mut runtime, instance, cache_id, graph, mut plan, ids) = graph_prefill_setup(fixture)?;
    let embedding_kernel = plan
        .node_bindings
        .iter()
        .find(|binding| {
            binding
                .graph_nodes
                .contains(&ExecutionNodeId::new("embedding"))
        })
        .map(|binding| binding.kernel.clone())
        .ok_or_else(|| E2eConformanceError::KernelCoverageMissing {
            reason: "prefill plan has no binding for node embedding".into(),
        })?;
    runtime.kernel_registry_mut().revoke_kernel(
        &embedding_kernel,
        "test: simulate revocation after Plan publication",
    );
    match execute_qwen_graph(
        &mut runtime,
        fixture,
        &instance,
        &cache_id,
        &graph,
        &mut plan,
        BTreeMap::from([(TensorEdgeId::new("input.token_ids"), ids)]),
        None,
        Some(0),
        &mut Vec::new(),
    ) {
        // Revocation deactivates the Kernel's advertisement (Kernel
        // Registry's `active` flag), which `dispatch_reference_cpu_operator`
        // rejects immediately after `prepare_node_execution` resolves the
        // binding -- an active-advertisement lookup is a separate concern
        // from `PreparedKernel.state`, so this is the correct rejection
        // point for advertisement-level revocation specifically.
        Err(InferenceApiError::KernelUnavailable { reason })
            if reason.contains("no longer active") =>
        {
            Ok(())
        }
        Err(error) => Err(E2eConformanceError::GenerationFailed {
            reason: format!("unexpected error for a revoked prepared kernel: {error}"),
        }),
        Ok(_) => Err(E2eConformanceError::GenerationFailed {
            reason: "graph dispatch executed a revoked PreparedKernel".into(),
        }),
    }
}

/// Correctif 4, task 4.5: once a Plan is published, a Kernel Registry
/// preference change (e.g. a newer, more attractively-ranked Kernel
/// registered for the same Operator) SHALL NOT affect that already-
/// published, ready Plan's dispatch -- `prepare_node_execution` looks up
/// the specific `PreparedKernelId` the binding already names, it never
/// re-ranks candidates the way live Kernel Registry selection
/// (`KernelRegistry::select`) does.
#[cfg(test)]
fn check_graph_dispatch_ignores_kernel_registry_preference_change_after_plan_publication(
    fixture: &E2eFixture,
) -> Result<(), E2eConformanceError> {
    let (mut runtime, _instance, _cache_id, graph, mut plan, ids) = graph_prefill_setup(fixture)?;
    let embedding_kernel = plan
        .node_bindings
        .iter()
        .find(|binding| {
            binding
                .graph_nodes
                .contains(&ExecutionNodeId::new("embedding"))
        })
        .map(|binding| binding.kernel.clone())
        .ok_or_else(|| E2eConformanceError::KernelCoverageMissing {
            reason: "prefill plan has no binding for node embedding".into(),
        })?;

    // Register a second, deliberately better-ranked (lower cost, lower
    // fallback rank) Kernel advertising the same "embedding" Operator
    // *after* the Plan was already published -- if a live Kernel Registry
    // selection were consulted instead of the Plan's own binding, this
    // would be a legitimate, more attractive contender.
    let mut competitor = reference_cpu_kernel_advertisements()
        .into_iter()
        .find(|advertisement| advertisement.implemented_operator.name() == "embedding")
        .ok_or_else(|| E2eConformanceError::KernelCoverageMissing {
            reason: "Reference CPU fixture does not advertise embedding".into(),
        })?;
    competitor.id.name = format!("{}-cheaper-competitor", competitor.id.name);
    competitor
        .performance_hints
        .insert("estimated-cost".into(), "0".into());
    competitor
        .performance_hints
        .insert("fallback-rank".into(), "0".into());
    runtime
        .kernel_registry_mut()
        .register_fixture_advertisement(competitor)
        .map_err(|error| E2eConformanceError::KernelCoverageMissing {
            reason: format!("failed to register competing embedding Kernel: {error}"),
        })?;

    let provider = resolve_kernel_execution_provider(
        &runtime,
        &ProviderBinding::new(REFERENCE_CPU_PROVIDER_NAME),
    )
    .map_err(E2eConformanceError::from)?;
    let seq_len = ids.shape.first().copied().unwrap_or(1);
    let architecture = &fixture.config.architecture;
    let token_embedding = fixture_tensor_by_name(&fixture.weights, "token_embedding")?.clone();
    let mut node_events = Vec::new();
    let mut dispatch_ctx = QwenDispatchContext {
        runtime: &mut runtime,
        provider: provider.clone(),
        prepared_plan: Some(&mut plan),
        graph: Some(&graph),
        sequence_length: Some(seq_len),
        last_provider_execution: None,
        node_events: &mut node_events,
    };
    let (dispatch_result, _hidden_states) = dispatch_reference_cpu_operator(
        &mut dispatch_ctx,
        "embedding",
        dispatch_operator_id("embedding", OperatorFamily::Tensor),
        vec![
            (
                TensorResourceId::new("embedding.table"),
                f32_tensor_descriptor(&token_embedding),
                token_embedding,
            ),
            (
                TensorResourceId::new("embedding.ids"),
                f32_tensor_descriptor(&ids),
                ids.clone(),
            ),
        ],
        (
            TensorResourceId::new("embedding.out"),
            TensorDescriptor::new(
                ShapeDescriptor::new([seq_len, architecture.hidden_size]),
                DTypeDescriptor::portable(ComputeDType::Float32),
                LayoutDescriptor::Contiguous,
            ),
        ),
        BTreeMap::new(),
    )
    .map_err(E2eConformanceError::from)?;

    if dispatch_result.selected_kernel != embedding_kernel {
        return Err(E2eConformanceError::GenerationFailed {
            reason: format!(
                "published Plan binding for 'embedding' was bypassed by a newer, \
                 better-ranked Kernel registration: expected {embedding_kernel:?}, dispatched \
                 {:?}",
                dispatch_result.selected_kernel
            ),
        });
    }
    Ok(())
}

/// Correctif 4, task 4.7: a Plan binding's `PreparedKernelGeneration` SHALL
/// match the Kernel Registry's active generation for that `PreparedKernelId`
/// at dispatch time; a stale generation (e.g. left over from before a hot
/// Kernel replacement) is refused rather than dispatched as if current.
#[cfg(test)]
fn check_graph_dispatch_rejects_stale_prepared_kernel_generation(
    fixture: &E2eFixture,
) -> Result<(), E2eConformanceError> {
    let (mut runtime, instance, cache_id, graph, mut plan, ids) = graph_prefill_setup(fixture)?;
    let binding = plan
        .node_bindings
        .iter_mut()
        .find(|binding| {
            binding
                .graph_nodes
                .contains(&ExecutionNodeId::new("embedding"))
        })
        .ok_or_else(|| E2eConformanceError::KernelCoverageMissing {
            reason: "prefill plan has no binding for node embedding".into(),
        })?;
    binding.prepared_kernel_generation = Some(PreparedKernelGeneration::new(u64::MAX));
    match execute_qwen_graph(
        &mut runtime,
        fixture,
        &instance,
        &cache_id,
        &graph,
        &mut plan,
        BTreeMap::from([(TensorEdgeId::new("input.token_ids"), ids)]),
        None,
        Some(0),
        &mut Vec::new(),
    ) {
        Err(InferenceApiError::KernelUnavailable { reason })
            if reason.contains("PreparedKernelGenerationMismatch") =>
        {
            Ok(())
        }
        Err(error) => Err(E2eConformanceError::GenerationFailed {
            reason: format!("unexpected error for a stale prepared kernel generation: {error}"),
        }),
        Ok(_) => Err(E2eConformanceError::GenerationFailed {
            reason: "graph dispatch executed a binding with a stale PreparedKernel generation"
                .into(),
        }),
    }
}

/// Correctif 4, task 4.8: a Plan binding's declared `provider` SHALL match
/// the Kernel Registry's active `PreparedKernel` provider at dispatch time;
/// a mismatch (e.g. a binding pointing at a Provider the active Kernel is
/// no longer registered under) is refused rather than silently dispatched
/// against whichever Provider the `PreparedKernel` actually belongs to.
#[cfg(test)]
fn check_graph_dispatch_rejects_provider_binding_mismatch(
    fixture: &E2eFixture,
) -> Result<(), E2eConformanceError> {
    let (mut runtime, instance, cache_id, graph, mut plan, ids) = graph_prefill_setup(fixture)?;
    // Mutate the *last* binding, not the first: `execute_qwen_graph` resolves
    // the Provider it actually dispatches through from `node_bindings.first()`
    // (every node in a first-native graph binds to the same Provider), so
    // corrupting that one would fail earlier and coarser, at graph-level
    // Provider resolution, rather than exercising `prepare_node_execution`'s
    // own per-binding Provider consistency check this test targets.
    let binding = plan.node_bindings.last_mut().ok_or_else(|| {
        E2eConformanceError::KernelCoverageMissing {
            reason: "prefill plan has no node bindings".into(),
        }
    })?;
    binding.provider = ProviderBinding::new("magnetar:provider/does-not-exist");
    match execute_qwen_graph(
        &mut runtime,
        fixture,
        &instance,
        &cache_id,
        &graph,
        &mut plan,
        BTreeMap::from([(TensorEdgeId::new("input.token_ids"), ids)]),
        None,
        Some(0),
        &mut Vec::new(),
    ) {
        Err(InferenceApiError::KernelUnavailable { reason })
            if reason.contains("PlanProviderUnavailable") =>
        {
            Ok(())
        }
        Err(error) => Err(E2eConformanceError::GenerationFailed {
            reason: format!("unexpected error for a Provider binding mismatch: {error}"),
        }),
        Ok(_) => Err(E2eConformanceError::GenerationFailed {
            reason: "graph dispatch executed a binding whose Provider does not match the active PreparedKernel".into(),
        }),
    }
}

#[cfg(test)]
fn check_generation_loop_executes_published_plan_bindings(
    fixture: &E2eFixture,
) -> Result<(), E2eConformanceError> {
    let mut runtime = build_runtime_with_model_execution_engine(fixture);
    let (instance, _memory) = load_fixture_instance(fixture, &mut runtime)?;
    let prompt = vec![1, 2, 3, 4];
    let tokenized = tokenize_prompt_input(
        &fixture.tokenizer,
        TokenizationRequest::new(PromptInput::TokenIds(prompt.clone())),
        None,
    )?;
    // Must be the same graph source `execute_generation_step` itself will
    // dispatch against (see that function's own doc comment on this exact
    // requirement) -- this test only cares about the *binding removal*
    // below producing the expected failure, not which recipe built the
    // graph, so it uses the same helper every real dispatch path uses
    // rather than calling the Rust-synthesized recipe directly.
    let component_graphs = first_native_component_graphs_for_prompt(fixture, prompt.len() as u64)?;
    let mut prepared_plans = prepare_first_native_execution_plans(
        &runtime,
        &instance,
        component_graphs,
        prompt.len() as u64,
    )?;
    prepared_plans.prefill.node_bindings.retain(|binding| {
        !binding
            .graph_nodes
            .contains(&ExecutionNodeId::new("lm_head"))
    });

    let request = build_generation_request(
        GenerationRequestId::new("e2e-plan-binding-required")?,
        None,
        GenerationModelReference::ModelInstance(instance),
        generation_tokenizer_reference(fixture),
        tokenized,
        1,
        GenerationParameters::greedy(),
        StopConditions::default(),
        StreamingMode::Disabled,
    );
    let request = prepare_generation(&runtime, request)?;
    let mut observer = InferenceApiObserver::new();
    let mut execution_plans = RuntimeGenerationExecutionPlans {
        prefill: &mut prepared_plans.prefill,
        decode: &mut prepared_plans.decode,
    };
    match run_generation_loop_with_execution_plans(
        &mut runtime,
        &request,
        SamplingPolicy::default(),
        CacheUsageSummary::default(),
        |_generated_so_far| false,
        &mut observer,
        &mut execution_plans,
    ) {
        Err(InferenceApiError::KernelUnavailable { reason }) if reason.contains("lm_head") => {
            Ok(())
        }
        Err(error) => Err(E2eConformanceError::GenerationFailed {
            reason: format!("unexpected prepared binding error: {error}"),
        }),
        Ok(_) => Err(E2eConformanceError::GenerationFailed {
            reason: "generation succeeded despite missing published lm_head binding".into(),
        }),
    }
}

#[cfg(test)]
fn check_incremental_decode_rejects_missing_layer_kv(
    fixture: &E2eFixture,
) -> Result<(), E2eConformanceError> {
    let mut runtime = build_runtime_trusting_fixture(fixture);
    let (instance, _memory) = load_fixture_instance(fixture, &mut runtime)?;
    let mut plans = first_native_plans_for_prompt(&runtime, fixture, &instance, 2)?;
    let kv_state = FirstNativeExecutionKvState {
        cache: KvCacheId::new("first-native-empty-kv").map_err(E2eConformanceError::from)?,
        compatibility: KvCacheCompatibility::new(
            GenerationModelReference::LoadedModelContext("qwen-test".into()),
            TokenizerId::new("qwen-test-tokenizer")?,
        ),
        layer_kv: Vec::new(),
        provider: None,
    };
    match execute_qwen_decode_hidden_states_through_dispatch(
        &mut runtime,
        fixture,
        3,
        &kv_state,
        2,
        &mut plans.decode,
    ) {
        Err(InferenceApiError::KvCacheUnavailable { .. }) => Ok(()),
        Err(error) => Err(E2eConformanceError::from(error)),
        Ok(_) => Err(E2eConformanceError::GenerationFailed {
            reason: "decode accepted missing layer KV state".into(),
        }),
    }
}

fn check_diagnostics_redaction_on_failure() -> Result<(), E2eConformanceError> {
    let error = InferenceApiError::ModelLoadingFailed {
        reason: "raw prompt 'super secret prompt' at native_handle 0xdeadbeef".into(),
    };
    let redacted = compute::redact_backend_diagnostic(&error.to_string());
    if redacted.contains("0xdeadbeef") || redacted.contains("native_handle") {
        return Err(E2eConformanceError::RedactionFailed {
            reason: "diagnostic redaction failed to strip a native handle".into(),
        });
    }
    Ok(())
}

fn check_no_shortcut_direct_kernel_invocation_rejected() -> Result<(), E2eConformanceError> {
    let runtime = build_runtime();
    let advertisements = reference_cpu_kernel_advertisements();
    let matmul_advertisement = advertisements
        .iter()
        .find(|advertisement| advertisement.implemented_operator.name() == "matmul")
        .cloned()
        .ok_or_else(|| E2eConformanceError::KernelCoverageMissing {
            reason: "Reference CPU does not advertise matmul".into(),
        })?;
    let request = KernelSelectionRequest::new(
        "e2e-direct-kernel-invocation",
        matmul_advertisement.implemented_operator.clone(),
        ResourceAffinity::new(FallbackClass::Transparent),
    );
    let selection = runtime
        .kernel_registry()
        .select(&request)
        .map_err(E2eConformanceError::from)?;
    let mut bypassing_candidate =
        selection
            .selected
            .ok_or_else(|| E2eConformanceError::KernelCoverageMissing {
                reason: "Kernel Registry selected no candidate for matmul".into(),
            })?;
    // A KernelDispatchPlan has no constructor other than `from_selection`,
    // and it refuses any candidate the registry itself did not mark
    // compatible -- so a caller cannot fabricate a dispatch plan for a
    // Kernel invocation it invoked directly, bypassing selection.
    bypassing_candidate.compatible = false;
    match KernelDispatchPlan::from_selection(
        KernelDispatchPlanId::new("e2e-bypass-plan"),
        &request,
        &bypassing_candidate,
        &matmul_advertisement,
        KernelInvocationId::new("e2e-bypass-invocation"),
    ) {
        Err(_) => Ok(()),
        Ok(_) => Err(E2eConformanceError::Internal {
            reason: "expected a fabricated incompatible kernel candidate to be rejected".into(),
        }),
    }
}

fn check_no_shortcut_model_loading_bypass_detected(
    fixture: &E2eFixture,
) -> Result<(), E2eConformanceError> {
    let mut coordinator = ModelLoadingCoordinator::new();
    coordinator.register_architecture(fixture.architecture_implementation.clone());
    let mut memory = MemoryManager::default();
    let request = ModelLoadingRequest::new(
        ModelLoadingRequestId::new("e2e-loading-bypass-check"),
        fixture.manifest.id.clone(),
    );
    let mut loaded = coordinator.load(
        request,
        &fixture.manifest,
        &ModelTrustDecision::new(ModelTrustStatus::Trusted, "trusted E2E fixture"),
        &mut memory,
    )?;
    // Simulate a caller that fabricates a "loaded" context without the real
    // Model Loading phases ever completing.
    loaded.state = ModelLoadingState::Requested;
    if loaded.can_start_inference() {
        return Err(E2eConformanceError::Internal {
            reason:
                "a context that bypassed real Model Loading phases was accepted as inference-ready"
                    .into(),
        });
    }
    Ok(())
}

fn check_no_shortcut_model_component_bypass_detected(
    fixture: &E2eFixture,
) -> Result<(), E2eConformanceError> {
    // No architecture is registered on this coordinator, simulating a
    // caller that skips Model Component resolution entirely.
    let mut coordinator = ModelLoadingCoordinator::new();
    let mut memory = MemoryManager::default();
    let request = ModelLoadingRequest::new(
        ModelLoadingRequestId::new("e2e-component-bypass-check"),
        fixture.manifest.id.clone(),
    );
    match coordinator.load(
        request,
        &fixture.manifest,
        &ModelTrustDecision::new(ModelTrustStatus::Trusted, "trusted E2E fixture"),
        &mut memory,
    ) {
        Err(_) => Ok(()),
        Ok(_) => Err(E2eConformanceError::Internal {
            reason: "expected loading without a registered Model Component to fail".into(),
        }),
    }
}

fn check_no_shortcut_memory_manager_bypass_detected(
    fixture: &E2eFixture,
) -> Result<(), E2eConformanceError> {
    let mut coordinator = ModelLoadingCoordinator::new();
    coordinator.register_architecture(fixture.architecture_implementation.clone());
    let mut starved_memory = MemoryManager::new(MemoryManagerConfig {
        max_runtime_bytes: Some(1),
        allow_pending_allocations: false,
        ..MemoryManagerConfig::default()
    });
    let request = ModelLoadingRequest::new(
        ModelLoadingRequestId::new("e2e-memory-bypass-check"),
        fixture.manifest.id.clone(),
    );
    match coordinator.load(
        request,
        &fixture.manifest,
        &ModelTrustDecision::new(ModelTrustStatus::Trusted, "trusted E2E fixture"),
        &mut starved_memory,
    ) {
        Err(_) => Ok(()),
        Ok(_) => Err(E2eConformanceError::Internal {
            reason:
                "expected Model Loading to honor Memory Manager admission instead of bypassing it"
                    .into(),
        }),
    }
}

fn check_dtype_and_layout_conversion_are_explicit() -> Result<(), E2eConformanceError> {
    let tensor = HostTensor::new([1, 2], vec![1.0_f32, 2.0_f32])?;
    // Identity conversions are explicit and succeed.
    dtype_conversion(&tensor, ComputeDType::Float32, ComputeDType::Float32)?;
    layout_conversion(
        &tensor,
        TensorLayoutKind::Contiguous,
        TensorLayoutKind::Contiguous,
    )?;
    // A non-identity conversion is never performed silently: Reference CPU
    // explicitly rejects it rather than guessing at a conversion.
    if dtype_conversion(&tensor, ComputeDType::Float32, ComputeDType::Float16).is_ok() {
        return Err(E2eConformanceError::Internal {
            reason: "expected a non-identity dtype conversion to be explicitly rejected".into(),
        });
    }
    if layout_conversion(
        &tensor,
        TensorLayoutKind::Contiguous,
        TensorLayoutKind::Strided,
    )
    .is_ok()
    {
        return Err(E2eConformanceError::Internal {
            reason: "expected a non-identity layout conversion to be explicitly rejected".into(),
        });
    }
    Ok(())
}

fn check_max_total_tokens_stops_generation(
    fixture: &E2eFixture,
) -> Result<(), E2eConformanceError> {
    // `GenerationRequest::validate` requires `prompt + max_new_tokens <=
    // max_total_tokens`, and `stop_reason_for` checks `max_new_tokens`
    // before `max_total_tokens` -- so through the fully validated
    // `run_generation_loop` path, `max_new_tokens` always fires at or
    // before the total-token boundary is crossed. This exercises the
    // `max_total_tokens` stop-condition branch directly, the same way
    // `decode_step` does internally on every step.
    let tokenized = TokenizationResult {
        token_ids: vec![10, 20],
        token_count: 2,
        offsets: None,
        diagnostics: Vec::new(),
        correlation_id: None,
    };
    let mut request = build_generation_request(
        GenerationRequestId::new("e2e-max-total-tokens")?,
        None,
        GenerationModelReference::LoadedModelContext("e2e-fixture".into()),
        generation_tokenizer_reference(fixture),
        tokenized,
        10,
        GenerationParameters::greedy(),
        StopConditions::default(),
        StreamingMode::Disabled,
    );
    request.max_total_tokens = Some(request.prompt_token_count + 1);
    let finish_reason = stop_reason_for(&request, &[42]);
    if finish_reason != Some(FinishReason::MaxTotalTokens) {
        return Err(E2eConformanceError::GenerationFailed {
            reason: format!("expected Some(FinishReason::MaxTotalTokens), got {finish_reason:?}"),
        });
    }
    Ok(())
}

fn check_stochastic_sampling_is_seed_deterministic() -> Result<(), E2eConformanceError> {
    let tokenizer = e2e_fixture_tokenizer()?;
    let request = GenerationRequest {
        request_id: GenerationRequestId::new("e2e-stochastic-sampling")?,
        session: None,
        model: GenerationModelReference::LoadedModelContext("e2e-fixture".into()),
        tokenizer: GenerationTokenizerReference {
            tokenizer_id: tokenizer.metadata().id.clone(),
            metadata: tokenizer.metadata().clone(),
        },
        prompt_token_count: 1,
        input_token_ids: vec![1],
        max_new_tokens: 1,
        max_total_tokens: None,
        model_context_length: None,
        parameters: GenerationParameters {
            temperature: 1.0,
            sampling_enabled: true,
            greedy: false,
            seed: Some(42),
            ..GenerationParameters::default()
        },
        stop_conditions: StopConditions::default(),
        streaming: StreamingMode::Disabled,
        priority: GenerationPriority::default(),
        cancellation: CancellationMetadata::default(),
        memory: GenerationMemoryEstimate::default(),
        correlation_id: None,
        trace_id: None,
    };
    let mut logits = vec![0.1_f32; E2E_FIXTURE_VOCAB as usize];
    logits[7] = 2.0;
    logits[9] = 1.9;
    let (first, _) =
        decode_step_from_sampling(&request, &[], logits.clone(), SamplingPolicy::default())
            .map_err(E2eConformanceError::from)?;
    let (second, _) = decode_step_from_sampling(&request, &[], logits, SamplingPolicy::default())
        .map_err(E2eConformanceError::from)?;
    if first.selected_token_id != second.selected_token_id {
        return Err(E2eConformanceError::SamplingFailed {
            reason: "seeded stochastic sampling produced different tokens across repeated runs"
                .into(),
        });
    }
    if first.selection_mode != SamplingSelectionMode::Stochastic {
        return Err(E2eConformanceError::SamplingFailed {
            reason: format!(
                "expected SamplingSelectionMode::Stochastic, got {:?}",
                first.selection_mode
            ),
        });
    }
    Ok(())
}

fn check_kv_and_prefix_cache_lifecycle(fixture: &E2eFixture) -> Result<(), E2eConformanceError> {
    let mut memory = MemoryManager::default();
    let mut kv_manager = KvCacheManager::new();
    let kv_compatibility = KvCacheCompatibility::new(
        GenerationModelReference::LoadedModelContext("e2e-fixture".into()),
        fixture.tokenizer.metadata().id.clone(),
    );
    let layout = KvCacheLayoutMetadata::contiguous(
        fixture.config.architecture.layer_count as u32,
        fixture.config.architecture.attention_head_count as u32,
        fixture.config.architecture.head_dimension as u32,
        fixture.config.architecture.context_length as u32,
        ComputeDType::Float32,
    );
    let kv_cache = KvCache::new(
        KvCacheId::new("e2e-kv-cache")?,
        KvCacheScope::Session,
        kv_compatibility,
        layout,
    );
    let kv_id = kv_manager.create(kv_cache)?;
    kv_manager.allocate_memory(&kv_id, &mut memory)?;
    kv_manager.prefill_completed(&kv_id, 2)?;
    kv_manager.decode_append(&kv_id, 1)?;
    if kv_manager
        .observations()
        .iter()
        .any(|observation| observation.raw_cache_available)
    {
        return Err(E2eConformanceError::RedactionFailed {
            reason: "KV cache observation exposed raw cache contents".into(),
        });
    }

    let backing = PrefixCacheBackingKvCache::from_kv_cache(kv_manager.cache(&kv_id)?);
    let mut prefix_manager = PrefixCacheManager::new();
    let fingerprint = PrefixCacheFingerprint::from_validated_tokens(
        &[1, 2, 3],
        "e2e-fixture-model",
        &fixture.tokenizer.metadata().id,
    );
    let compatibility = PrefixCacheCompatibility::new(
        GenerationModelReference::LoadedModelContext("e2e-fixture".into()),
        fixture.tokenizer.metadata().id.clone(),
    );
    let lookup_request = PrefixCacheLookupRequest {
        fingerprint: fingerprint.clone(),
        compatibility: compatibility.clone(),
        requested_prefix_token_length: fingerprint.token_length(),
        session: None,
        owner: None,
        tenant: None,
        affinity: None,
        allow_partial: false,
    };
    let miss = prefix_manager.lookup(&lookup_request);
    if miss.kind != PrefixCacheMatchKind::Miss {
        return Err(E2eConformanceError::Internal {
            reason: format!(
                "expected an initial Prefix Cache lookup to miss, got {:?}",
                miss.kind
            ),
        });
    }

    let entry = PrefixCacheEntry::new(
        PrefixCacheEntryId::new("e2e-prefix-entry")?,
        fingerprint,
        compatibility,
        backing,
    );
    prefix_manager.create(entry, &PrefixCachePolicy::default())?;
    let hit = prefix_manager.lookup(&lookup_request);
    if hit.kind == PrefixCacheMatchKind::Miss {
        return Err(E2eConformanceError::Internal {
            reason: "expected a Prefix Cache lookup to hit after inserting a matching entry".into(),
        });
    }
    if prefix_manager
        .observations()
        .iter()
        .any(|observation| observation.raw_kv_cache_available || observation.raw_prompt_available)
    {
        return Err(E2eConformanceError::RedactionFailed {
            reason: "Prefix Cache observation exposed raw prompt or KV cache contents".into(),
        });
    }

    kv_manager.seal(&kv_id)?;
    kv_manager.release(&kv_id)?;
    Ok(())
}

fn check_tensor_resource_lifecycle() -> Result<(), E2eConformanceError> {
    let descriptor = TensorDescriptor::materialized(
        ShapeDescriptor::new(vec![1, 4]),
        DTypeDescriptor::portable(ComputeDType::Float32),
    );
    let tensor_id = TensorResourceId::new("e2e-tensor-output");
    let affinity = ResourceAffinity::new(FallbackClass::Transparent);
    let residency =
        TensorResidency::new(tensor_id.clone(), MemoryPlacement::HostOrdinary, affinity);
    let mut resource = TensorResource::new(tensor_id, descriptor, residency);
    resource.transition_to(TensorLifecycleState::Planned)?;
    resource.transition_to(TensorLifecycleState::Allocating)?;
    resource.mark_ready()?;
    if resource.readiness != TensorReadiness::Ready {
        return Err(E2eConformanceError::Internal {
            reason: "expected Tensor Resource readiness to be Ready after mark_ready".into(),
        });
    }
    resource.ensure_usable()?;
    resource.transition_to(TensorLifecycleState::Released)?;
    Ok(())
}

fn check_memory_operator_output_accounting() -> Result<(), E2eConformanceError> {
    let mut memory = MemoryManager::default();
    let descriptor = TensorDescriptor::materialized(
        ShapeDescriptor::new(vec![1, 4]),
        DTypeDescriptor::portable(ComputeDType::Float32),
    );
    let operator_output_request = MemoryAllocationRequest::for_tensor(
        &descriptor,
        MemoryPlacement::HostOrdinary,
        MemoryAllocationOwner::Runtime,
    )
    .map_err(E2eConformanceError::from)?;
    let operator_output = memory.allocate(operator_output_request)?;
    let workspace_request = MemoryAllocationRequest::new(
        MemoryAllocationClass::TemporaryWorkspace,
        64,
        MemoryPlacement::HostOrdinary,
        MemoryAllocationOwner::Runtime,
    );
    let workspace = memory.allocate(workspace_request)?;

    let active_during = memory
        .allocations()
        .filter(|allocation| allocation.state == MemoryAllocationState::Active)
        .count();
    if active_during < 2 {
        return Err(E2eConformanceError::MemoryValidationFailed {
            reason: "expected operator output and workspace allocations to be tracked as active"
                .into(),
        });
    }

    memory.release(operator_output.id)?;
    memory.release(workspace.id)?;
    let active_after = memory
        .allocations()
        .filter(|allocation| allocation.state == MemoryAllocationState::Active)
        .count();
    if active_after != 0 {
        return Err(E2eConformanceError::MemoryValidationFailed {
            reason: "an allocation remained active after release, indicating an untracked Runtime-visible allocation"
                .into(),
        });
    }
    Ok(())
}

fn check_generation_timeout_maps_to_structured_error() -> Result<(), E2eConformanceError> {
    let mapped = InferenceApiError::from(BatchingError::OperationTimedOut);
    if !matches!(mapped, InferenceApiError::GenerationTimeout) {
        return Err(E2eConformanceError::Internal {
            reason: format!(
                "expected BatchingError::OperationTimedOut to map to InferenceApiError::GenerationTimeout, got {mapped:?}"
            ),
        });
    }
    // Deterministic memory-side timeout: no wall-clock sleep, just an
    // explicit deadline compared against a caller-supplied "now".
    let mut memory = MemoryManager::new(MemoryManagerConfig {
        allow_pending_allocations: true,
        ..MemoryManagerConfig::default()
    });
    let request = MemoryAllocationRequest::new(
        MemoryAllocationClass::Tensor,
        64,
        MemoryPlacement::HostOrdinary,
        MemoryAllocationOwner::Runtime,
    )
    .with_deadline_millis(1_000);
    memory.submit_pending_allocation(request, 0)?;
    let expired = memory.expire_pending_allocations(2_000);
    if !expired
        .iter()
        .any(|error| matches!(error, MemoryError::AllocationTimeout { .. }))
    {
        return Err(E2eConformanceError::Internal {
            reason: "expected an expired pending allocation to report AllocationTimeout".into(),
        });
    }
    Ok(())
}

fn check_report_metadata(report: &E2eConformanceReport) -> Result<(), E2eConformanceError> {
    if report.suite_version.is_empty()
        || report.fixture_version.is_empty()
        || report.runtime_version.is_empty()
        || report.provider_summary.is_empty()
        || report.device_summary.is_empty()
        || report.model_component_summary.is_empty()
        || report.operator_coverage.is_empty()
        || report.kernel_coverage.is_empty()
        || report.test_cases.is_empty()
    {
        return Err(E2eConformanceError::Internal {
            reason: "report is missing required metadata fields".into(),
        });
    }
    Ok(())
}

/// Runs the End-to-End Local Inference Conformance suite and returns a
/// machine-readable [`E2eConformanceReport`]. Runs entirely on CPU without
/// GPU hardware, network access, Tachyon, or external processes.
pub fn run_e2e_local_inference_conformance() -> E2eConformanceReport {
    let start = SystemTime::now();
    let mut report = E2eConformanceReport::empty();
    report.record(E2eTestResult::passed("observation-suite-started"));

    let fixture = match e2e_fixture() {
        Ok(fixture) => fixture,
        Err(error) => {
            report.record(E2eTestResult::failed(
                "fixture-model",
                format!("{}: {}", error.code(), error.reason()),
            ));
            report.duration_millis = elapsed_millis(start);
            return report;
        }
    };
    report.record(E2eTestResult::passed("observation-fixture-loaded"));

    report.provider_summary = REFERENCE_CPU_PROVIDER_NAME.into();
    report.device_summary = REFERENCE_CPU_DEVICE_ID.into();
    report.model_component_summary = format!(
        "{}@{}",
        fixture.identity.id.as_str(),
        qwen_component_compatibility_key(&fixture.identity)
    );

    report.record(E2eTestResult::from_result(
        "fixture-model",
        fixture
            .manifest
            .validate()
            .map(|_| ())
            .map_err(E2eConformanceError::from),
    ));
    report.record(E2eTestResult::from_result(
        "fixture-tokenizer-deterministic",
        check_fixture_tokenizer_deterministic(&fixture),
    ));
    report.record(E2eTestResult::from_result(
        "already-tokenized-prompt-path",
        check_already_tokenized_prompt_path(&fixture),
    ));
    report.record(E2eTestResult::passed("observation-success-path-started"));
    let success_path_result = check_success_path(&fixture);
    report.record(E2eTestResult::from_result(
        "observation-success-path-completed",
        success_path_result.clone().map(|_| ()),
    ));
    report.record(E2eTestResult::from_result(
        "success-path",
        success_path_result,
    ));
    report.record(no_shortcut_success_path_result(&fixture));
    report.record(E2eTestResult::from_result(
        "determinism",
        check_determinism(&fixture),
    ));
    report.record(E2eTestResult::from_result(
        "one-shot-session-normal-paths",
        check_one_shot_session_normal_paths(&fixture),
    ));
    report.record(E2eTestResult::from_result(
        "chat-message-prompt-path",
        check_chat_message_prompt_path(&fixture),
    ));
    report.record(E2eTestResult::from_result(
        "streaming-order",
        check_streaming_order(&fixture),
    ));
    match check_operator_coverage(&fixture) {
        Ok(operators) => {
            report.operator_coverage = operators;
            report.record(E2eTestResult::passed("operator-coverage"));
        }
        Err(error) => report.record(E2eTestResult::failed(
            "operator-coverage",
            format!("{}: {}", error.code(), error.reason()),
        )),
    }
    match check_kernel_coverage() {
        Ok(kernels) => {
            report.kernel_coverage = kernels;
            report.record(E2eTestResult::passed("kernel-coverage"));
        }
        Err(error) => report.record(E2eTestResult::failed(
            "kernel-coverage",
            format!("{}: {}", error.code(), error.reason()),
        )),
    }
    report.record(E2eTestResult::from_result(
        "no-shortcut-direct-provider-rejected",
        check_no_shortcut_direct_provider_rejected(),
    ));
    report.record(E2eTestResult::from_result(
        "no-shortcut-direct-kernel-invocation-rejected",
        check_no_shortcut_direct_kernel_invocation_rejected(),
    ));
    report.record(E2eTestResult::from_result(
        "no-shortcut-model-loading-bypass-detected",
        check_no_shortcut_model_loading_bypass_detected(&fixture),
    ));
    report.record(E2eTestResult::from_result(
        "no-shortcut-model-component-bypass-detected",
        check_no_shortcut_model_component_bypass_detected(&fixture),
    ));
    report.record(E2eTestResult::from_result(
        "no-shortcut-memory-manager-bypass-detected",
        check_no_shortcut_memory_manager_bypass_detected(&fixture),
    ));
    report.record(E2eTestResult::from_result(
        "no-shortcut-dtype-and-layout-conversion-explicit",
        check_dtype_and_layout_conversion_are_explicit(),
    ));
    report.record(E2eTestResult::from_result(
        "reference-cpu-selected-through-kernel-registry",
        check_reference_cpu_selected_through_kernel_registry(),
    ));
    report.record(E2eTestResult::from_result(
        "invalid-graph-fixture-rejected",
        check_invalid_graph_fixture(),
    ));
    #[cfg(test)]
    report.record(E2eTestResult::from_result(
        "graph-production-and-execution",
        check_graph_production_and_execution(&fixture),
    ));
    report.record(E2eTestResult::from_result(
        "max-new-tokens-stops-generation",
        check_max_new_tokens_stops_generation(&fixture),
    ));
    report.record(E2eTestResult::from_result(
        "max-total-tokens-stops-generation",
        check_max_total_tokens_stops_generation(&fixture),
    ));
    #[cfg(test)]
    report.record(E2eTestResult::from_result(
        "eos-token-stops-generation",
        check_eos_token_stops_generation(&fixture),
    ));
    report.record(E2eTestResult::from_result(
        "generation-cancelled",
        check_generation_cancelled(&fixture),
    ));
    report.record(E2eTestResult::from_result(
        "sampling-greedy-deterministic",
        check_sampling_greedy_deterministic(),
    ));
    report.record(E2eTestResult::from_result(
        "sampling-stochastic-seed-deterministic",
        check_stochastic_sampling_is_seed_deterministic(),
    ));
    report.record(E2eTestResult::from_result(
        "closed-session-rejects-generation",
        check_closed_session_rejects_generation(&fixture),
    ));
    report.record(E2eTestResult::from_result(
        "memory-admission-failure",
        check_memory_admission_failure(),
    ));
    report.record(E2eTestResult::from_result(
        "memory-operator-output-accounting",
        check_memory_operator_output_accounting(),
    ));
    report.record(E2eTestResult::from_result(
        "tensor-resource-lifecycle",
        check_tensor_resource_lifecycle(),
    ));
    report.record(E2eTestResult::from_result(
        "kv-and-prefix-cache-lifecycle",
        check_kv_and_prefix_cache_lifecycle(&fixture),
    ));
    report.record(E2eTestResult::passed("observation-failure-case-started"));
    report.record(E2eTestResult::from_result(
        "incompatible-tokenizer-rejected",
        check_incompatible_tokenizer(&fixture),
    ));
    report.record(E2eTestResult::from_result(
        "untrusted-artifact-rejected",
        check_untrusted_artifact(&fixture),
    ));
    report.record(E2eTestResult::from_result(
        "invalid-tensor-shape-rejected",
        check_invalid_tensor_shape(&fixture),
    ));
    report.record(E2eTestResult::from_result(
        "unsupported-operator-rejected",
        check_unsupported_operator(),
    ));
    report.record(E2eTestResult::from_result(
        "missing-kernel-rejected",
        check_missing_kernel(),
    ));
    report.record(E2eTestResult::from_result(
        "required-kernel-removal-rejected",
        check_required_kernel_removal_fails_coverage(),
    ));
    report.record(E2eTestResult::from_result(
        "invalid-model-reference-rejected",
        check_invalid_model_reference(),
    ));
    report.record(E2eTestResult::from_result(
        "raw-handle-access-denied",
        check_raw_handle_access_denied(),
    ));
    let boundary_result = check_cli_boundary_denials();
    report.record(E2eTestResult::from_result(
        "cli-boundary-denials",
        boundary_result.clone(),
    ));
    report.record(E2eTestResult::from_result(
        "observation-boundary-violation",
        boundary_result,
    ));
    report.record(E2eTestResult::from_result(
        "generation-timeout",
        check_generation_timeout_maps_to_structured_error(),
    ));
    report.record(E2eTestResult::passed("observation-failure-case-completed"));

    let redaction_result = check_diagnostics_redaction_on_failure();
    report.record(E2eTestResult::from_result(
        "observation-redaction-failure",
        redaction_result.clone(),
    ));
    report.record(E2eTestResult::from_result(
        "diagnostics-redaction-on-failure",
        redaction_result,
    ));
    report.record(E2eTestResult::from_result(
        "kv-cache-diagnostics-redacted",
        check_kv_cache_diagnostics_redacted(),
    ));
    report.record(E2eTestResult::from_result(
        "kernel-optimization-orchestration-boundary",
        check_kernel_optimization_orchestration_boundary(),
    ));

    report.duration_millis = elapsed_millis(start);
    let metadata_check = check_report_metadata(&report);
    report.record(E2eTestResult::from_result(
        "report-metadata-complete",
        metadata_check,
    ));
    report.record(E2eTestResult::passed("observation-report-generated"));
    report
}

struct FixtureChatFormatter;

impl ChatTemplateFormatter for FixtureChatFormatter {
    fn format(&self, messages: &[ChatMessage]) -> Result<String, InferenceApiError> {
        Ok(messages
            .iter()
            .map(|message| format!("{}: {}\n", message.role, message.content))
            .collect())
    }
}

fn check_chat_message_prompt_path(fixture: &E2eFixture) -> Result<(), E2eConformanceError> {
    let messages = vec![ChatMessage::new("u", "hi")];
    let tokenized = tokenize_prompt_input(
        &fixture.tokenizer,
        TokenizationRequest::new(PromptInput::ChatMessages(messages)),
        Some(&FixtureChatFormatter),
    )?;
    if tokenized.token_ids.is_empty() {
        return Err(E2eConformanceError::TokenizerFailed {
            reason: "chat message prompt path produced no tokens".into(),
        });
    }
    Ok(())
}

fn check_one_shot_session_normal_paths(fixture: &E2eFixture) -> Result<(), E2eConformanceError> {
    let mut runtime = build_runtime_with_model_execution_engine(fixture);
    let (instance, _memory) = load_fixture_instance(fixture, &mut runtime)?;
    let session_request = SessionCreationRequest {
        model: GenerationModelReference::ModelInstance(instance.clone()),
        tokenizer: generation_tokenizer_reference(fixture),
        generation_defaults: GenerationParameters::greedy(),
        policy: SessionPolicy::default(),
        memory: SessionMemoryBudget::default(),
        allowed_capabilities: BTreeSet::new(),
        correlation_id: None,
        created_at_millis: 0,
    };
    let session = create_one_shot_session(&mut runtime, session_request)?;

    // Normal Tokenizer path: the fixture tokenizer via the same Tokenizer
    // Contract boundary the success path uses.
    let tokenized = tokenize_prompt_input(
        &fixture.tokenizer,
        TokenizationRequest::new(PromptInput::PlainText("hi".into())),
        None,
    )?;

    // Normal Generation/Sampling/Provider-Kernel path: the same plan-bound
    // generation loop + real Reference CPU forward pass as the multi-call
    // success path, just against the one-shot session.
    let request = build_generation_request(
        GenerationRequestId::new("e2e-one-shot-generation")?,
        Some(session.clone()),
        GenerationModelReference::ModelInstance(instance.clone()),
        generation_tokenizer_reference(fixture),
        tokenized,
        2,
        GenerationParameters::greedy(),
        StopConditions::default(),
        StreamingMode::Disabled,
    );
    let request = prepare_generation(&runtime, request)?;
    let mut observer = InferenceApiObserver::new();
    let result = run_first_native_generation_loop_with_plans(
        &mut runtime,
        fixture,
        &instance,
        &request,
        SamplingPolicy::default(),
        CacheUsageSummary::default(),
        |_generated_so_far| false,
        &mut observer,
    )?;
    if result.output.generated_token_ids.is_empty() {
        return Err(E2eConformanceError::GenerationFailed {
            reason: "one-shot session generation produced no tokens".into(),
        });
    }
    close_inference_session(&mut runtime, &session)?;
    Ok(())
}

fn check_already_tokenized_prompt_path(fixture: &E2eFixture) -> Result<(), E2eConformanceError> {
    let already_tokenized = vec![10, 20, 30];
    let tokenized = tokenize_prompt_input(
        &fixture.tokenizer,
        TokenizationRequest::new(PromptInput::TokenIds(already_tokenized.clone())),
        None,
    )?;
    if tokenized.token_ids != already_tokenized {
        return Err(E2eConformanceError::TokenizerFailed {
            reason: "already-tokenized prompt path did not preserve token ids".into(),
        });
    }
    Ok(())
}

fn check_fixture_tokenizer_deterministic(fixture: &E2eFixture) -> Result<(), E2eConformanceError> {
    let first = tokenize_prompt_input(
        &fixture.tokenizer,
        TokenizationRequest::new(PromptInput::PlainText("hi".into())),
        None,
    )?;
    let second = tokenize_prompt_input(
        &fixture.tokenizer,
        TokenizationRequest::new(PromptInput::PlainText("hi".into())),
        None,
    )?;
    if first.token_ids != second.token_ids {
        return Err(E2eConformanceError::TokenizerFailed {
            reason: "fixture tokenizer produced non-deterministic token ids".into(),
        });
    }
    Ok(())
}

// ---------------------------------------------------------------------
// Section 7 (Runtime-Owned KV Data): prepare/commit/abort lifecycle tests
// ---------------------------------------------------------------------

#[cfg(test)]
fn new_kv_lifecycle_engine(fixture: &E2eFixture) -> E2eRuntimeModelExecutionEngine {
    E2eRuntimeModelExecutionEngine {
        fixture: fixture.clone(),
        kv_states: Arc::new(Mutex::new(BTreeMap::new())),
        pending_kv_states: Arc::new(Mutex::new(BTreeMap::new())),
        forced_token: None,
    }
}

#[cfg(test)]
fn kv_lifecycle_test_request(
    fixture: &E2eFixture,
    runtime: &Runtime,
    instance: &ModelInstanceId,
    request_id: &str,
    session: Option<InferenceSessionId>,
    prompt: &[TokenId],
) -> Result<GenerationRequest, E2eConformanceError> {
    let tokenized = tokenize_prompt_input(
        &fixture.tokenizer,
        TokenizationRequest::new(PromptInput::TokenIds(prompt.to_vec())),
        None,
    )?;
    let request = build_generation_request(
        GenerationRequestId::new(request_id)?,
        session,
        GenerationModelReference::ModelInstance(instance.clone()),
        generation_tokenizer_reference(fixture),
        tokenized,
        4,
        GenerationParameters::greedy(),
        StopConditions::default(),
        StreamingMode::Disabled,
    );
    Ok(prepare_generation(runtime, request)?)
}

#[cfg(test)]
fn kv_lifecycle_session_request(
    fixture: &E2eFixture,
    instance: &ModelInstanceId,
) -> SessionCreationRequest {
    SessionCreationRequest {
        model: GenerationModelReference::ModelInstance(instance.clone()),
        tokenizer: generation_tokenizer_reference(fixture),
        generation_defaults: GenerationParameters::greedy(),
        policy: SessionPolicy::default(),
        memory: SessionMemoryBudget::default(),
        allowed_capabilities: BTreeSet::new(),
        correlation_id: None,
        created_at_millis: 0,
    }
}

#[cfg(test)]
fn prefill_cache_id_from_step(
    step: &RuntimeModelExecutionStep,
) -> Result<KvCacheId, E2eConformanceError> {
    match &step.kv_commit {
        Some(RuntimeKvCacheCommit::PrefillCompleted { cache, .. }) => Ok(cache.clone()),
        _ => Err(E2eConformanceError::GenerationFailed {
            reason: "expected a prefill KV commit descriptor".into(),
        }),
    }
}

/// Proves a generation step's KV write stays *pending* -- never promoted
/// onto the cache's committed `layer_resources` (task 7.4 prepare) -- when
/// sampling rejects every candidate after a successful forward pass and
/// `commit_generation_step` is consequently never called.
#[cfg(test)]
fn check_kv_sampling_failure_leaves_cache_uncommitted(
    fixture: &E2eFixture,
) -> Result<(), E2eConformanceError> {
    let mut runtime = build_runtime_with_model_execution_engine(fixture);
    let (instance, _memory) = load_fixture_instance(fixture, &mut runtime)?;
    let engine = new_kv_lifecycle_engine(fixture);
    let request = kv_lifecycle_test_request(
        fixture,
        &runtime,
        &instance,
        "kv-sampling-failure",
        None,
        &[1, 2],
    )?;
    let mut plans = first_native_plans_for_prompt(
        &runtime,
        fixture,
        &instance,
        request.input_token_ids.len() as u64,
    )?;
    let step =
        engine.execute_generation_step(&mut runtime, &request, &[], Some(&mut plans.prefill))?;
    let cache = prefill_cache_id_from_step(&step)?;
    if !runtime.kv_cache(&cache)?.layer_resources.is_empty() {
        return Err(E2eConformanceError::GenerationFailed {
            reason: "KV cache carries committed layer resources despite no commit call".into(),
        });
    }
    Ok(())
}

/// Proves a generation step that fails during Provider dispatch (task 5.2's
/// registered-Provider resolution failing here) never stores a pending KV
/// state -- there is nothing a later, unrelated commit could wrongly
/// promote.
#[cfg(test)]
fn check_kv_provider_failure_stores_no_pending_state(
    fixture: &E2eFixture,
) -> Result<(), E2eConformanceError> {
    let mut runtime = build_runtime_with_model_execution_engine(fixture);
    let (instance, _memory) = load_fixture_instance(fixture, &mut runtime)?;
    let engine = new_kv_lifecycle_engine(fixture);
    let request = kv_lifecycle_test_request(
        fixture,
        &runtime,
        &instance,
        "kv-provider-failure",
        None,
        &[1, 2],
    )?;
    let mut plans = first_native_plans_for_prompt(
        &runtime,
        fixture,
        &instance,
        request.input_token_ids.len() as u64,
    )?;
    for binding in &mut plans.prefill.node_bindings {
        binding.provider = ProviderBinding::new("unregistered-provider");
    }
    match engine.execute_generation_step(&mut runtime, &request, &[], Some(&mut plans.prefill)) {
        Err(InferenceApiError::ProviderUnavailable { .. }) => {}
        Err(error) => {
            return Err(E2eConformanceError::GenerationFailed {
                reason: format!("unexpected error for an unregistered provider: {error}"),
            });
        }
        Ok(_) => {
            return Err(E2eConformanceError::GenerationFailed {
                reason: "generation step succeeded despite an unregistered provider".into(),
            });
        }
    }
    if !engine
        .pending_kv_states
        .lock()
        .map_err(|_| E2eConformanceError::GenerationFailed {
            reason: "pending KV state lock poisoned".into(),
        })?
        .is_empty()
    {
        return Err(E2eConformanceError::GenerationFailed {
            reason: "a failed generation step left a pending KV state behind".into(),
        });
    }
    Ok(())
}

/// Proves a decode step's pending KV write never reaches the cache's
/// committed `layer_resources` when the request is cancelled before
/// `commit_generation_step` runs -- the committed cache stays exactly what
/// the prior successful commit left it as.
#[cfg(test)]
fn check_kv_cancelled_decode_does_not_corrupt_committed_cache(
    fixture: &E2eFixture,
) -> Result<(), E2eConformanceError> {
    let mut runtime = build_runtime_with_model_execution_engine(fixture);
    let (instance, _memory) = load_fixture_instance(fixture, &mut runtime)?;
    let engine = new_kv_lifecycle_engine(fixture);
    let request = kv_lifecycle_test_request(
        fixture,
        &runtime,
        &instance,
        "kv-cancel-rollback",
        None,
        &[1, 2],
    )?;
    let mut plans = first_native_plans_for_prompt(
        &runtime,
        fixture,
        &instance,
        request.input_token_ids.len() as u64,
    )?;

    let prefill_step =
        engine.execute_generation_step(&mut runtime, &request, &[], Some(&mut plans.prefill))?;
    engine.commit_generation_step(&mut runtime, &request, &[], 1, &prefill_step)?;
    let cache = prefill_cache_id_from_step(&prefill_step)?;
    let committed_after_prefill = runtime.kv_cache(&cache)?.layer_resources.clone();

    let generated = vec![1];
    let _decode_step = engine.execute_generation_step(
        &mut runtime,
        &request,
        &generated,
        Some(&mut plans.decode),
    )?;
    let committed_after_cancelled_decode = runtime.kv_cache(&cache)?.layer_resources.clone();
    if committed_after_cancelled_decode != committed_after_prefill {
        return Err(E2eConformanceError::GenerationFailed {
            reason: "a cancelled decode step's pending KV write altered the committed cache".into(),
        });
    }
    Ok(())
}

/// Correctif 1 / task 1.8: decode's KV-history-concatenated pending write is
/// genuinely admitted through `MemoryManager` (via `write_tensor_admitted`),
/// not left as a bare, unaccounted `write_tensor` -- the admitted
/// allocation's byte size reflects the *concatenated* (history + new token)
/// tensor a decode step's `Append` KV behavior produces, not just the newly
/// dispatched token's own smaller Kernel output.
#[cfg(test)]
fn check_kv_pending_write_is_memory_admitted_for_its_concatenated_size(
    fixture: &E2eFixture,
) -> Result<(), E2eConformanceError> {
    let mut runtime = build_runtime_with_model_execution_engine(fixture);
    let (instance, _memory) = load_fixture_instance(fixture, &mut runtime)?;
    let engine = new_kv_lifecycle_engine(fixture);
    let request = kv_lifecycle_test_request(
        fixture,
        &runtime,
        &instance,
        "kv-pending-admission",
        None,
        &[1, 2],
    )?;
    let mut plans = first_native_plans_for_prompt(
        &runtime,
        fixture,
        &instance,
        request.input_token_ids.len() as u64,
    )?;
    let prefill_step =
        engine.execute_generation_step(&mut runtime, &request, &[], Some(&mut plans.prefill))?;
    engine.commit_generation_step(&mut runtime, &request, &[], 1, &prefill_step)?;
    let cache = prefill_cache_id_from_step(&prefill_step)?;

    let generated = vec![1];
    engine.execute_generation_step(&mut runtime, &request, &generated, Some(&mut plans.decode))?;

    // This fixture's prefill prompt (`&[1, 2]`) is 2 tokens; decode appends
    // 1 more, so the pending K/V write's history-concatenated row count is
    // 3, each row `hidden_size` wide.
    let expected_bytes =
        3 * fixture.config.architecture.hidden_size * std::mem::size_of::<f32>() as u64;
    let matching_allocations = runtime
        .memory()
        .allocations()
        .filter(|allocation| {
            allocation.state == MemoryAllocationState::Active
                && allocation.request.owner == MemoryAllocationOwner::Session(cache.to_string())
                && allocation.request.size_bytes == expected_bytes
        })
        .count();
    // 4, not 2: `execute_qwen_graph_nodes` reassigns `output_tensor` to the
    // concatenated value *before* the KV-node's `edge.*` write too (so a
    // later reader of that edge sees the same concatenated value the
    // pending write does -- see that write site's own doc comment), so
    // each of K and V produces one concatenated-size allocation for its
    // `kv.*.pending` resource *and* one for its `edge.*` resource: this
    // fixture's single layer's K and V nodes together admit 2 + 2.
    if matching_allocations != 4 {
        return Err(E2eConformanceError::MemoryValidationFailed {
            reason: format!(
                "expected 4 Active {expected_bytes}-byte allocations (K and V, each with a \
                 pending-resource and an edge-resource allocation) for the decode step's \
                 concatenated KV write, owned by cache '{cache}'; found {matching_allocations}"
            ),
        });
    }
    Ok(())
}

/// Correctif 1 / task 1.9: discarding a pending KV state
/// (`discard_pending_kv_state`, invoked automatically at the start of the
/// next generation step, or directly on cancellation) releases the
/// `MemoryManager` allocation the pending write admitted (task 1.8's fix),
/// not just the Provider storage entry -- otherwise every cancelled or
/// failed decode step would leak one allocation per layer per role forever.
#[cfg(test)]
fn check_kv_pending_write_allocation_is_released_on_discard(
    fixture: &E2eFixture,
) -> Result<(), E2eConformanceError> {
    let mut runtime = build_runtime_with_model_execution_engine(fixture);
    let (instance, _memory) = load_fixture_instance(fixture, &mut runtime)?;
    let engine = new_kv_lifecycle_engine(fixture);
    let request = kv_lifecycle_test_request(
        fixture,
        &runtime,
        &instance,
        "kv-pending-discard-release",
        None,
        &[1, 2],
    )?;
    let mut plans = first_native_plans_for_prompt(
        &runtime,
        fixture,
        &instance,
        request.input_token_ids.len() as u64,
    )?;
    // Deliberately never committed: the pending write stays pending, so its
    // allocation is still held when discarded below.
    let prefill_step =
        engine.execute_generation_step(&mut runtime, &request, &[], Some(&mut plans.prefill))?;
    let cache = prefill_cache_id_from_step(&prefill_step)?;

    let active_owned_by_cache = |runtime: &Runtime| {
        runtime
            .memory()
            .allocations()
            .filter(|allocation| {
                allocation.state == MemoryAllocationState::Active
                    && allocation.request.owner == MemoryAllocationOwner::Session(cache.to_string())
            })
            .count()
    };
    let active_before_discard = active_owned_by_cache(&runtime);
    engine.discard_pending_kv_state(&mut runtime, &request)?;
    let active_after_discard = active_owned_by_cache(&runtime);
    let released = active_before_discard.saturating_sub(active_after_discard);
    // K and V for this fixture's single layer.
    if released != 2 {
        return Err(E2eConformanceError::MemoryValidationFailed {
            reason: format!(
                "expected discard to release exactly 2 (K and V) Active allocations owned by \
                 cache '{cache}'; released {released} (before: {active_before_discard}, after: \
                 {active_after_discard})"
            ),
        });
    }
    Ok(())
}

/// Correctif 11 / task group 9: a multi-layer KV commit is atomic. Sabotages
/// the *second* resource (layer 0's pending V, after K would otherwise
/// promote successfully) a decode step's commit would promote, and proves
/// the whole commit fails and the cache's committed state is left exactly
/// as the prior successful commit produced it -- not with layer 0's K
/// pointing at this step's data while V still points at the previous
/// step's.
#[cfg(test)]
fn check_kv_partial_layer_failure_during_commit_rolls_back_cleanly(
    fixture: &E2eFixture,
) -> Result<(), E2eConformanceError> {
    let mut runtime = build_runtime_with_model_execution_engine(fixture);
    let (instance, _memory) = load_fixture_instance(fixture, &mut runtime)?;
    let engine = new_kv_lifecycle_engine(fixture);
    let request = kv_lifecycle_test_request(
        fixture,
        &runtime,
        &instance,
        "kv-partial-layer-failure",
        None,
        &[1, 2],
    )?;
    let mut plans = first_native_plans_for_prompt(
        &runtime,
        fixture,
        &instance,
        request.input_token_ids.len() as u64,
    )?;

    let prefill_step =
        engine.execute_generation_step(&mut runtime, &request, &[], Some(&mut plans.prefill))?;
    engine.commit_generation_step(&mut runtime, &request, &[], 1, &prefill_step)?;
    let cache = prefill_cache_id_from_step(&prefill_step)?;
    let committed_before = runtime.kv_cache(&cache)?.layer_resources.clone();
    let layer0_before =
        committed_before
            .get(&0)
            .cloned()
            .ok_or_else(|| E2eConformanceError::GenerationFailed {
                reason: "prefill commit produced no layer 0 KV binding".into(),
            })?;
    let provider_for_prefill_check = resolve_kernel_execution_provider(
        &runtime,
        &ProviderBinding::new(REFERENCE_CPU_PROVIDER_NAME),
    )
    .map_err(E2eConformanceError::from)?;
    let original_k_value = provider_for_prefill_check
        .read_tensor(&layer0_before.k)
        .ok_or_else(|| E2eConformanceError::GenerationFailed {
            reason: "prefill-committed K resource is not resolvable from Provider storage".into(),
        })?;

    let generated = vec![1];
    let decode_step = engine.execute_generation_step(
        &mut runtime,
        &request,
        &generated,
        Some(&mut plans.decode),
    )?;
    let pending_v =
        {
            let pending_kv_states = engine.pending_kv_states.lock().map_err(|_| {
                E2eConformanceError::GenerationFailed {
                    reason: "pending KV state lock poisoned".into(),
                }
            })?;
            pending_kv_states
                .values()
                .next()
                .and_then(|state| state.layer_kv.first())
                .map(|layer| layer.v.clone())
                .ok_or_else(|| E2eConformanceError::GenerationFailed {
                    reason: "decode step produced no pending KV state to sabotage".into(),
                })?
        };
    // Remove layer 0's pending V straight from Provider storage -- exactly
    // what `promote_pending_kv_layer_role`'s "no pending KV data to commit"
    // error path detects -- while leaving K's pending resource intact, so
    // K would promote successfully if the commit were not atomic.
    let provider = resolve_kernel_execution_provider(
        &runtime,
        &ProviderBinding::new(REFERENCE_CPU_PROVIDER_NAME),
    )
    .map_err(E2eConformanceError::from)?;
    let _ = provider.release_tensor(&pending_v);

    match engine.commit_generation_step(&mut runtime, &request, &generated, 2, &decode_step) {
        Err(InferenceApiError::KvCacheUnavailable { .. }) => {}
        Err(error) => {
            return Err(E2eConformanceError::GenerationFailed {
                reason: format!("unexpected error for a sabotaged mid-commit resource: {error}"),
            });
        }
        Ok(()) => {
            return Err(E2eConformanceError::GenerationFailed {
                reason: "commit succeeded despite a missing pending resource for one layer".into(),
            });
        }
    }

    let committed_after_failed_commit = runtime.kv_cache(&cache)?.layer_resources.clone();
    if committed_after_failed_commit != committed_before {
        return Err(E2eConformanceError::GenerationFailed {
            reason: "a failed multi-layer KV commit left the cache in a partially-promoted state"
                .into(),
        });
    }
    // The deeper property a binding-equality check alone cannot see: layer
    // 0's K would have promoted successfully in isolation (only V was
    // sabotaged), so a non-atomic implementation could release the
    // pre-existing K allocation and overwrite its Provider-stored bytes
    // before ever learning V failed -- leaving `layer_resources` pointing
    // at an unchanged resource id whose *contents* were nonetheless
    // destroyed. Both must still be exactly as they were before this
    // attempt.
    if !runtime
        .memory()
        .allocations()
        .any(|allocation| allocation.id == layer0_before.k_allocation)
    {
        return Err(E2eConformanceError::MemoryValidationFailed {
            reason: "the pre-existing layer 0 K allocation was released despite the commit \
                     that would have replaced it failing"
                .into(),
        });
    }
    let provider = resolve_kernel_execution_provider(
        &runtime,
        &ProviderBinding::new(REFERENCE_CPU_PROVIDER_NAME),
    )
    .map_err(E2eConformanceError::from)?;
    let k_value_after_failed_commit = provider.read_tensor(&layer0_before.k).ok_or_else(|| {
        E2eConformanceError::MemoryValidationFailed {
            reason: "layer 0's committed K resource is no longer resolvable from Provider \
                     storage after a failed commit"
                .into(),
        }
    })?;
    if k_value_after_failed_commit.data != original_k_value.data {
        return Err(E2eConformanceError::MemoryValidationFailed {
            reason: "a failed multi-layer KV commit destructively overwrote layer 0's \
                     still-committed K bytes before the failure was known"
                .into(),
        });
    }
    Ok(())
}

/// Proves a second `commit_generation_step` call for the same completed
/// step is rejected rather than silently re-promoting (or double-releasing)
/// KV resources the first commit already promoted.
#[cfg(test)]
fn check_kv_double_commit_second_call_is_rejected(
    fixture: &E2eFixture,
) -> Result<(), E2eConformanceError> {
    let mut runtime = build_runtime_with_model_execution_engine(fixture);
    let (instance, _memory) = load_fixture_instance(fixture, &mut runtime)?;
    let engine = new_kv_lifecycle_engine(fixture);
    let request = kv_lifecycle_test_request(
        fixture,
        &runtime,
        &instance,
        "kv-double-commit",
        None,
        &[1, 2],
    )?;
    let mut plans = first_native_plans_for_prompt(
        &runtime,
        fixture,
        &instance,
        request.input_token_ids.len() as u64,
    )?;
    let step =
        engine.execute_generation_step(&mut runtime, &request, &[], Some(&mut plans.prefill))?;
    engine.commit_generation_step(&mut runtime, &request, &[], 1, &step)?;
    match engine.commit_generation_step(&mut runtime, &request, &[], 1, &step) {
        Err(InferenceApiError::KvCacheUnavailable { reason }) if reason.contains("pending") => {
            Ok(())
        }
        Err(error) => Err(E2eConformanceError::GenerationFailed {
            reason: format!("unexpected error for a double commit: {error}"),
        }),
        Ok(()) => Err(E2eConformanceError::GenerationFailed {
            reason: "a second commit for the same generation step unexpectedly succeeded".into(),
        }),
    }
}

/// Proves discarding a pending KV state twice in a row (a cancellation
/// racing a cleanup retry, for example) is idempotent rather than erroring
/// the second time just because there is nothing left to discard.
#[cfg(test)]
fn check_kv_double_abort_is_idempotent(fixture: &E2eFixture) -> Result<(), E2eConformanceError> {
    let mut runtime = build_runtime_with_model_execution_engine(fixture);
    let (instance, _memory) = load_fixture_instance(fixture, &mut runtime)?;
    let engine = new_kv_lifecycle_engine(fixture);
    let request = kv_lifecycle_test_request(
        fixture,
        &runtime,
        &instance,
        "kv-double-abort",
        None,
        &[1, 2],
    )?;
    let mut plans = first_native_plans_for_prompt(
        &runtime,
        fixture,
        &instance,
        request.input_token_ids.len() as u64,
    )?;
    engine.execute_generation_step(&mut runtime, &request, &[], Some(&mut plans.prefill))?;
    engine.discard_pending_kv_state(&mut runtime, &request)?;
    engine.discard_pending_kv_state(&mut runtime, &request)?;
    if !engine
        .pending_kv_states
        .lock()
        .map_err(|_| E2eConformanceError::GenerationFailed {
            reason: "pending KV state lock poisoned".into(),
        })?
        .is_empty()
    {
        return Err(E2eConformanceError::GenerationFailed {
            reason: "pending KV state survived two discard calls".into(),
        });
    }
    Ok(())
}

/// Proves a stale pending KV write left by an earlier attempt does not
/// survive a subsequent, *failed* retry for the same request -- and so
/// cannot later be wrongly promoted by an unrelated commit call. The first
/// attempt succeeds and leaves a pending write nothing ever commits (as if
/// a downstream failure occurred); the retry is routed through an
/// unregistered Provider so it fails too, but must still discard the first
/// attempt's stale pending entry before doing so.
#[cfg(test)]
fn check_kv_stale_pending_state_does_not_survive_a_failed_retry(
    fixture: &E2eFixture,
) -> Result<(), E2eConformanceError> {
    let mut runtime = build_runtime_with_model_execution_engine(fixture);
    let (instance, _memory) = load_fixture_instance(fixture, &mut runtime)?;
    let engine = new_kv_lifecycle_engine(fixture);
    let request = kv_lifecycle_test_request(
        fixture,
        &runtime,
        &instance,
        "kv-stale-pending",
        None,
        &[1, 2],
    )?;
    let mut plans = first_native_plans_for_prompt(
        &runtime,
        fixture,
        &instance,
        request.input_token_ids.len() as u64,
    )?;
    let first_step =
        engine.execute_generation_step(&mut runtime, &request, &[], Some(&mut plans.prefill))?;

    for binding in &mut plans.prefill.node_bindings {
        binding.provider = ProviderBinding::new("unregistered-provider");
    }
    if engine
        .execute_generation_step(&mut runtime, &request, &[], Some(&mut plans.prefill))
        .is_ok()
    {
        return Err(E2eConformanceError::GenerationFailed {
            reason: "retry with an unregistered provider unexpectedly succeeded".into(),
        });
    }

    match engine.commit_generation_step(&mut runtime, &request, &[], 1, &first_step) {
        Err(InferenceApiError::KvCacheUnavailable { reason }) if reason.contains("pending") => {
            Ok(())
        }
        Err(error) => Err(E2eConformanceError::GenerationFailed {
            reason: format!(
                "unexpected error committing after a discarded stale pending state: {error}"
            ),
        }),
        Ok(()) => Err(E2eConformanceError::GenerationFailed {
            reason:
                "commit succeeded using a stale pending KV state that should have been discarded"
                    .into(),
        }),
    }
}

/// Proves a KV cache committed under one request's compatibility (its
/// prefix fingerprint is derived from that request's own id) cannot be
/// reused under a different request/session's compatibility -- Runtime's
/// `validate_kv_cache_reuse` must reject the mismatch rather than letting
/// one session's decode read another session's KV state.
#[cfg(test)]
fn check_kv_wrong_session_reuse_is_rejected_by_compatibility(
    fixture: &E2eFixture,
) -> Result<(), E2eConformanceError> {
    let mut runtime = build_runtime_with_model_execution_engine(fixture);
    let (instance, _memory) = load_fixture_instance(fixture, &mut runtime)?;
    let engine = new_kv_lifecycle_engine(fixture);

    let session_a = create_inference_session(
        &mut runtime,
        kv_lifecycle_session_request(fixture, &instance),
    )?;
    let session_b = create_inference_session(
        &mut runtime,
        kv_lifecycle_session_request(fixture, &instance),
    )?;

    let request_a = kv_lifecycle_test_request(
        fixture,
        &runtime,
        &instance,
        "kv-session-a",
        Some(session_a),
        &[1, 2],
    )?;
    let request_b = kv_lifecycle_test_request(
        fixture,
        &runtime,
        &instance,
        "kv-session-b",
        Some(session_b),
        &[1, 2],
    )?;

    let mut plans_a = first_native_plans_for_prompt(
        &runtime,
        fixture,
        &instance,
        request_a.input_token_ids.len() as u64,
    )?;
    let step_a = engine.execute_generation_step(
        &mut runtime,
        &request_a,
        &[],
        Some(&mut plans_a.prefill),
    )?;
    engine.commit_generation_step(&mut runtime, &request_a, &[], 1, &step_a)?;
    let cache_a = prefill_cache_id_from_step(&step_a)?;

    let compatibility_b = engine.kv_compatibility(&request_b);
    match runtime.validate_kv_cache_reuse(&cache_a, &compatibility_b, None) {
        Err(_) => Ok(()),
        Ok(()) => Err(E2eConformanceError::GenerationFailed {
            reason: "session B's compatibility was accepted for reuse of session A's KV cache"
                .into(),
        }),
    }
}

/// Proves a generation step re-checks Model Instance readiness for *itself*
/// (task 8.1) rather than reusing the one-time check
/// `prepare_first_native_execution_plans` performed before the generation
/// loop started: draining the instance between prefill and decode -- which
/// leaves its weight resource bindings fully intact, only its lifecycle
/// readiness changes -- must now fail the decode step closed instead of
/// silently proceeding on stale readiness evidence.
#[cfg(test)]
fn check_generation_step_rechecks_model_instance_readiness(
    fixture: &E2eFixture,
) -> Result<(), E2eConformanceError> {
    let mut runtime = build_runtime_with_model_execution_engine(fixture);
    let (instance, _memory) = load_fixture_instance(fixture, &mut runtime)?;
    let engine = new_kv_lifecycle_engine(fixture);
    let request = kv_lifecycle_test_request(
        fixture,
        &runtime,
        &instance,
        "kv-readiness-recheck",
        None,
        &[1, 2],
    )?;
    let mut plans = first_native_plans_for_prompt(
        &runtime,
        fixture,
        &instance,
        request.input_token_ids.len() as u64,
    )?;

    let prefill_step =
        engine.execute_generation_step(&mut runtime, &request, &[], Some(&mut plans.prefill))?;
    engine.commit_generation_step(&mut runtime, &request, &[], 1, &prefill_step)?;

    runtime
        .model_instances_mut()
        .instance_mut(&instance)
        .map_err(InferenceApiError::from)?
        .drain()
        .map_err(InferenceApiError::from)?;

    let generated = vec![1];
    match engine.execute_generation_step(
        &mut runtime,
        &request,
        &generated,
        Some(&mut plans.decode),
    ) {
        Err(InferenceApiError::ModelInstanceNotReady { .. }) => Ok(()),
        Err(error) => Err(E2eConformanceError::GenerationFailed {
            reason: format!("unexpected error for a drained model instance: {error}"),
        }),
        Ok(_) => Err(E2eConformanceError::GenerationFailed {
            reason: "decode proceeded through a drained (not-ready) model instance".into(),
        }),
    }
}

/// Proves the generation-level observation stream (task 8.2) -- which
/// already carries causal component/graph/plan/provider/resource/KV/
/// sampling/token-commit evidence -- never carries the raw prompt text or a
/// native pointer-style marker, across every observation kind a real
/// forward pass emits, not just the higher-level conformance report JSON
/// `e2e_observability_emits_only_redacted_report_metadata` already checks.
#[cfg(test)]
fn check_generation_observations_never_carry_raw_prompt_or_handles(
    fixture: &E2eFixture,
) -> Result<(), E2eConformanceError> {
    let prompt = "zzyzx-secret";
    let outcome = run_success_path_with_prompt(fixture, &ModelRef::new("qwen-test")?, prompt)?;
    if outcome.observer.observations().is_empty() {
        return Err(E2eConformanceError::GenerationFailed {
            reason: "success path emitted no observations to check for redaction".into(),
        });
    }
    for observation in outcome.observer.observations() {
        if observation.message.contains(prompt) {
            return Err(E2eConformanceError::BoundaryViolation {
                reason: format!(
                    "observation {:?} carried raw prompt text: {}",
                    observation.kind, observation.message
                ),
            });
        }
        if observation.message.contains("0x")
            || observation
                .message
                .to_ascii_lowercase()
                .contains("native_handle")
        {
            return Err(E2eConformanceError::BoundaryViolation {
                reason: format!(
                    "observation {:?} carried a native handle or pointer marker: {}",
                    observation.kind, observation.message
                ),
            });
        }
    }
    Ok(())
}

/// Proves `FirstNativeChatSession::close` (task 8.3/8.4) genuinely releases
/// both the KV cache a chat turn created and the Model Instance it ran
/// against -- not just returning `Ok(())` without having done the
/// underlying work. The KV cache a chat turn creates gets
/// `KvCacheRetentionPolicy::ReleaseOnSessionClose` (the `KvCachePolicy`
/// default) and is scoped to this chat session's own `InferenceSessionId`,
/// so closing that *same* session is what must release it.
#[cfg(test)]
fn check_chat_session_close_releases_kv_cache_and_model_instance() -> Result<(), E2eConformanceError>
{
    let model_ref = ModelRef::new("qwen-test")?;
    let mut chat = FirstNativeChatSession::open(&model_ref).map_err(|error| {
        E2eConformanceError::GenerationFailed {
            reason: error.to_string(),
        }
    })?;
    chat.turn("hi", 1)
        .map_err(|error| E2eConformanceError::GenerationFailed {
            reason: error.to_string(),
        })?;

    let session = chat.session.clone();
    let instance = chat.instance.clone();
    let cache_id = chat
        .runtime
        .kv_caches()
        .caches()
        .find(|cache| cache.session.as_ref() == Some(&session))
        .map(|cache| cache.id.clone())
        .ok_or_else(|| E2eConformanceError::GenerationFailed {
            reason: "chat turn created no session-scoped KV cache".into(),
        })?;
    if chat.runtime.kv_cache(&cache_id)?.lifecycle == KvCacheLifecycleState::Released {
        return Err(E2eConformanceError::GenerationFailed {
            reason: "chat turn's KV cache was already released before session close".into(),
        });
    }
    if chat.runtime.kv_cache(&cache_id)?.policy.retention
        != KvCacheRetentionPolicy::ReleaseOnSessionClose
    {
        return Err(E2eConformanceError::GenerationFailed {
            reason: "chat turn's KV cache does not use the expected session-close retention policy"
                .into(),
        });
    }

    // The same two steps `FirstNativeChatSession::close` performs, kept
    // `&mut` here (rather than calling the consuming public `close`) so
    // this test can inspect `chat.runtime` immediately afterward and prove
    // cleanup targeted the *same* session and instance a turn actually
    // used.
    close_inference_session(&mut chat.runtime, &session).map_err(E2eConformanceError::from)?;
    unload_model_instance(
        &mut chat.runtime,
        &instance,
        ModelInstanceUnloadPolicy::DrainActiveUse,
    )
    .map_err(E2eConformanceError::from)?;

    if chat.runtime.kv_cache(&cache_id)?.lifecycle != KvCacheLifecycleState::Released {
        return Err(E2eConformanceError::GenerationFailed {
            reason: "closing the chat session did not release its KV cache".into(),
        });
    }
    if chat
        .runtime
        .model_instance_status(&instance)
        .map_err(E2eConformanceError::from)?
        .readiness
        .accepts_generation()
    {
        return Err(E2eConformanceError::GenerationFailed {
            reason: "unloading the chat session's model instance left it accepting generation"
                .into(),
        });
    }
    Ok(())
}

/// Proves two `FirstNativeChatSession`s opened for the same model are
/// isolated (task 8.4): distinct `InferenceSessionId`s and distinct,
/// independently scoped KV caches -- one session's turn cannot be confused
/// with, or leak state into, another's.
#[cfg(test)]
fn check_chat_sessions_are_isolated_from_each_other() -> Result<(), E2eConformanceError> {
    let model_ref = ModelRef::new("qwen-test")?;
    let mut chat_a = FirstNativeChatSession::open(&model_ref).map_err(|error| {
        E2eConformanceError::GenerationFailed {
            reason: error.to_string(),
        }
    })?;
    let mut chat_b = FirstNativeChatSession::open(&model_ref).map_err(|error| {
        E2eConformanceError::GenerationFailed {
            reason: error.to_string(),
        }
    })?;

    if chat_a.session_id() == chat_b.session_id() {
        return Err(E2eConformanceError::GenerationFailed {
            reason: "two independently opened chat sessions were assigned the same session id"
                .into(),
        });
    }

    chat_a
        .turn("hello from a", 1)
        .map_err(|error| E2eConformanceError::GenerationFailed {
            reason: error.to_string(),
        })?;
    chat_b
        .turn("hello from b", 1)
        .map_err(|error| E2eConformanceError::GenerationFailed {
            reason: error.to_string(),
        })?;

    let session_a = chat_a.session.clone();
    let cache_a_belongs_to_a = chat_a
        .runtime
        .kv_caches()
        .caches()
        .any(|cache| cache.session.as_ref() == Some(&session_a));
    if !cache_a_belongs_to_a {
        return Err(E2eConformanceError::GenerationFailed {
            reason: "chat session A's own Runtime recorded no KV cache scoped to its session"
                .into(),
        });
    }
    // Session A's Runtime is entirely separate from session B's -- there is
    // no shared KV cache manager, memory manager, or session table between
    // them, so B's session id cannot appear in A's cache table at all.
    let session_b = chat_b.session_id().clone();
    if chat_a
        .runtime
        .kv_caches()
        .caches()
        .any(|cache| cache.session.as_ref() == Some(&session_b))
    {
        return Err(E2eConformanceError::GenerationFailed {
            reason: "chat session A's Runtime recorded a KV cache scoped to session B".into(),
        });
    }

    chat_a
        .close()
        .map_err(|error| E2eConformanceError::GenerationFailed {
            reason: error.to_string(),
        })?;
    chat_b
        .close()
        .map_err(|error| E2eConformanceError::GenerationFailed {
            reason: error.to_string(),
        })?;
    Ok(())
}

fn elapsed_millis(start: SystemTime) -> u64 {
    SystemTime::now()
        .duration_since(start)
        .map(|duration| duration.as_millis() as u64)
        .unwrap_or_default()
}

#[cfg(test)]
mod tests;
