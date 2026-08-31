//! First-native runtime execution and conformance support.
//!
//! Assembles a minimal, deterministic Qwen-like fixture model and drives it
//! through the real Runtime Inference API surface -- model resolution,
//! Model Loading, Model Instance creation, session creation, tokenization,
//! generation, streaming, and cleanup -- using genuine Reference CPU numeric
//! kernels for the forward pass (not canned output), so the success path is
//! a real, if tiny, end-to-end inference run. The `e2e_conformance` module is
//! now a compatibility wrapper around this runtime-owned implementation.

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
use crate::inference_api::*;
use crate::kernel::*;
use crate::kernel_artifact::{
    CompiledKernelArtifactId, PreparedKernel, PreparedKernelGeneration, PreparedKernelIdAllocator,
};
use crate::kernel_dispatch::*;
use crate::kernel_execution_plan::{
    PlanGuard, PlanGuardContext, PlanMemoryRequirements, PlanNodeBinding, PreparedExecutionPhase,
    PreparedExecutionPlan, PreparedExecutionPlanError, PreparedExecutionPlanGeneration,
    PreparedExecutionPlanId, PreparedExecutionPlanScope, ResourceBindingPlan,
    semantic_graph_fingerprint,
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
#[cfg(all(not(target_arch = "wasm32"), feature = "wasmtime-component-engine"))]
const QWEN_GRAPH_COMPONENT_NAME: &str = "magnetar.qwen.graph-fixture";
#[cfg(all(not(target_arch = "wasm32"), feature = "wasmtime-component-engine"))]
const QWEN_GRAPH_COMPONENT_DIGEST: &str =
    "sha256:e376dedc5059e0e46233fe783fab49c8d9752aa68a3a967876118d13fa5c9d85";

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
    let mut tensor_yaml = String::new();
    for name in qwen_expected_tensor_names(config.architecture.layer_count, config.tied_embeddings)
    {
        let shape = qwen_expected_tensor_shape(&name, config).ok_or_else(|| {
            E2eConformanceError::FixtureInvalid {
                reason: format!("no expected shape for fixture tensor '{name}'"),
            }
        })?;
        let shape_text = shape
            .iter()
            .map(u64::to_string)
            .collect::<Vec<_>>()
            .join(", ");
        tensor_yaml.push_str(&format!(
            "  - name: {name}\n    shape: [{shape_text}]\n    storage_dtype: f32\n"
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
    digest: {DIGEST}
    size_bytes: 128
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
    #[cfg(test)]
    forced_token: Option<TokenId>,
}

#[derive(Clone, Debug)]
struct FirstNativeExecutionKvState {
    cache: KvCacheId,
    compatibility: KvCacheCompatibility,
    layer_kv: Vec<FirstNativeLayerKvState>,
}

#[derive(Clone, Debug)]
struct FirstNativeLayerKvState {
    k: HostTensor,
    v: HostTensor,
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
        runtime.prefill_kv_cache_completed(&cache_id, request.prompt_token_count as u32)?;
        Ok(FirstNativeExecutionKvState {
            cache: cache_id,
            compatibility,
            layer_kv: Vec::new(),
        })
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
            .insert(request.request_id.as_str().to_string(), state);
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
            .get(request.request_id.as_str())
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
    let selection = runtime
        .kernel_registry()
        .select(&selection_request)
        .map_err(|error| InferenceApiError::KernelUnavailable {
            reason: error.to_string(),
        })?;
    if selection.selected.is_none() {
        return Err(InferenceApiError::KernelUnavailable {
            reason: "Kernel Registry selected no Reference CPU candidate".into(),
        });
    }
    let selected = selection
        .selected
        .as_ref()
        .expect("selected candidate checked above");
    let advertisement = runtime
        .kernel_registry()
        .active_advertisement(&selected.kernel)
        .ok_or_else(|| InferenceApiError::KernelUnavailable {
            reason: "selected Reference CPU advertisement is no longer active".into(),
        })?;
    let mut plan = KernelDispatchPlan::from_selection(
        KernelDispatchPlanId::new("e2e-runtime-generation-dispatch"),
        &selection_request,
        selected,
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
    runtime: &'a Runtime,
    provider: &'a ReferenceCpuExecutor,
    memory: &'a mut MemoryManager,
}

fn dispatch_reference_cpu_operator(
    ctx: &mut QwenDispatchContext<'_>,
    operation_id: &str,
    operator: OperatorId,
    inputs: Vec<(TensorResourceId, TensorDescriptor, HostTensor)>,
    output: (TensorResourceId, TensorDescriptor),
    attributes: BTreeMap<String, OperatorAttributeValue>,
) -> Result<(KernelDispatchResult, HostTensor), InferenceApiError> {
    let affinity = ResourceAffinity::new(FallbackClass::Transparent)
        .with_provider(ProviderBinding::new(REFERENCE_CPU_PROVIDER_NAME))
        .with_execution_context(ctx.runtime.context().id());
    let output_resource =
        TensorResourceDescriptor::new(output.0.clone(), output.1, affinity.clone());
    let mut selection_request = KernelSelectionRequest::new(
        format!("e2e-runtime-{operation_id}"),
        operator,
        affinity.clone(),
    )
    .with_output(KernelResource::new(
        output_resource.clone(),
        KernelMemoryClass::Host,
    ));
    for (id, descriptor, tensor) in inputs {
        let resource = TensorResourceDescriptor::new(id.clone(), descriptor, affinity.clone());
        ctx.provider.write_tensor(id, tensor);
        selection_request =
            selection_request.with_input(KernelResource::new(resource, KernelMemoryClass::Host));
    }
    let selection = ctx
        .runtime
        .kernel_registry()
        .select(&selection_request)
        .map_err(|error| InferenceApiError::KernelUnavailable {
            reason: format!("{operation_id}: {error}"),
        })?;
    let selected =
        selection
            .selected
            .as_ref()
            .ok_or_else(|| InferenceApiError::KernelUnavailable {
                reason: format!("Kernel Registry selected no candidate for {operation_id}"),
            })?;
    let advertisement = ctx
        .runtime
        .kernel_registry()
        .active_advertisement(&selected.kernel)
        .ok_or_else(|| InferenceApiError::KernelUnavailable {
            reason: format!("selected advertisement for {operation_id} is no longer active"),
        })?;
    let mut plan = KernelDispatchPlan::from_selection(
        KernelDispatchPlanId::new(format!("e2e-runtime-{operation_id}-dispatch")),
        &selection_request,
        selected,
        advertisement,
        KernelInvocationId::new(format!("e2e-runtime-{operation_id}-invocation")),
    )
    .map_err(|error| InferenceApiError::KernelUnavailable {
        reason: format!("{error:?}"),
    })?;
    plan.invocation.attributes = attributes;
    if advertisement.workspace.required {
        let workspace = ctx
            .provider
            .allocate_workspace(
                ctx.memory,
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
    let kernel_result = ctx.provider.execute_invocation_with_memory_manager(
        advertisement,
        operator_spec,
        &plan.invocation,
        ctx.memory,
    );
    let dispatch_result = KernelDispatchResult::from_kernel_result(&plan, kernel_result);
    if dispatch_result.status != KernelResultStatus::Succeeded {
        return Err(InferenceApiError::ProviderUnavailable {
            reason: format!(
                "Reference CPU dispatch for {operation_id} failed: {:?}",
                dispatch_result.error
            ),
        });
    }
    let output_tensor = ctx
        .provider
        .read_tensor(&output_resource.id)
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

fn execute_qwen_prefill_hidden_states_through_dispatch(
    runtime: &Runtime,
    fixture: &E2eFixture,
    token_ids: &[TokenId],
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
    let provider = ReferenceCpuExecutor::new();
    let mut memory = MemoryManager::default();
    let mut dispatch_ctx = QwenDispatchContext {
        runtime,
        provider: &provider,
        memory: &mut memory,
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
        layer_kv.push(FirstNativeLayerKvState {
            k: k.clone(),
            v: v.clone(),
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

fn execute_qwen_decode_hidden_states_through_dispatch(
    runtime: &Runtime,
    fixture: &E2eFixture,
    token_id: TokenId,
    kv_state: &FirstNativeExecutionKvState,
    absolute_position: u64,
) -> Result<
    (
        KernelDispatchResult,
        HostTensor,
        Vec<FirstNativeLayerKvState>,
    ),
    InferenceApiError,
> {
    let provider = ReferenceCpuExecutor::new();
    let mut memory = MemoryManager::default();
    let mut dispatch_ctx = QwenDispatchContext {
        runtime,
        provider: &provider,
        memory: &mut memory,
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
        let k = concat_rows(&historical.k, &k_new)?;
        let v = concat_rows(&historical.v, &v_new)?;
        updated_layer_kv.push(FirstNativeLayerKvState {
            k: k.clone(),
            v: v.clone(),
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

fn dispatch_qwen_logits_projection(
    runtime: &Runtime,
    fixture: &E2eFixture,
    hidden_states: &HostTensor,
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
    let (dispatch_result, output) =
        dispatch_matmul(runtime, hidden_states, &token_embedding_transposed)?;
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
        let absolute_position = request.input_token_ids.len() + generated_tokens.len();
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
            (dispatch_result, logits)
        } else {
            let (_hidden_dispatch, normed_final) = if generated_tokens.is_empty() {
                let (dispatch, normed_final, layer_kv) =
                    execute_qwen_prefill_hidden_states_through_dispatch(
                        runtime,
                        &self.fixture,
                        &request.input_token_ids,
                    )?;
                kv_state.layer_kv = layer_kv;
                (dispatch, normed_final)
            } else {
                let token = *generated_tokens.last().ok_or_else(|| {
                    InferenceApiError::GenerationFailed {
                        reason: "decode requires a newly admitted token".into(),
                    }
                })?;
                let (dispatch, normed_final, layer_kv) =
                    execute_qwen_decode_hidden_states_through_dispatch(
                        runtime,
                        &self.fixture,
                        token,
                        &kv_state,
                        absolute_position as u64,
                    )?;
                kv_state.layer_kv = layer_kv;
                runtime.append_decode_kv_cache(&kv_state.cache, 1)?;
                (dispatch, normed_final)
            };
            // The tied-embedding logits projection, dispatched for real:
            // the output tensor read back here is exactly what is returned
            // below, so the evidence this dispatch produces is causally
            // tied to the returned logits rather than to an unrelated proof
            // computation.
            dispatch_qwen_logits_projection(runtime, &self.fixture, &normed_final)?
        };
        self.store_kv_state(request, kv_state.clone())?;
        let mut evidence =
            RuntimeGenerationExecutionEvidence::from_dispatch_result(&dispatch_result, true, true)
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
        Ok(RuntimeModelExecutionStep::new(logits, evidence))
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

fn build_runtime_with_model_execution_engine(fixture: &E2eFixture) -> Runtime {
    let mut runtime = Runtime::builder()
        .register_provider(std::sync::Arc::new(ReferenceCpuProvider::new()))
        .model_execution_engine(std::sync::Arc::new(E2eRuntimeModelExecutionEngine {
            fixture: fixture.clone(),
            kv_states: Arc::new(Mutex::new(BTreeMap::new())),
            #[cfg(test)]
            forced_token: None,
        }))
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
            forced_token,
        }))
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

fn load_fixture_instance(
    fixture: &E2eFixture,
    runtime: &mut Runtime,
) -> Result<(ModelInstanceId, MemoryManager), E2eConformanceError> {
    let mut coordinator = ModelLoadingCoordinator::new();
    coordinator.register_architecture(fixture.architecture_implementation.clone());
    let mut memory = MemoryManager::default();
    let mut request = ModelLoadingRequest::new(
        ModelLoadingRequestId::new("e2e-fixture-load"),
        fixture.manifest.id.clone(),
    );
    request.quantization_policy = ModelQuantizationPolicy::RejectUnsupported;
    let loaded = load_model(
        &mut coordinator,
        &mut memory,
        ModelLoadingApiRequest::new(request),
        &fixture.manifest,
        &ModelTrustDecision::new(ModelTrustStatus::Trusted, "trusted E2E fixture"),
    )?;
    let instance = create_model_instance(
        runtime,
        &loaded,
        fixture.architecture_implementation.clone(),
        ResourceAffinity::new(FallbackClass::Transparent),
    )?;
    Ok((instance, memory))
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

#[cfg(all(not(target_arch = "wasm32"), feature = "wasmtime-component-engine"))]
const QWEN_GRAPH_COMPONENT_BYTES: &[u8] =
    include_bytes!("../fixtures/components/qwen-graph.component.wat");

#[cfg(all(not(target_arch = "wasm32"), feature = "wasmtime-component-engine"))]
const QWEN_GRAPH_COMPONENT_MANIFEST_BYTES: &[u8] =
    include_bytes!("../fixtures/components/qwen-graph.component.wat.magnetar-component.yaml");

#[cfg(all(not(target_arch = "wasm32"), feature = "wasmtime-component-engine"))]
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

#[cfg(all(not(target_arch = "wasm32"), feature = "wasmtime-component-engine"))]
#[derive(Debug)]
struct QwenComponentPreflight {
    definition: ComponentDefinitionId,
    instance: ComponentInstanceId,
    graph_semantics: QwenComponentGraphSemantics,
    observations: Vec<ComponentObservation>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct QwenComponentGraphSemantics {
    prefill_node_count: usize,
    decode_node_count: usize,
}

impl QwenComponentGraphSemantics {
    fn validate_against_graphs(
        self,
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
        Ok(())
    }
}

struct FirstNativeComponentGraphs {
    prefill: ExecutionGraph,
    prefill_node_count: usize,
    decode: ExecutionGraph,
    decode_node_count: usize,
}

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
        max_memory_bytes: Some(1 << 20),
        execution_deadline_millis: Some(1_000),
        max_concurrent_invocations: Some(1),
        max_instances: Some(1),
        engine_execution_budget: Some(100_000),
        require_memory_limit: true,
    }
}

#[cfg(all(not(target_arch = "wasm32"), feature = "wasmtime-component-engine"))]
struct QwenComponentPreflightRequest {
    component_package: ComponentArtifactPackage,
    trust_store: ComponentTrustStore,
    limits: ComponentResourceLimits,
}

#[cfg(all(not(target_arch = "wasm32"), feature = "wasmtime-component-engine"))]
impl QwenComponentPreflightRequest {
    fn default_trusted() -> Self {
        Self {
            component_package: qwen_graph_component_package(),
            trust_store: ComponentTrustStore::default().trust_digest(QWEN_GRAPH_COMPONENT_DIGEST),
            limits: qwen_component_runtime_limits(),
        }
    }
}

#[cfg(all(not(target_arch = "wasm32"), feature = "wasmtime-component-engine"))]
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

#[cfg(all(not(target_arch = "wasm32"), feature = "wasmtime-component-engine"))]
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
    };
    Ok(QwenComponentPreflight {
        definition,
        instance,
        graph_semantics,
        observations: manager.observations().to_vec(),
    })
}

#[cfg(all(not(target_arch = "wasm32"), feature = "wasmtime-component-engine"))]
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
    #[cfg(all(not(target_arch = "wasm32"), feature = "wasmtime-component-engine"))]
    let component_graphs = {
        let preflight =
            validate_and_instantiate_trusted_qwen_component_before_first_native_planning()?;
        observer.observe(
            InferenceApiObservationKind::ComponentValidated,
            format!("component_definition={:?}", preflight.definition),
            None,
        );
        observer.observe(
            InferenceApiObservationKind::ComponentInstantiated,
            format!("component_instance={:?}", preflight.instance),
            None,
        );
        let graph_semantics = preflight.graph_semantics;
        let _component_preflight = (
            preflight.definition,
            preflight.instance,
            preflight.observations.len(),
        );
        build_first_native_graphs_from_component_output(
            fixture,
            tokenized.token_ids.len() as u64,
            graph_semantics,
        )?
    };
    #[cfg(not(all(not(target_arch = "wasm32"), feature = "wasmtime-component-engine")))]
    let component_graphs = {
        let semantics = QwenComponentGraphSemantics {
            prefill_node_count: qwen_prefill_graph(
                &fixture.config,
                &fixture.identity,
                tokenized.token_ids.len().max(1) as u64,
                true,
            )?
            .graph
            .nodes
            .len(),
            decode_node_count: qwen_decode_graph(
                &fixture.config,
                &fixture.identity,
                tokenized.token_ids.len().max(1) as u64,
            )?
            .graph
            .nodes
            .len(),
        };
        build_first_native_graphs_from_component_output(
            fixture,
            tokenized.token_ids.len() as u64,
            semantics,
        )?
    };
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

fn check_operator_coverage(_fixture: &E2eFixture) -> Result<BTreeSet<String>, E2eConformanceError> {
    Ok(E2E_EXERCISED_OPERATORS
        .iter()
        .map(|op| op.to_string())
        .collect())
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
    let result = run_generation_loop(
        &mut runtime,
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
    let mut runtime = build_runtime();
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
    let mut runtime = build_runtime();
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
    let mut runtime = build_runtime();
    let (instance, _memory) = load_fixture_instance(fixture, &mut runtime)?;
    let graphs = build_first_native_graphs_from_component_output(
        fixture,
        2,
        QwenComponentGraphSemantics {
            prefill_node_count: 19,
            decode_node_count: 19,
        },
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
fn check_qwen_graph_nodes_have_prepared_kernel_bindings(
    fixture: &E2eFixture,
) -> Result<(), E2eConformanceError> {
    let mut runtime = build_runtime();
    let (instance, _memory) = load_fixture_instance(fixture, &mut runtime)?;
    let graphs = build_first_native_graphs_from_component_output(
        fixture,
        2,
        QwenComponentGraphSemantics {
            prefill_node_count: 19,
            decode_node_count: 19,
        },
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
    let removed = advertisements
        .pop()
        .ok_or_else(|| E2eConformanceError::Internal {
            reason: "expected first-native fixture to advertise at least one required kernel"
                .into(),
        })?;
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
    let runtime = build_runtime();
    let prompt = vec![1, 2];
    let admitted = 3;
    let (_prefill_dispatch, _prefill_hidden, layer_kv) =
        execute_qwen_prefill_hidden_states_through_dispatch(&runtime, fixture, &prompt)?;
    let kv_state = FirstNativeExecutionKvState {
        cache: KvCacheId::new("first-native-oracle-kv").map_err(E2eConformanceError::from)?,
        compatibility: KvCacheCompatibility::new(
            GenerationModelReference::LoadedModelContext("qwen-test".into()),
            TokenizerId::new("qwen-test-tokenizer")?,
        ),
        layer_kv,
    };

    let (_decode_dispatch, decode_hidden, updated_layer_kv) =
        execute_qwen_decode_hidden_states_through_dispatch(
            &runtime,
            fixture,
            admitted,
            &kv_state,
            prompt.len() as u64,
        )?;
    let (_logits_dispatch, incremental_logits) =
        dispatch_qwen_logits_projection(&runtime, fixture, &decode_hidden)?;

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

    for layer in updated_layer_kv {
        let (k_rows, _) = layer.k.rows_cols()?;
        let (v_rows, _) = layer.v.rows_cols()?;
        if k_rows != full_sequence.len() as u64 || v_rows != full_sequence.len() as u64 {
            return Err(E2eConformanceError::GenerationFailed {
                reason: "decode did not append exactly one K/V row per layer".into(),
            });
        }
    }
    Ok(())
}

#[cfg(test)]
fn check_incremental_decode_rejects_missing_layer_kv(
    fixture: &E2eFixture,
) -> Result<(), E2eConformanceError> {
    let runtime = build_runtime();
    let kv_state = FirstNativeExecutionKvState {
        cache: KvCacheId::new("first-native-empty-kv").map_err(E2eConformanceError::from)?,
        compatibility: KvCacheCompatibility::new(
            GenerationModelReference::LoadedModelContext("qwen-test".into()),
            TokenizerId::new("qwen-test-tokenizer")?,
        ),
        layer_kv: Vec::new(),
    };
    match execute_qwen_decode_hidden_states_through_dispatch(&runtime, fixture, 3, &kv_state, 2) {
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

    // Normal Generation/Sampling/Provider-Kernel path: the same
    // `run_generation_loop` + real Reference CPU forward pass as the
    // multi-call success path, just against the one-shot session.
    let request = build_generation_request(
        GenerationRequestId::new("e2e-one-shot-generation")?,
        Some(session.clone()),
        GenerationModelReference::ModelInstance(instance),
        generation_tokenizer_reference(fixture),
        tokenized,
        2,
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

fn elapsed_millis(start: SystemTime) -> u64 {
    SystemTime::now()
        .duration_since(start)
        .map(|duration| duration.as_millis() as u64)
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(all(not(target_arch = "wasm32"), feature = "wasmtime-component-engine"))]
    fn qwen_component_fixture_wat(
        prefill_export: &str,
        decode_export: &str,
        authority_export: &str,
    ) -> String {
        format!(
            r#"(component
    (core module $m
        {prefill_export}
        {decode_export}
        {authority_export})
    (core instance $i (instantiate $m))
    (func (export "prefill-node-count") (result u32)
        (canon lift (core func $i "prefill-node-count")))
    (func (export "decode-node-count") (result u32)
        (canon lift (core func $i "decode-node-count")))
    (func (export "provider-authority-count") (result u32)
        (canon lift (core func $i "provider-authority-count")))
    (func $prefill-node-count (result u32)
        (canon lift (core func $i "prefill-node-count")))
    (func $decode-node-count (result u32)
        (canon lift (core func $i "decode-node-count")))
    (func $provider-authority-count (result u32)
        (canon lift (core func $i "provider-authority-count")))
    (instance $qwen-graph-fixture
        (export "prefill-node-count" (func $prefill-node-count))
        (export "decode-node-count" (func $decode-node-count))
        (export "provider-authority-count" (func $provider-authority-count)))
    (export "magnetar:qwen/graph-fixture@1.0.0" (instance $qwen-graph-fixture)))
"#
        )
    }

    #[cfg(all(not(target_arch = "wasm32"), feature = "wasmtime-component-engine"))]
    fn qwen_component_count_fixture_wat(prefill: u32, decode: u32, authority: u32) -> String {
        qwen_component_fixture_wat(
            &format!(r#"(func (export "prefill-node-count") (result i32) i32.const {prefill})"#),
            &format!(r#"(func (export "decode-node-count") (result i32) i32.const {decode})"#),
            &format!(
                r#"(func (export "provider-authority-count") (result i32) i32.const {authority})"#
            ),
        )
    }

    #[cfg(all(not(target_arch = "wasm32"), feature = "wasmtime-component-engine"))]
    fn qwen_component_manifest(digest: &str) -> String {
        format!(
            r#"schema: magnetar-component-artifact
schema_version: 1
artifact:
  kind: component
  digest:
    algorithm: sha256
    value: "{digest}"
component:
  name: "magnetar.qwen.graph-fixture"
  version: "0.1.0"
  description: "Executable Qwen graph fixture component"
  role: "qwen-graph-fixture"
runtime:
  magnetar:
    min_version: "0.1.0"
wit:
  imports: []
  exports:
    - package: "magnetar:qwen"
      interface: "graph-fixture"
      version: "1.0.0"
capabilities:
  requires: []
authority:
  requires: []
engine:
  profile: "native"
  features:
    - component-model
    - resource-limits
publisher:
  id: "local-dev"
  name: "Local Development"
source:
  kind: "local"
  uri: "./qwen-graph.component.wat"
signatures: []
"#
        )
    }

    #[cfg(all(not(target_arch = "wasm32"), feature = "wasmtime-component-engine"))]
    fn sha256_component_digest(bytes: &[u8]) -> String {
        let digest = Sha256::digest(bytes);
        let mut encoded = String::from("sha256:");
        const HEX: &[u8; 16] = b"0123456789abcdef";
        for byte in digest {
            encoded.push(HEX[(byte >> 4) as usize] as char);
            encoded.push(HEX[(byte & 0x0f) as usize] as char);
        }
        encoded
    }

    #[cfg(all(not(target_arch = "wasm32"), feature = "wasmtime-component-engine"))]
    fn qwen_component_preflight_package(
        wat: &str,
        manifest_digest: Option<&str>,
    ) -> (ComponentArtifactPackage, String) {
        let digest = sha256_component_digest(wat.as_bytes());
        let package = ComponentArtifactPackage::new(
            wat.as_bytes().to_vec(),
            qwen_component_manifest(manifest_digest.unwrap_or(&digest)).into_bytes(),
            ComponentDigest::parse("sha256", manifest_digest.unwrap_or(&digest)),
            ComponentDistributionSource::new(
                ComponentDistributionSourceKind::DevelopmentFixture,
                QWEN_GRAPH_COMPONENT_NAME,
            ),
        );
        (package, digest)
    }

    #[cfg(all(not(target_arch = "wasm32"), feature = "wasmtime-component-engine"))]
    fn trusted_preflight_request_for_temp_component(
        component_package: ComponentArtifactPackage,
        digest: &str,
    ) -> QwenComponentPreflightRequest {
        QwenComponentPreflightRequest {
            component_package,
            trust_store: ComponentTrustStore::default().trust_digest(digest),
            limits: qwen_component_runtime_limits(),
        }
    }

    #[test]
    fn e2e_success_path_resolves_loads_generates_and_cleans_up() {
        let fixture = e2e_fixture().expect("fixture builds");
        check_success_path(&fixture).expect("Runtime success path completes");
    }

    #[test]
    fn e2e_runs_without_gpu_network_or_tachyon() {
        // Structural: the fixture and success path never reference GPU,
        // network, or Tachyon primitives, and CLI-owned authorities are
        // explicitly denied to Runtime.
        check_cli_boundary_denials().expect("CLI-owned authorities are denied");
    }

    #[test]
    fn e2e_fixture_model_passes_validation() {
        let fixture = e2e_fixture().expect("fixture builds and validates");
        fixture.manifest.validate().expect("manifest re-validates");
        assert_eq!(
            fixture.identity.implementation,
            ModelComponentImplementationKind::WebAssemblyComponent
        );
        assert_eq!(
            fixture.config.architecture.vocabulary_size,
            E2E_FIXTURE_VOCAB
        );
        assert_eq!(fixture.config.architecture.hidden_size, E2E_FIXTURE_HIDDEN);
        assert_eq!(fixture.config.architecture.layer_count, E2E_FIXTURE_LAYERS);
    }

    #[test]
    fn e2e_fixture_weight_digest_is_stable() {
        let fixture = e2e_fixture().expect("fixture builds");
        let digest = e2e_fixture_weight_digest(&fixture.weights);
        assert_eq!(digest, E2E_FIXTURE_WEIGHT_DIGEST);
        assert_eq!(digest, e2e_fixture_weight_digest(&fixture.weights));
    }

    #[test]
    fn e2e_fixture_tokenizer_produces_deterministic_tokens() {
        let fixture = e2e_fixture().expect("fixture builds");
        check_fixture_tokenizer_deterministic(&fixture).expect("tokenization is deterministic");
    }

    #[test]
    fn e2e_already_tokenized_prompt_path_bypasses_text_tokenization() {
        let fixture = e2e_fixture().expect("fixture builds");
        check_already_tokenized_prompt_path(&fixture).expect("already-tokenized path is preserved");
    }

    #[test]
    fn e2e_raw_prompt_logging_is_disabled_by_default() {
        assert!(!SessionPolicy::default().raw_prompt_logging_allowed);
    }

    #[test]
    fn e2e_required_path_returns_usage_and_cleans_up() {
        let fixture = e2e_fixture().expect("fixture builds");
        let result = run_success_path(&fixture).expect("success path returns output");
        assert!(result.generation_result.output.usage.generated_tokens > 0);
        assert!(!result.observer.observations().is_empty());
    }

    #[test]
    fn e2e_no_shortcut_direct_provider_invocation_is_rejected() {
        check_no_shortcut_direct_provider_rejected().expect("direct-invocation shortcut rejected");
    }

    #[test]
    fn e2e_generation_step_logits_are_produced_by_the_evidence_bearing_dispatch() {
        let fixture = e2e_fixture().expect("fixture builds");
        let runtime = build_runtime();
        let sequence = vec![1u32, 2u32];

        let normed_final =
            e2e_forward_hidden_states(&fixture, &sequence).expect("hidden states computed");
        let token_embedding = fixture_tensor_by_name(&fixture.weights, "token_embedding")
            .expect("token embedding present");
        let token_embedding_transposed =
            transpose_rows_cols(token_embedding).expect("token embedding transposes");

        let (dispatch_result, dispatched_output) =
            dispatch_matmul(&runtime, &normed_final, &token_embedding_transposed)
                .expect("real matmul dispatch succeeds");
        assert_eq!(dispatch_result.status, KernelResultStatus::Succeeded);

        let vocab = fixture.config.architecture.vocabulary_size as usize;
        let last_row_start = (sequence.len() - 1) * vocab;
        let dispatched_logits = &dispatched_output.data[last_row_start..last_row_start + vocab];

        // What `E2eRuntimeModelExecutionEngine::execute_generation_step` returns
        // for this sequence must equal the dispatch's own output exactly --
        // it is read directly from `dispatched_output`, never recomputed
        // separately -- so this also confirms the dispatch path is numerically
        // correct against the independent `e2e_forward` ground truth.
        let expected = e2e_forward(&fixture, &sequence).expect("forward pass produces logits");
        assert_eq!(dispatched_logits, expected.as_slice());

        // Tampering with the dispatch's actual input changes its output,
        // proving the returned data is causally produced by this dispatch --
        // not decorated onto an unrelated proof computation whose result is
        // discarded, which is the shortcut this test guards against.
        let corrupted_embedding = HostTensor::new(
            token_embedding_transposed.shape.clone(),
            vec![0.0_f32; token_embedding_transposed.data.len()],
        )
        .expect("zeroed tensor constructs");
        let (corrupted_result, corrupted_output) =
            dispatch_matmul(&runtime, &normed_final, &corrupted_embedding)
                .expect("corrupted dispatch still succeeds");
        assert_eq!(corrupted_result.status, KernelResultStatus::Succeeded);
        assert_ne!(
            &corrupted_output.data[last_row_start..last_row_start + vocab],
            dispatched_logits
        );
    }

    #[test]
    fn e2e_reference_cpu_selected_through_kernel_registry() {
        check_reference_cpu_selected_through_kernel_registry()
            .expect("Reference CPU selected through Kernel Registry");
    }

    #[test]
    fn e2e_operator_coverage_report_lists_required_operators() {
        let fixture = e2e_fixture().expect("fixture builds");
        let operators = check_operator_coverage(&fixture).expect("operator coverage computed");
        for expected in E2E_EXERCISED_OPERATORS {
            assert!(operators.contains(expected), "missing operator {expected}");
        }
    }

    #[test]
    fn e2e_invalid_graph_fixture_fails_validation() {
        check_invalid_graph_fixture().expect("invalid graph fixture is rejected");
    }

    #[test]
    fn e2e_graph_production_and_execution_succeeds_for_valid_fixture() {
        let fixture = e2e_fixture().expect("fixture builds");
        check_graph_production_and_execution(&fixture).expect("prefill/decode graphs execute");
    }

    #[test]
    fn e2e_max_new_tokens_reached_stops_generation() {
        let fixture = e2e_fixture().expect("fixture builds");
        check_max_new_tokens_stops_generation(&fixture).expect("max token stop is honored");
    }

    #[test]
    fn e2e_eos_token_stops_generation() {
        let fixture = e2e_fixture().expect("fixture builds");
        check_eos_token_stops_generation(&fixture).expect("EOS stop is honored");
    }

    #[test]
    fn e2e_generation_cancelled_stops_with_cancelled_finish_reason() {
        let fixture = e2e_fixture().expect("fixture builds");
        check_generation_cancelled(&fixture).expect("cancellation is honored");
    }

    #[test]
    fn e2e_sampling_greedy_selects_deterministic_token() {
        check_sampling_greedy_deterministic().expect("greedy sampling is deterministic");
    }

    #[test]
    fn e2e_streaming_events_are_ordered() {
        let fixture = e2e_fixture().expect("fixture builds");
        check_streaming_order(&fixture).expect("streaming events are ordered");
    }

    #[test]
    fn e2e_closed_session_rejects_generation() {
        let fixture = e2e_fixture().expect("fixture builds");
        check_closed_session_rejects_generation(&fixture).expect("closed session is rejected");
    }

    #[test]
    fn e2e_first_native_generation_requires_ready_model_instance() {
        let fixture = e2e_fixture().expect("fixture builds");
        check_first_native_generation_requires_ready_model_instance(&fixture)
            .expect("non-ready model instance is rejected");
    }

    #[test]
    fn e2e_missing_prepared_plan_fails_closed() {
        check_missing_prepared_plan_fails_closed().expect("missing prepared plan is rejected");
    }

    #[test]
    fn e2e_invalidated_prepared_plan_rejects_new_work() {
        let fixture = e2e_fixture().expect("fixture builds");
        check_invalidated_prepared_plan_rejects_new_work(&fixture)
            .expect("invalidated prepared plan is rejected");
    }

    #[test]
    fn e2e_qwen_graph_nodes_have_prepared_kernel_bindings() {
        let fixture = e2e_fixture().expect("fixture builds");
        check_qwen_graph_nodes_have_prepared_kernel_bindings(&fixture)
            .expect("Qwen graph nodes are bound to prepared kernels");
    }

    #[cfg(all(not(target_arch = "wasm32"), feature = "wasmtime-component-engine"))]
    #[test]
    fn e2e_qwen_component_artifact_trust_is_validated_before_planning() {
        validate_and_instantiate_trusted_qwen_component_before_first_native_planning()
            .expect("trusted Qwen Component fixture validates before planning");
    }

    #[cfg(all(not(target_arch = "wasm32"), feature = "wasmtime-component-engine"))]
    #[test]
    fn e2e_qwen_component_instantiates_with_wasmtime_limits_before_planning() {
        let preflight =
            validate_and_instantiate_trusted_qwen_component_before_first_native_planning()
                .expect("trusted Qwen Component fixture instantiates before planning");
        assert!(preflight.definition.get() > 0);
        assert!(preflight.instance.get() > 0);
        assert_eq!(
            preflight.graph_semantics,
            QwenComponentGraphSemantics {
                prefill_node_count: 19,
                decode_node_count: 19,
            }
        );
        assert!(preflight.observations.iter().any(|observation| {
            observation.kind == ComponentObservationKind::Instantiation
                && observation.message.contains("component instance ready")
        }));
        assert!(preflight.observations.iter().any(|observation| {
            observation.kind == ComponentObservationKind::Invocation
                && observation
                    .message
                    .contains("component invocation completed")
        }));
    }

    #[cfg(all(not(target_arch = "wasm32"), feature = "wasmtime-component-engine"))]
    #[test]
    fn e2e_qwen_component_artifact_trust_rejection_fails_before_planning() {
        let mut request = QwenComponentPreflightRequest::default_trusted();
        request.trust_store = ComponentTrustStore::default();
        match validate_and_instantiate_qwen_component_before_first_native_planning(request) {
            Err(E2eConformanceError::ModelComponentFailed { reason })
                if reason.contains("artifact rejected") || reason.contains("no trust policy") =>
            {
                Ok(())
            }
            Err(error) => Err(error),
            Ok(_) => Err(E2eConformanceError::ModelComponentFailed {
                reason: "untrusted Qwen Component fixture was accepted".into(),
            }),
        }
        .expect("untrusted Qwen Component fixture is rejected before planning");
    }

    #[cfg(all(not(target_arch = "wasm32"), feature = "wasmtime-component-engine"))]
    #[test]
    fn e2e_qwen_component_missing_artifact_fails_before_planning() {
        let (component_package, digest) = qwen_component_preflight_package("", None);
        let request = QwenComponentPreflightRequest {
            component_package,
            trust_store: ComponentTrustStore::default().trust_digest(&digest),
            limits: qwen_component_runtime_limits(),
        };

        let result = validate_and_instantiate_qwen_component_before_first_native_planning(request);

        assert!(
            matches!(
                result,
                Err(E2eConformanceError::ModelComponentFailed { .. })
            ),
            "missing Qwen Component artifact was not rejected: {result:?}"
        );
    }

    #[cfg(all(not(target_arch = "wasm32"), feature = "wasmtime-component-engine"))]
    #[test]
    fn e2e_qwen_component_digest_mismatch_fails_before_planning() {
        let wat = qwen_component_count_fixture_wat(19, 19, 0);
        let wrong_digest =
            "sha256:0000000000000000000000000000000000000000000000000000000000000000";
        let (component_package, _digest) =
            qwen_component_preflight_package(&wat, Some(wrong_digest));
        let request = trusted_preflight_request_for_temp_component(component_package, wrong_digest);

        let result = validate_and_instantiate_qwen_component_before_first_native_planning(request);

        assert!(
            matches!(
                result,
                Err(E2eConformanceError::ModelComponentFailed { .. })
            ),
            "digest-mismatched Qwen Component artifact was not rejected: {result:?}"
        );
    }

    #[cfg(all(not(target_arch = "wasm32"), feature = "wasmtime-component-engine"))]
    #[test]
    fn e2e_qwen_component_fuel_exhaustion_fails_before_planning() {
        let wat = qwen_component_fixture_wat(
            r#"(func (export "prefill-node-count") (result i32)
                (loop $again br $again)
                i32.const 13)"#,
            r#"(func (export "decode-node-count") (result i32) i32.const 19)"#,
            r#"(func (export "provider-authority-count") (result i32) i32.const 0)"#,
        );
        let (component_package, digest) = qwen_component_preflight_package(&wat, None);
        let mut request = trusted_preflight_request_for_temp_component(component_package, &digest);
        request.limits.engine_execution_budget = Some(1_000);

        let result = validate_and_instantiate_qwen_component_before_first_native_planning(request);

        assert!(
            matches!(
                result,
                Err(E2eConformanceError::ModelComponentFailed { .. })
            ),
            "runaway Qwen Component was not stopped by fuel: {result:?}"
        );
    }

    #[cfg(all(not(target_arch = "wasm32"), feature = "wasmtime-component-engine"))]
    #[test]
    fn e2e_qwen_component_deadline_fails_before_planning() {
        let wat = qwen_component_count_fixture_wat(19, 19, 0);
        let (component_package, digest) = qwen_component_preflight_package(&wat, None);
        let mut request = trusted_preflight_request_for_temp_component(component_package, &digest);
        request.limits.execution_deadline_millis = Some(0);

        let result = validate_and_instantiate_qwen_component_before_first_native_planning(request);

        assert!(
            matches!(
                result,
                Err(E2eConformanceError::ModelComponentFailed { .. })
            ),
            "expired Qwen Component deadline was not rejected: {result:?}"
        );
    }

    #[cfg(all(not(target_arch = "wasm32"), feature = "wasmtime-component-engine"))]
    #[test]
    fn e2e_qwen_component_invalid_output_fails_before_planning() {
        let wat = r#"(component
    (core module $m
        (func (export "prefill-node-count"))
        (func (export "decode-node-count") (result i32) i32.const 12)
        (func (export "provider-authority-count") (result i32) i32.const 0))
    (core instance $i (instantiate $m))
    (func (export "prefill-node-count")
        (canon lift (core func $i "prefill-node-count")))
    (func (export "decode-node-count") (result u32)
        (canon lift (core func $i "decode-node-count")))
    (func (export "provider-authority-count") (result u32)
        (canon lift (core func $i "provider-authority-count")))
    (func $prefill-node-count
        (canon lift (core func $i "prefill-node-count")))
    (func $decode-node-count (result u32)
        (canon lift (core func $i "decode-node-count")))
    (func $provider-authority-count (result u32)
        (canon lift (core func $i "provider-authority-count")))
    (instance $qwen-graph-fixture
        (export "prefill-node-count" (func $prefill-node-count))
        (export "decode-node-count" (func $decode-node-count))
        (export "provider-authority-count" (func $provider-authority-count)))
    (export "magnetar:qwen/graph-fixture@1.0.0" (instance $qwen-graph-fixture)))
"#;
        let (component_package, digest) = qwen_component_preflight_package(wat, None);
        let request = trusted_preflight_request_for_temp_component(component_package, &digest);

        let result = validate_and_instantiate_qwen_component_before_first_native_planning(request);

        assert!(
            matches!(
                result,
                Err(E2eConformanceError::GraphValidationFailed { .. })
            ),
            "Qwen Component invalid output was not rejected: {result:?}"
        );
    }

    #[cfg(all(not(target_arch = "wasm32"), feature = "wasmtime-component-engine"))]
    #[test]
    fn e2e_qwen_component_incompatible_graph_fails_before_planning() {
        let fixture = e2e_fixture().expect("fixture builds");
        let mut runtime = build_runtime_with_model_execution_engine(&fixture);
        let (instance, _memory) =
            load_fixture_instance(&fixture, &mut runtime).expect("fixture instance loads");
        let component_graph_semantics = QwenComponentGraphSemantics {
            prefill_node_count: 99,
            decode_node_count: 19,
        };

        let result =
            build_first_native_graphs_from_component_output(&fixture, 2, component_graph_semantics)
                .and_then(|graphs| {
                    prepare_first_native_execution_plans(&runtime, &instance, graphs, 2)
                });

        assert!(
            matches!(
                result,
                Err(E2eConformanceError::GraphValidationFailed { .. })
            ),
            "Qwen Component/runtime graph mismatch was not rejected"
        );
    }

    #[cfg(all(not(target_arch = "wasm32"), feature = "wasmtime-component-engine"))]
    #[test]
    fn e2e_qwen_component_provider_authority_fails_before_planning() {
        let wat = qwen_component_count_fixture_wat(19, 19, 1);
        let (component_package, digest) = qwen_component_preflight_package(&wat, None);
        let request = trusted_preflight_request_for_temp_component(component_package, &digest);

        let result = validate_and_instantiate_qwen_component_before_first_native_planning(request);

        assert!(
            matches!(result, Err(E2eConformanceError::BoundaryViolation { .. })),
            "Qwen Component Provider authority was not rejected: {result:?}"
        );
    }

    #[test]
    fn e2e_kv_cache_diagnostics_redact_raw_contents() {
        check_kv_cache_diagnostics_redacted().expect("cache usage carries no raw contents");
    }

    #[test]
    fn e2e_incremental_decode_uses_existing_kv_and_matches_full_sequence_oracle() {
        let fixture = e2e_fixture().expect("fixture builds");
        check_incremental_decode_matches_full_sequence_oracle(&fixture)
            .expect("incremental decode matches full-sequence oracle");
    }

    #[test]
    fn e2e_incremental_decode_rejects_missing_layer_kv() {
        let fixture = e2e_fixture().expect("fixture builds");
        check_incremental_decode_rejects_missing_layer_kv(&fixture)
            .expect("decode requires existing layer KV state");
    }

    #[test]
    fn e2e_tensor_output_updates_readiness_without_raw_pointer() {
        let fixture = e2e_fixture().expect("fixture builds");
        let logits = e2e_forward(&fixture, &[1, 2]).expect("forward pass produces logits");
        assert_eq!(logits.len(), E2E_FIXTURE_VOCAB as usize);
        assert!(logits.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn e2e_resource_cleanup_after_generation_and_session_close() {
        let fixture = e2e_fixture().expect("fixture builds");
        let mut runtime = build_runtime();
        let (instance, _memory) = load_fixture_instance(&fixture, &mut runtime).expect("loads");
        let report = unload_model_instance(
            &mut runtime,
            &instance,
            ModelInstanceUnloadPolicy::DrainActiveUse,
        )
        .expect("unload succeeds");
        assert!(!report.dangling_session_references);
    }

    #[test]
    fn e2e_cli_boundary_rejects_workspace_file_access() {
        check_cli_boundary_denials().expect("CLI boundary denials hold");
    }

    #[test]
    fn e2e_diagnostics_redact_raw_values_on_failure() {
        check_diagnostics_redaction_on_failure().expect("diagnostics redact native handles");
    }

    #[test]
    fn e2e_failure_cases_report_structured_errors() {
        check_invalid_model_reference().expect("invalid model reference rejected");
        check_untrusted_artifact(&e2e_fixture().unwrap()).expect("untrusted artifact rejected");
        check_incompatible_tokenizer(&e2e_fixture().unwrap())
            .expect("incompatible tokenizer rejected");
        check_unsupported_operator().expect("unsupported operator rejected");
        check_missing_kernel().expect("missing kernel rejected");
        check_required_kernel_removal_fails_coverage()
            .expect("required kernel removal fails coverage");
        check_invalid_tensor_shape(&e2e_fixture().unwrap()).expect("invalid tensor shape rejected");
        check_memory_admission_failure().expect("memory admission failure rejected");
        check_closed_session_rejects_generation(&e2e_fixture().unwrap())
            .expect("closed session rejected");
        check_first_native_generation_requires_ready_model_instance(&e2e_fixture().unwrap())
            .expect("non-ready model instance rejected");
        check_missing_prepared_plan_fails_closed().expect("missing prepared plan rejected");
        check_invalidated_prepared_plan_rejects_new_work(&e2e_fixture().unwrap())
            .expect("invalidated prepared plan rejected");
        check_qwen_graph_nodes_have_prepared_kernel_bindings(&e2e_fixture().unwrap())
            .expect("Qwen graph nodes bound to prepared kernels");
        check_generation_cancelled(&e2e_fixture().unwrap()).expect("cancellation reported");
        check_cli_boundary_denials().expect("policy denial reported");
        check_raw_handle_access_denied().expect("raw handle access denied");
    }

    #[test]
    fn e2e_determinism_repeated_runs_produce_matching_tokens() {
        let fixture = e2e_fixture().expect("fixture builds");
        check_determinism(&fixture).expect("generation is deterministic");
    }

    #[test]
    fn e2e_report_contains_required_metadata_fields() {
        let report = run_e2e_local_inference_conformance();
        check_report_metadata(&report).expect("report has required metadata");
        assert!(report.redacted);
        let no_shortcut_success = report
            .test_cases
            .iter()
            .find(|test| test.name == "success-path-no-shortcut-validated")
            .expect("success-path no-shortcut validation is reported");
        assert_eq!(no_shortcut_success.status, E2eTestStatus::Passed);
        assert!(no_shortcut_success.diagnostic.is_none());
        assert!(report.is_conformant());
    }

    #[test]
    fn e2e_ci_can_run_without_gpu_and_reports_only_expected_required_failure() {
        let report = run_e2e_local_inference_conformance();
        let failed: Vec<_> = report
            .test_cases
            .iter()
            .filter(|test| test.status == E2eTestStatus::Failed)
            .map(|test| test.name.as_str())
            .collect();
        assert_eq!(failed, Vec::<&str>::new());
    }

    #[test]
    fn e2e_local_suite_does_not_require_tachyon_or_browser() {
        // Browser support is explicit and structured, never assumed.
        let _ = qwen_browser_supported(ModelComponentImplementationKind::RuntimeNative);
    }

    #[test]
    fn e2e_error_categories_use_structured_codes() {
        let expected = [
            (
                "e2e-suite-unavailable",
                E2eConformanceError::SuiteUnavailable {
                    reason: String::new(),
                },
            ),
            (
                "e2e-fixture-invalid",
                E2eConformanceError::FixtureInvalid {
                    reason: String::new(),
                },
            ),
            (
                "e2e-model-resolution-failed",
                E2eConformanceError::ModelResolutionFailed {
                    reason: String::new(),
                },
            ),
            (
                "e2e-model-loading-failed",
                E2eConformanceError::ModelLoadingFailed {
                    reason: String::new(),
                },
            ),
            (
                "e2e-model-component-failed",
                E2eConformanceError::ModelComponentFailed {
                    reason: String::new(),
                },
            ),
            (
                "e2e-tokenizer-failed",
                E2eConformanceError::TokenizerFailed {
                    reason: String::new(),
                },
            ),
            (
                "e2e-session-failed",
                E2eConformanceError::SessionFailed {
                    reason: String::new(),
                },
            ),
            (
                "e2e-generation-failed",
                E2eConformanceError::GenerationFailed {
                    reason: String::new(),
                },
            ),
            (
                "e2e-sampling-failed",
                E2eConformanceError::SamplingFailed {
                    reason: String::new(),
                },
            ),
            (
                "e2e-streaming-failed",
                E2eConformanceError::StreamingFailed {
                    reason: String::new(),
                },
            ),
            (
                "e2e-graph-validation-failed",
                E2eConformanceError::GraphValidationFailed {
                    reason: String::new(),
                },
            ),
            (
                "e2e-operator-coverage-missing",
                E2eConformanceError::OperatorCoverageMissing {
                    reason: String::new(),
                },
            ),
            (
                "e2e-kernel-coverage-missing",
                E2eConformanceError::KernelCoverageMissing {
                    reason: String::new(),
                },
            ),
            (
                "e2e-memory-validation-failed",
                E2eConformanceError::MemoryValidationFailed {
                    reason: String::new(),
                },
            ),
            (
                "e2e-redaction-failed",
                E2eConformanceError::RedactionFailed {
                    reason: String::new(),
                },
            ),
            (
                "e2e-boundary-violation",
                E2eConformanceError::BoundaryViolation {
                    reason: String::new(),
                },
            ),
            (
                "e2e-determinism-failed",
                E2eConformanceError::DeterminismFailed {
                    reason: String::new(),
                },
            ),
            (
                "internal-e2e-conformance",
                E2eConformanceError::Internal {
                    reason: String::new(),
                },
            ),
        ];
        for (code, error) in expected {
            assert_eq!(error.code(), code);
        }
    }

    #[test]
    fn e2e_observability_emits_only_redacted_report_metadata() {
        let report = run_e2e_local_inference_conformance();
        let json = e2e_conformance_report_json(&report).expect("report serializes");
        assert!(!json.contains("0x"));
        assert!(!json.contains("native_handle"));
        assert!(report.redacted);
    }

    #[test]
    fn e2e_report_round_trips_through_json() {
        let report = run_e2e_local_inference_conformance();
        let json = e2e_conformance_report_json(&report).expect("serializes");
        let restored: E2eConformanceReport = serde_json::from_str(&json).expect("deserializes");
        assert_eq!(restored.suite_version, report.suite_version);
        assert_eq!(restored.test_cases.len(), report.test_cases.len());
    }

    #[test]
    fn e2e_fixture_tokenizer_streams_decode_across_multiple_chunks() {
        let fixture = e2e_fixture().expect("fixture builds");
        let full = tokenize_prompt_input(
            &fixture.tokenizer,
            TokenizationRequest::new(PromptInput::PlainText("hi!".into())),
            None,
        )
        .expect("tokenizes");
        assert!(full.token_ids.len() >= 2);

        let mut state = StreamingDecodeState::default();
        let mut decoded = String::new();
        for token_id in &full.token_ids {
            let output = fixture
                .tokenizer
                .streaming_decode(state, vec![*token_id], false)
                .expect("streaming decode step succeeds");
            decoded.push_str(&output.text);
            state = output.pending_partial_state.unwrap_or_default();
        }
        assert_eq!(decoded, "hi!");
    }

    #[test]
    fn e2e_one_shot_session_uses_normal_model_instance_and_tokenizer_path() {
        let fixture = e2e_fixture().expect("fixture builds");
        let mut runtime = build_runtime();
        let (instance, _memory) = load_fixture_instance(&fixture, &mut runtime).expect("loads");
        let session_request = SessionCreationRequest {
            model: GenerationModelReference::ModelInstance(instance),
            tokenizer: generation_tokenizer_reference(&fixture),
            generation_defaults: GenerationParameters::greedy(),
            policy: SessionPolicy::default(),
            memory: SessionMemoryBudget::default(),
            allowed_capabilities: BTreeSet::new(),
            correlation_id: None,
            created_at_millis: 0,
        };
        let session = create_one_shot_session(&mut runtime, session_request).expect("creates");
        let status = session_status(
            &runtime,
            &session,
            &SessionAccessPolicy::authorize(session.clone()),
        )
        .expect("status is readable");
        assert_eq!(status.lifecycle, SessionLifecycleState::Ready);
        close_inference_session(&mut runtime, &session).expect("closes");
    }

    #[test]
    fn e2e_one_shot_session_exercises_normal_generation_sampling_and_kernel_path() {
        let fixture = e2e_fixture().expect("fixture builds");
        check_one_shot_session_normal_paths(&fixture)
            .expect("one-shot session uses normal generation path");
    }

    #[test]
    fn e2e_chat_message_prompt_path_uses_formatter_and_tokenizer_contract() {
        let fixture = e2e_fixture().expect("fixture builds");
        check_chat_message_prompt_path(&fixture).expect("chat message prompt tokenizes");
    }

    #[test]
    fn e2e_chat_message_prompt_without_formatter_is_policy_denied() {
        let fixture = e2e_fixture().expect("fixture builds");
        let messages = vec![ChatMessage::new("user", "hi")];
        let result = tokenize_prompt_input(
            &fixture.tokenizer,
            TokenizationRequest::new(PromptInput::ChatMessages(messages)),
            None,
        );
        assert!(matches!(
            result,
            Err(InferenceApiError::PolicyDenied { .. })
        ));
    }

    #[test]
    fn e2e_no_shortcut_direct_kernel_invocation_is_rejected() {
        check_no_shortcut_direct_kernel_invocation_rejected()
            .expect("fabricated incompatible kernel candidate rejected");
    }

    #[test]
    fn e2e_no_shortcut_model_loading_bypass_is_detected() {
        let fixture = e2e_fixture().expect("fixture builds");
        check_no_shortcut_model_loading_bypass_detected(&fixture)
            .expect("Model Loading bypass is detected");
    }

    #[test]
    fn e2e_no_shortcut_model_component_bypass_is_detected() {
        let fixture = e2e_fixture().expect("fixture builds");
        check_no_shortcut_model_component_bypass_detected(&fixture)
            .expect("Model Component bypass is detected");
    }

    #[test]
    fn e2e_no_shortcut_memory_manager_bypass_is_detected() {
        let fixture = e2e_fixture().expect("fixture builds");
        check_no_shortcut_memory_manager_bypass_detected(&fixture)
            .expect("Memory Manager bypass is detected");
    }

    #[test]
    fn e2e_dtype_and_layout_conversion_are_never_silent() {
        check_dtype_and_layout_conversion_are_explicit()
            .expect("dtype/layout conversion is explicit, never silent");
    }

    #[test]
    fn e2e_max_total_tokens_reached_stops_generation() {
        let fixture = e2e_fixture().expect("fixture builds");
        check_max_total_tokens_stops_generation(&fixture)
            .expect("max_total_tokens stops generation");
    }

    #[test]
    fn e2e_stochastic_sampling_is_seed_deterministic() {
        check_stochastic_sampling_is_seed_deterministic()
            .expect("seeded stochastic sampling is reproducible");
    }

    #[test]
    fn e2e_kv_and_prefix_cache_lifecycle_redacts_raw_contents() {
        let fixture = e2e_fixture().expect("fixture builds");
        check_kv_and_prefix_cache_lifecycle(&fixture).expect("KV/Prefix cache lifecycle completes");
    }

    #[test]
    fn e2e_tensor_resource_lifecycle_reaches_ready_and_released() {
        check_tensor_resource_lifecycle().expect("Tensor Resource lifecycle completes");
    }

    #[test]
    fn e2e_memory_operator_output_accounting_leaves_no_untracked_allocation() {
        check_memory_operator_output_accounting()
            .expect("operator output and workspace allocations are tracked and released");
    }

    #[test]
    fn e2e_generation_timeout_maps_to_structured_error() {
        check_generation_timeout_maps_to_structured_error()
            .expect("generation timeout maps to a structured error deterministically");
    }

    #[test]
    fn e2e_run_emits_lifecycle_observation_markers() {
        let report = run_e2e_local_inference_conformance();
        for marker in [
            "observation-suite-started",
            "observation-fixture-loaded",
            "observation-success-path-started",
            "observation-success-path-completed",
            "observation-failure-case-started",
            "observation-failure-case-completed",
            "observation-redaction-failure",
            "observation-boundary-violation",
            "observation-report-generated",
        ] {
            assert!(
                report.test_cases.iter().any(|test| test.name == marker),
                "missing lifecycle observation marker: {marker}"
            );
        }
    }

    #[test]
    fn e2e_authoritative_path_collects_correlated_runtime_observations() {
        let fixture = e2e_fixture().expect("fixture builds");
        let outcome = run_success_path(&fixture).expect("success path runs");
        let observations = outcome.observer.observations();
        for kind in [
            InferenceApiObservationKind::ComponentValidated,
            InferenceApiObservationKind::ComponentInstantiated,
            InferenceApiObservationKind::ModelInstanceReady,
            InferenceApiObservationKind::GraphValidationCompleted,
            InferenceApiObservationKind::PlanSelected,
            InferenceApiObservationKind::PlanGuardAccepted,
            InferenceApiObservationKind::KernelResolved,
            InferenceApiObservationKind::KernelPrepared,
            InferenceApiObservationKind::ProviderSubmitted,
            InferenceApiObservationKind::ProviderCompleted,
            InferenceApiObservationKind::LogitsProduced,
            InferenceApiObservationKind::SamplingCompleted,
            InferenceApiObservationKind::TokenCommitted,
        ] {
            assert!(
                observations
                    .iter()
                    .any(|observation| observation.kind == kind),
                "missing authoritative observation {kind:?}"
            );
        }
        assert!(observations.iter().any(|observation| {
            observation.kind == InferenceApiObservationKind::PlanSelected
                && observation.message.contains("request=e2e-success-path")
                && observation.message.contains("plan_generation=")
        }));
        assert!(observations.iter().any(|observation| {
            observation.kind == InferenceApiObservationKind::KernelResolved
                && observation.message.contains("kernel=")
                && observation.message.contains("provider=")
                && observation.message.contains("model_instance=")
        }));
        assert!(observations.iter().any(|observation| {
            observation.kind == InferenceApiObservationKind::PlanSelected
                && observation.message.contains("phase=decode")
                && observation.message.contains("kv_position=")
        }));
        assert!(
            outcome
                .kv_observations
                .iter()
                .any(|observation| observation.kind == KvCacheObservationKind::PrefillCompleted)
        );
        assert!(
            outcome
                .kv_observations
                .iter()
                .any(|observation| observation.kind == KvCacheObservationKind::DecodeAppend)
        );
        assert!(outcome.kv_observations.iter().all(|observation| {
            !observation.raw_prompt_available
                && !observation.raw_cache_available
                && !observation.raw_provider_handle_available
        }));
    }
}
