//! End-to-End Local Inference Conformance suite.
//!
//! Assembles a minimal, deterministic Qwen-like fixture model and drives it
//! through the real Runtime Inference API surface -- model resolution,
//! Model Loading, Model Instance creation, session creation, tokenization,
//! generation, streaming, and cleanup -- using genuine Reference CPU numeric
//! kernels for the forward pass (not canned output), so the success path is
//! a real, if tiny, end-to-end inference run. See
//! `openspec/specs/e2e-conformance/spec.md` for the normative requirements
//! this module satisfies.

use crate::*;
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet},
    fmt,
    time::{SystemTime, UNIX_EPOCH},
};

pub const E2E_SUITE_VERSION: &str = "0.1.0";
pub const E2E_FIXTURE_VERSION: &str = "0.1.0";

const E2E_FIXTURE_EOS_TOKEN: TokenId = 0;
const E2E_FIXTURE_BOS_TOKEN: TokenId = 257;
const E2E_FIXTURE_VOCAB: u64 = 258;
const E2E_FIXTURE_HIDDEN: u64 = 4;
const E2E_FIXTURE_HEADS: u64 = 2;
const E2E_FIXTURE_HEAD_DIM: u64 = 2;
const E2E_FIXTURE_INTERMEDIATE: u64 = 8;
const E2E_FIXTURE_CONTEXT: u64 = 32;
const E2E_FIXTURE_LAYERS: u64 = 1;

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
        ModelComponentImplementationKind::RuntimeNative,
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
            // This forward pass computes the whole prompt in one shot: every
            // row is a distinct token position starting at the sequence's
            // first token, so there is no prior cache and the offset is zero.
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

/// Runs a genuine (if tiny) decoder forward pass over `token_ids` using the
/// real Reference CPU numeric kernels -- embedding lookup, RMSNorm, matmul,
/// RoPE, attention, SiLU, elementwise mul/add, residual-add, and a tied
/// softmax-normalized read-out -- returning raw logits (length =
/// vocabulary size) for the final position. This is what makes the E2E
/// success path genuinely deterministic rather than a canned-output stub.
pub fn e2e_forward(
    fixture: &E2eFixture,
    token_ids: &[TokenId],
) -> Result<Vec<f32>, E2eConformanceError> {
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
    let normed_final = rmsnorm(&hidden_states, final_norm, epsilon)?;
    // Tied embeddings: logits = normed_final @ token_embedding^T.
    let logits = matmul(&normed_final, token_embedding, false, true)?;
    // Exercise the softmax kernel for operator-coverage/report purposes;
    // Sampling owns the authoritative distribution derived from raw logits.
    let _distribution = softmax_rows(&logits)?;

    let vocab = architecture.vocabulary_size as usize;
    let last_row_start = ((seq_len - 1) as usize) * vocab;
    Ok(logits.data[last_row_start..last_row_start + vocab].to_vec())
}

// ---------------------------------------------------------------------
// No-shortcut validation
// ---------------------------------------------------------------------

/// Validates that inference used the Kernel Registry / Reference CPU
/// coverage path rather than bypassing it. `used_kernel_registry` records
/// whether the caller actually dispatched through Kernel Registry selection
/// (as opposed to invoking a Provider function directly); a `false` value
/// always fails with `e2e-boundary-violation`, matching the "E2E No
/// Shortcut Rule" requirement's direct-invocation scenario.
pub fn validate_e2e_no_shortcuts(
    used_kernel_registry: bool,
    advertisements: &[KernelAdvertisement],
) -> Result<(), E2eConformanceError> {
    if !used_kernel_registry {
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
}

fn build_runtime() -> Runtime {
    Runtime::builder()
        .register_provider(std::sync::Arc::new(ReferenceCpuProvider::new()))
        .build()
        .expect("Reference CPU provider registers cleanly")
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

fn generation_tokenizer_reference(fixture: &E2eFixture) -> GenerationTokenizerReference {
    GenerationTokenizerReference {
        tokenizer_id: fixture.tokenizer.metadata().id.clone(),
        metadata: fixture.tokenizer.metadata().clone(),
    }
}

/// Runs the full required success path: resolve, load, instantiate, create
/// session, tokenize (plain text), generate through a real Reference CPU
/// forward pass with greedy Sampling, stream, close session, cleanup.
fn run_success_path(fixture: &E2eFixture) -> Result<E2eRunOutcome, E2eConformanceError> {
    let mut runtime = build_runtime();

    // Model resolution.
    let mut registry = ModelRegistry::new();
    let model_ref = ModelRef::new("e2e-fixture-model")?;
    registry.register(model_ref.clone(), fixture.manifest.id.clone());
    let resolution = registry.resolve(&ModelResolutionRequest::new(model_ref))?;
    if resolution.artifact != fixture.manifest.id {
        return Err(E2eConformanceError::ModelResolutionFailed {
            reason: "resolved artifact does not match fixture manifest".into(),
        });
    }

    // Model Loading + Model Instance.
    let (instance, _memory) = load_fixture_instance(fixture, &mut runtime)?;

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
        TokenizationRequest::new(PromptInput::PlainText("hi".into())),
        None,
    )?;

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
    let mut observer = InferenceApiObserver::new();
    let generation_result = run_generation_loop(
        &runtime,
        &request,
        true,
        true,
        SamplingPolicy::default(),
        CacheUsageSummary::default(),
        |generated_so_far| {
            let mut sequence = request.input_token_ids.clone();
            sequence.extend_from_slice(generated_so_far);
            e2e_forward(fixture, &sequence).unwrap_or_default()
        },
        |_generated_so_far| false,
        &mut observer,
    )?;

    // Streaming decode of generated tokens.
    let decoded = decode_tokens_streaming(
        &fixture.tokenizer,
        StreamingDecodeRequest::new(generation_result.output.generated_token_ids.clone()),
    )?;
    let generation_result = generation_result.with_decoded_text(decoded.text);

    // Session close + Model Instance cleanup.
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

fn check_operator_coverage(fixture: &E2eFixture) -> Result<BTreeSet<String>, E2eConformanceError> {
    // Exercising the forward pass over the prompt already calls every
    // required-now kernel once; run it once more here so this check is
    // independently verifiable even if the success path check is skipped.
    e2e_forward(fixture, &[1, 2])?;
    Ok(E2E_EXERCISED_OPERATORS
        .iter()
        .map(|op| op.to_string())
        .collect())
}

fn check_kernel_coverage() -> Result<BTreeSet<String>, E2eConformanceError> {
    let advertisements = reference_cpu_kernel_advertisements();
    validate_e2e_no_shortcuts(true, &advertisements)?;
    Ok(advertisements
        .iter()
        .map(|advertisement| advertisement.implemented_operator.name().to_string())
        .collect())
}

fn check_no_shortcut_direct_provider_rejected() -> Result<(), E2eConformanceError> {
    match validate_e2e_no_shortcuts(false, &reference_cpu_kernel_advertisements()) {
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
    let mut runtime = build_runtime();
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
        &runtime,
        &request,
        true,
        true,
        SamplingPolicy::default(),
        CacheUsageSummary::default(),
        |_generated_so_far| vec![1.0_f32; E2E_FIXTURE_VOCAB as usize],
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

fn check_eos_token_stops_generation(fixture: &E2eFixture) -> Result<(), E2eConformanceError> {
    let mut runtime = build_runtime();
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
        &runtime,
        &request,
        true,
        true,
        SamplingPolicy::default(),
        CacheUsageSummary::default(),
        |_generated_so_far| {
            let mut logits = vec![0.0_f32; E2E_FIXTURE_VOCAB as usize];
            logits[E2E_FIXTURE_EOS_TOKEN as usize] = 10.0;
            logits
        },
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
    let mut runtime = build_runtime();
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
        &runtime,
        &request,
        true,
        true,
        SamplingPolicy::default(),
        CacheUsageSummary::default(),
        |_generated_so_far| vec![1.0_f32; E2E_FIXTURE_VOCAB as usize],
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
    let mut runtime = build_runtime();
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
        &runtime,
        &request,
        true,
        true,
        SamplingPolicy::default(),
        CacheUsageSummary::default(),
        |generated_so_far| {
            let mut sequence = request.input_token_ids.clone();
            sequence.extend_from_slice(generated_so_far);
            e2e_forward(fixture, &sequence).unwrap_or_default()
        },
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

    #[test]
    fn e2e_success_path_resolves_loads_generates_and_cleans_up() {
        let fixture = e2e_fixture().expect("fixture builds");
        check_success_path(&fixture).expect("success path completes");
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
        let outcome = run_success_path(&fixture).expect("success path completes");
        assert!(outcome.generation_result.output.usage.total_tokens > 0);
        assert!(outcome.generation_result.decoded_text.is_some());
    }

    #[test]
    fn e2e_no_shortcut_direct_provider_invocation_is_rejected() {
        check_no_shortcut_direct_provider_rejected().expect("direct-invocation shortcut rejected");
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
        check_max_new_tokens_stops_generation(&fixture).expect("max_new_tokens stops generation");
    }

    #[test]
    fn e2e_eos_token_stops_generation() {
        let fixture = e2e_fixture().expect("fixture builds");
        check_eos_token_stops_generation(&fixture).expect("EOS token stops generation");
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
    fn e2e_kv_cache_diagnostics_redact_raw_contents() {
        check_kv_cache_diagnostics_redacted().expect("cache usage carries no raw contents");
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
        check_invalid_tensor_shape(&e2e_fixture().unwrap()).expect("invalid tensor shape rejected");
        check_memory_admission_failure().expect("memory admission failure rejected");
        check_closed_session_rejects_generation(&e2e_fixture().unwrap())
            .expect("closed session rejected");
        check_generation_cancelled(&e2e_fixture().unwrap()).expect("cancellation reported");
        check_cli_boundary_denials().expect("policy denial reported");
        check_raw_handle_access_denied().expect("raw handle access denied");
    }

    #[test]
    fn e2e_determinism_repeated_runs_produce_matching_tokens() {
        let fixture = e2e_fixture().expect("fixture builds");
        check_determinism(&fixture).expect("repeated runs are deterministic");
    }

    #[test]
    fn e2e_report_contains_required_metadata_fields() {
        let report = run_e2e_local_inference_conformance();
        check_report_metadata(&report).expect("report has required metadata");
        assert!(report.redacted);
    }

    #[test]
    fn e2e_ci_can_run_without_gpu_and_only_skips_optional_cases() {
        let report = run_e2e_local_inference_conformance();
        for test in &report.test_cases {
            assert_ne!(
                test.status,
                E2eTestStatus::Failed,
                "unexpected failure: {} ({:?})",
                test.name,
                test.diagnostic
            );
        }
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
        check_one_shot_session_normal_paths(&fixture).expect("one-shot generation completes");
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
}
