//! Runtime Inference API: the stable, inference-only façade Magnetar exposes
//! to first-party and external callers (see
//! `openspec/changes/define-runtime-inference-api`).
//!
//! This module does not implement a new execution engine. It composes the
//! already Runtime-owned contracts ([`crate::session`], [`crate::generation`],
//! [`crate::tokenizer`], [`crate::model_instance`], [`crate::model_loading`],
//! [`crate::kv_cache`], [`crate::prefix_cache`], [`crate::adapter`],
//! [`crate::batching`]) behind a boundary that:
//!
//! - never exposes raw Provider handles, Device handles, Kernel handles,
//!   tensor pointers, memory pointers, raw KV cache contents, or raw model
//!   weights,
//! - is limited to inference responsibilities (model resolution, loading,
//!   session, tokenization, generation, streaming, cancellation,
//!   diagnostics, usage reporting), and
//! - never grants workspace filesystem, arbitrary filesystem, Git, network
//!   tool, shell, process execution, secret, external service, source
//!   editing, or agent orchestration authority.
//!
//! Those responsibilities remain owned by clients such as `magnetar-cli`.
//! Tachyon may call this API through an adapter boundary but SHALL not
//! bypass Runtime validation, Model Instance lifecycle, Kernel Registry,
//! Memory Manager, or Provider contracts.

use crate::adapter::*;
use crate::affinity::*;
use crate::batching::*;
use crate::generation::*;
use crate::kernel::*;
use crate::kernel_dispatch::*;
use crate::kernel_execution_plan::{
    PlanGuardContext, PreparedExecutionPhase, PreparedExecutionPlan, PreparedExecutionPlanError,
};
use crate::kv_cache::*;
use crate::memory::*;
use crate::model::*;
use crate::model_instance::*;
use crate::model_loading::*;
use crate::observability::*;
use crate::operator::*;
use crate::prefix_cache::*;
use crate::runtime::*;
use crate::sampling::*;
use crate::session::*;
use crate::tokenizer::{
    DecodeInput, DecodeOutput, EncodeInput, StreamingDecodeState, TokenId, TokenOffset, Tokenizer,
    TokenizerCompatibility, TokenizerDiagnostic, TokenizerError, TokenizerId, TruncationPolicy,
};
use std::{collections::BTreeMap, error::Error, fmt, sync::Arc};

// ---------------------------------------------------------------------
// Inference-only scope boundary
// ---------------------------------------------------------------------

/// Capability substrings that are outside Runtime Inference API scope.
///
/// Used to reject caller-supplied capability/scope strings (session
/// `allowed_capabilities`, adapter activation scopes, diagnostics filters)
/// that would reach into `magnetar-cli`-owned responsibilities.
pub const FORBIDDEN_INFERENCE_API_SCOPES: &[&str] = &[
    "workspace-filesystem",
    "workspace",
    "arbitrary-filesystem",
    "filesystem",
    "git",
    "network-tool",
    "shell",
    "process-execution",
    "process",
    "secrets",
    "secret",
    "external-service",
    "source-editing",
    "agent-orchestration",
    "task-automation",
    "tool-call",
    // Kernel Optimization Orchestration boundary (see
    // `crate::kernel_optimization_orchestration` and
    // `openspec/changes/define-kernel-optimization-orchestration-boundary`):
    // optimization-agent/tooling authority SHALL NOT become ambient Runtime
    // Inference API authority.
    "optimization-agent",
    "kernel-optimization-orchestration",
    "kernel-source-injection",
    "compiler-command",
    "benchmark-script",
    "optimization-service-url",
    "repository-credential",
    "generator-credential",
    "agent-prompt",
];

/// Rejects a caller-supplied capability/scope string that names a
/// responsibility outside the Runtime Inference API boundary.
pub fn validate_inference_scope(capability: &str) -> Result<(), InferenceApiError> {
    let normalized = capability.trim().to_ascii_lowercase();
    if normalized.is_empty() {
        return Err(InferenceApiError::PolicyDenied {
            reason: "capability scope must not be empty".into(),
        });
    }
    if FORBIDDEN_INFERENCE_API_SCOPES
        .iter()
        .any(|forbidden| normalized.contains(forbidden))
    {
        return Err(InferenceApiError::PolicyDenied {
            reason: format!("capability '{capability}' is outside Runtime Inference API scope"),
        });
    }
    Ok(())
}

/// Validates a set of session/adapter capability scopes against the
/// inference-only boundary. Used by callers building
/// [`SessionCreationRequest::allowed_capabilities`] or adapter activation
/// scopes from caller input.
pub fn validate_inference_scopes<'a>(
    capabilities: impl IntoIterator<Item = &'a str>,
) -> Result<(), InferenceApiError> {
    for capability in capabilities {
        validate_inference_scope(capability)?;
    }
    Ok(())
}

// ---------------------------------------------------------------------
// Model reference and resolution
// ---------------------------------------------------------------------

/// An opaque, caller-supplied model reference (for example `"qwen-test"`).
///
/// A `ModelRef` never grants filesystem access on its own: it is only
/// meaningful once resolved through a [`ModelRegistry`] into a
/// Runtime-known [`ModelArtifactId`].
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ModelRef(String);

impl ModelRef {
    pub fn new(value: impl Into<String>) -> Result<Self, InferenceApiError> {
        let value = value.into();
        validate_model_reference(&value)?;
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for ModelRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

fn validate_model_reference(value: &str) -> Result<(), InferenceApiError> {
    if value.trim().is_empty() {
        return Err(InferenceApiError::ModelReferenceInvalid {
            reason: "model reference must not be empty".into(),
        });
    }
    if value.contains('/')
        || value.contains('\\')
        || value.contains("..")
        || value.contains(':')
        || value.starts_with('.')
    {
        return Err(InferenceApiError::ModelReferenceInvalid {
            reason: "model reference must not resemble a filesystem path".into(),
        });
    }
    if !value
        .bytes()
        .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'-' | b'_'))
    {
        return Err(InferenceApiError::ModelReferenceInvalid {
            reason: "model reference must be opaque and portable".into(),
        });
    }
    Ok(())
}

/// Runtime-owned local registry mapping caller-supplied [`ModelRef`]s to
/// Runtime-known [`ModelArtifactId`]s. Corresponds to the "local Runtime
/// registry" resolution target; other resolution targets (client-provided
/// artifact reference, trusted cache, development fixture, future external
/// or Tachyon sources) are modeled by callers supplying a
/// [`ModelResolutionRequest::artifact_hint`] directly.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ModelRegistry {
    entries: BTreeMap<ModelRef, ModelArtifactId>,
}

impl ModelRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register(&mut self, reference: ModelRef, artifact: ModelArtifactId) {
        self.entries.insert(reference, artifact);
    }

    pub fn entries(&self) -> impl Iterator<Item = (&ModelRef, &ModelArtifactId)> {
        self.entries.iter()
    }

    pub fn resolve(
        &self,
        request: &ModelResolutionRequest,
    ) -> Result<ModelResolutionResult, InferenceApiError> {
        if matches!(
            request.source,
            ModelResolutionSource::FutureExternalSource
                | ModelResolutionSource::FutureTachyonSource
        ) {
            return Err(InferenceApiError::ModelResolutionFailed {
                reason: format!(
                    "resolution source {:?} is a placeholder and not yet implemented",
                    request.source
                ),
            });
        }
        if let Some(artifact) = &request.artifact_hint {
            return Ok(ModelResolutionResult {
                artifact: artifact.clone(),
                correlation_id: request.correlation_id.clone(),
            });
        }
        self.entries
            .get(&request.reference)
            .cloned()
            .map(|artifact| ModelResolutionResult {
                artifact,
                correlation_id: request.correlation_id.clone(),
            })
            .ok_or_else(|| InferenceApiError::ModelResolutionFailed {
                reason: format!("model reference '{}' is not registered", request.reference),
            })
    }

    /// [`ModelRegistry::resolve`] plus [`InferenceApiObserver`] emission of
    /// `ModelResolved`/`ModelResolutionFailed`.
    pub fn resolve_observed(
        &self,
        request: &ModelResolutionRequest,
        observer: &mut InferenceApiObserver,
    ) -> Result<ModelResolutionResult, InferenceApiError> {
        match self.resolve(request) {
            Ok(result) => {
                observer.observe(
                    InferenceApiObservationKind::ModelResolved,
                    format!("model reference '{}' resolved", request.reference),
                    result.correlation_id.clone(),
                );
                Ok(result)
            }
            Err(error) => {
                observer.observe(
                    InferenceApiObservationKind::ModelResolutionFailed,
                    error.to_string(),
                    request.correlation_id.clone(),
                );
                Err(error)
            }
        }
    }
}

/// Resolution target a [`ModelResolutionRequest`] targets. `FutureExternalSource`
/// and `FutureTachyonSource` are placeholders for resolution targets not yet
/// implemented: resolving through them fails with a structured error rather
/// than silently falling back to another source.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ModelResolutionSource {
    LocalRegistry,
    ClientProvidedArtifact,
    TrustedCache,
    DevelopmentFixture,
    FutureExternalSource,
    FutureTachyonSource,
}

/// Request to resolve a [`ModelRef`] into Runtime-known model metadata.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ModelResolutionRequest {
    pub reference: ModelRef,
    /// A caller-provided artifact reference (client-provided artifact
    /// reference / trusted cache / development fixture resolution target),
    /// bypassing the local registry when present.
    pub artifact_hint: Option<ModelArtifactId>,
    pub source: ModelResolutionSource,
    pub correlation_id: Option<CorrelationId>,
}

impl ModelResolutionRequest {
    pub fn new(reference: ModelRef) -> Self {
        Self {
            reference,
            artifact_hint: None,
            source: ModelResolutionSource::LocalRegistry,
            correlation_id: None,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ModelResolutionResult {
    pub artifact: ModelArtifactId,
    pub correlation_id: Option<CorrelationId>,
}

// ---------------------------------------------------------------------
// Model Loading API
// ---------------------------------------------------------------------

/// Model loading request as exposed through the Runtime Inference API.
/// Wraps the core [`ModelLoadingRequest`] with the Inference-API-level
/// fields the underlying contract does not itself carry: an optional
/// tokenizer reference, optional adapter references, an explicit layout
/// policy, and non-authoritative Provider preferences (Runtime still owns
/// Provider/Device selection).
#[derive(Clone, Debug, PartialEq)]
pub struct ModelLoadingApiRequest {
    pub core: ModelLoadingRequest,
    pub tokenizer_reference: Option<TokenizerId>,
    pub adapter_references: Vec<AdapterArtifactId>,
    pub layout_policy: Option<TensorLayoutKind>,
    pub provider_preferences: Vec<ProviderBinding>,
}

impl ModelLoadingApiRequest {
    pub fn new(core: ModelLoadingRequest) -> Self {
        Self {
            core,
            tokenizer_reference: None,
            adapter_references: Vec::new(),
            layout_policy: None,
            provider_preferences: Vec::new(),
        }
    }
}

/// Drives [`ModelLoadingCoordinator::load`] on behalf of the Runtime
/// Inference API, translating [`ModelLoadingError`] into
/// [`InferenceApiError`]. Model loading is explicit here; policy-controlled
/// implicit loading is expressed by callers invoking this from a one-shot
/// or session-creation code path.
pub fn load_model(
    coordinator: &mut ModelLoadingCoordinator,
    memory: &mut MemoryManager,
    request: ModelLoadingApiRequest,
    manifest: &ModelManifest,
    trust: &ModelTrustDecision,
) -> Result<LoadedModelContext, InferenceApiError> {
    coordinator
        .load(request.core, manifest, trust, memory)
        .map_err(InferenceApiError::from)
}

/// [`load_model`] plus [`InferenceApiObserver`] emission of
/// `ModelLoadingRequested`/`ModelLoaded`/`ModelLoadingFailed`.
pub fn load_model_observed(
    coordinator: &mut ModelLoadingCoordinator,
    memory: &mut MemoryManager,
    request: ModelLoadingApiRequest,
    manifest: &ModelManifest,
    trust: &ModelTrustDecision,
    observer: &mut InferenceApiObserver,
) -> Result<LoadedModelContext, InferenceApiError> {
    let correlation_id = request.core.correlation_id.clone().map(CorrelationId::new);
    observer.observe(
        InferenceApiObservationKind::ModelLoadingRequested,
        "model loading requested",
        correlation_id.clone(),
    );
    match load_model(coordinator, memory, request, manifest, trust) {
        Ok(loaded) => {
            observer.observe(
                InferenceApiObservationKind::ModelLoaded,
                "model loaded",
                correlation_id,
            );
            Ok(loaded)
        }
        Err(error) => {
            observer.observe(
                InferenceApiObservationKind::ModelLoadingFailed,
                error.to_string(),
                correlation_id,
            );
            Err(error)
        }
    }
}

// ---------------------------------------------------------------------
// Model Instance API
// ---------------------------------------------------------------------

/// Creates a Model Instance from a loaded model context through the
/// Runtime Inference API boundary. Thin wrapper over
/// [`Runtime::create_model_instance`] that normalizes the error type.
pub fn create_model_instance(
    runtime: &mut Runtime,
    loaded: &LoadedModelContext,
    architecture: ModelArchitectureImplementation,
    affinity: ResourceAffinity,
) -> Result<ModelInstanceId, InferenceApiError> {
    runtime
        .create_model_instance(loaded, architecture, affinity)
        .map_err(InferenceApiError::from)
}

/// [`create_model_instance`] plus [`InferenceApiObserver`] emission of
/// `ModelInstanceSelected`.
pub fn create_model_instance_observed(
    runtime: &mut Runtime,
    loaded: &LoadedModelContext,
    architecture: ModelArchitectureImplementation,
    affinity: ResourceAffinity,
    observer: &mut InferenceApiObserver,
) -> Result<ModelInstanceId, InferenceApiError> {
    let instance = create_model_instance(runtime, loaded, architecture, affinity)?;
    observer.observe(
        InferenceApiObservationKind::ModelInstanceSelected,
        format!("model instance '{instance}' created"),
        None,
    );
    Ok(instance)
}

/// Redacted Model Instance status exposed through the API boundary. This is
/// the existing [`ModelInstanceStatus`] type, which already excludes raw
/// handles; re-exported here as the API-level accessor.
pub fn model_instance_status(
    runtime: &Runtime,
    instance: &ModelInstanceId,
) -> Result<ModelInstanceStatus, InferenceApiError> {
    runtime
        .model_instance_status(instance)
        .map_err(InferenceApiError::from)
}

/// Runs the Model Instance warmup plan through the Runtime Inference API
/// boundary, without exposing Provider/Device handles.
pub fn warm_model_instance(
    runtime: &mut Runtime,
    instance: &ModelInstanceId,
    plan: &ModelInstanceWarmupPlan,
    checks: &ModelInstanceReadinessChecks,
) -> Result<(), InferenceApiError> {
    runtime
        .model_instances_mut()
        .instance_mut(instance)?
        .warmup(plan, checks)
        .map_err(InferenceApiError::from)
}

pub fn suspend_model_instance(
    runtime: &mut Runtime,
    instance: &ModelInstanceId,
    reason: ModelInstanceSuspensionReason,
) -> Result<(), InferenceApiError> {
    runtime
        .model_instances_mut()
        .instance_mut(instance)?
        .suspend(reason)
        .map_err(InferenceApiError::from)
}

pub fn resume_model_instance(
    runtime: &mut Runtime,
    instance: &ModelInstanceId,
) -> Result<(), InferenceApiError> {
    runtime
        .model_instances_mut()
        .instance_mut(instance)?
        .resume()
        .map_err(InferenceApiError::from)
}

pub fn drain_model_instance(
    runtime: &mut Runtime,
    instance: &ModelInstanceId,
) -> Result<(), InferenceApiError> {
    runtime
        .model_instances_mut()
        .instance_mut(instance)?
        .drain()
        .map_err(InferenceApiError::from)
}

pub fn unload_model_instance(
    runtime: &mut Runtime,
    instance: &ModelInstanceId,
    policy: ModelInstanceUnloadPolicy,
) -> Result<ModelInstanceUnloadReport, InferenceApiError> {
    runtime
        .unload_model_instance(instance, policy)
        .map_err(InferenceApiError::from)
}

// ---------------------------------------------------------------------
// Session API
// ---------------------------------------------------------------------

/// Creates an Inference Session through the Runtime Inference API boundary.
/// Rejects any `allowed_capabilities` entry that names a non-inference
/// responsibility before delegating to [`Runtime::create_inference_session`].
pub fn create_inference_session(
    runtime: &mut Runtime,
    request: SessionCreationRequest,
) -> Result<InferenceSessionId, InferenceApiError> {
    validate_inference_scopes(request.allowed_capabilities.iter().map(String::as_str))?;
    runtime
        .create_inference_session(request)
        .map_err(InferenceApiError::from)
}

/// [`create_inference_session`] plus [`InferenceApiObserver`] emission of
/// `SessionCreated`.
pub fn create_inference_session_observed(
    runtime: &mut Runtime,
    request: SessionCreationRequest,
    observer: &mut InferenceApiObserver,
) -> Result<InferenceSessionId, InferenceApiError> {
    let correlation_id = request.correlation_id.clone();
    let session = create_inference_session(runtime, request)?;
    observer.observe(
        InferenceApiObservationKind::SessionCreated,
        format!("session '{session}' created"),
        correlation_id,
    );
    Ok(session)
}

pub fn session_status(
    runtime: &Runtime,
    session: &InferenceSessionId,
    access: &SessionAccessPolicy,
) -> Result<SessionStatus, InferenceApiError> {
    runtime
        .session_status(session, access)
        .map_err(InferenceApiError::from)
}

pub fn close_inference_session(
    runtime: &mut Runtime,
    session: &InferenceSessionId,
) -> Result<(), InferenceApiError> {
    runtime
        .close_inference_session(session)
        .map_err(InferenceApiError::from)
}

/// [`close_inference_session`] plus [`InferenceApiObserver`] emission of
/// `SessionClosed`.
pub fn close_inference_session_observed(
    runtime: &mut Runtime,
    session: &InferenceSessionId,
    observer: &mut InferenceApiObserver,
) -> Result<(), InferenceApiError> {
    close_inference_session(runtime, session)?;
    observer.observe(
        InferenceApiObservationKind::SessionClosed,
        format!("session '{session}' closed"),
        None,
    );
    Ok(())
}

/// One-shot inference is modeled as policy-controlled implicit session
/// creation, generation admission, and session close. This helper performs
/// the create/close bracket; generation submission in between is left to
/// [`submit_generation`] so callers retain control over streaming.
pub fn create_one_shot_session(
    runtime: &mut Runtime,
    request: SessionCreationRequest,
) -> Result<InferenceSessionId, InferenceApiError> {
    validate_inference_scopes(request.allowed_capabilities.iter().map(String::as_str))?;
    runtime
        .create_one_shot_session(request)
        .map_err(InferenceApiError::from)
}

// ---------------------------------------------------------------------
// Prompt Input boundary + Tokenization API
// ---------------------------------------------------------------------

/// A single chat message. Rendering chat messages into text SHALL occur
/// only through an authorized [`ChatTemplateFormatter`]; this type carries
/// no formatting behavior of its own.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

impl ChatMessage {
    pub fn new(role: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: role.into(),
            content: content.into(),
        }
    }
}

/// Prompt input accepted by the Runtime Inference API. Plain text, chat
/// messages, already-tokenized input, and test token sequences are the only
/// accepted forms; none of them perform external retrieval, file reading,
/// workspace scanning, or tool execution.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PromptInput {
    PlainText(String),
    ChatMessages(Vec<ChatMessage>),
    TokenIds(Vec<TokenId>),
    TestTokenSequence(Vec<TokenId>),
}

/// Authorized Runtime prompt/template contract. Chat-message formatting is
/// only permitted through an implementation of this trait; the Runtime
/// Inference API never renders chat messages on its own.
pub trait ChatTemplateFormatter {
    fn format(&self, messages: &[ChatMessage]) -> Result<String, InferenceApiError>;
}

/// A tokenization request. Redaction of the raw `prompt` is the caller's
/// responsibility upstream of logging; [`TokenizationResult`] itself never
/// carries raw prompt text.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TokenizationRequest {
    pub prompt: PromptInput,
    pub add_special_tokens: bool,
    pub truncation: TruncationPolicy,
    pub max_tokens: Option<usize>,
    pub return_offsets: bool,
    pub correlation_id: Option<CorrelationId>,
}

impl TokenizationRequest {
    pub fn new(prompt: PromptInput) -> Self {
        Self {
            prompt,
            add_special_tokens: false,
            truncation: TruncationPolicy::None,
            max_tokens: None,
            return_offsets: false,
            correlation_id: None,
        }
    }
}

/// Redacted tokenization result: token IDs and usage/diagnostic metadata
/// only. Raw prompt logging is disabled by default because no field here
/// stores the original text.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TokenizationResult {
    pub token_ids: Vec<TokenId>,
    pub token_count: usize,
    pub offsets: Option<Vec<TokenOffset>>,
    pub diagnostics: Vec<TokenizerDiagnostic>,
    pub correlation_id: Option<CorrelationId>,
}

/// Tokenizes prompt input through the Tokenizer Contract, enforcing the
/// Prompt Input boundary: chat messages are rendered only via an
/// authorized `chat_formatter`, and already-tokenized input is validated
/// against the tokenizer's token ID range rather than re-encoded.
pub fn tokenize_prompt_input(
    tokenizer: &dyn Tokenizer,
    request: TokenizationRequest,
    chat_formatter: Option<&dyn ChatTemplateFormatter>,
) -> Result<TokenizationResult, InferenceApiError> {
    match &request.prompt {
        PromptInput::PlainText(text) => {
            let text = text.clone();
            encode_text(tokenizer, text, &request)
        }
        PromptInput::ChatMessages(messages) => {
            let formatter = chat_formatter.ok_or_else(|| InferenceApiError::PolicyDenied {
                reason: "chat message formatting requires an authorized Runtime prompt contract"
                    .into(),
            })?;
            let text = formatter.format(messages)?;
            encode_text(tokenizer, text, &request)
        }
        PromptInput::TokenIds(token_ids) | PromptInput::TestTokenSequence(token_ids) => {
            let token_ids = token_ids.clone();
            let range = tokenizer.metadata().token_id_range;
            for token_id in &token_ids {
                if !range.contains(*token_id) {
                    return Err(InferenceApiError::TokenizationFailed {
                        reason: format!("token id {token_id} is outside tokenizer range"),
                    });
                }
            }
            let token_count = token_ids.len();
            Ok(TokenizationResult {
                token_ids,
                token_count,
                offsets: None,
                diagnostics: Vec::new(),
                correlation_id: request.correlation_id,
            })
        }
    }
}

/// [`tokenize_prompt_input`] plus [`InferenceApiObserver`] emission of
/// `PromptTokenized`/`TokenizationFailed`.
pub fn tokenize_prompt_input_observed(
    tokenizer: &dyn Tokenizer,
    request: TokenizationRequest,
    chat_formatter: Option<&dyn ChatTemplateFormatter>,
    observer: &mut InferenceApiObserver,
) -> Result<TokenizationResult, InferenceApiError> {
    let correlation_id = request.correlation_id.clone();
    match tokenize_prompt_input(tokenizer, request, chat_formatter) {
        Ok(result) => {
            observer.observe(
                InferenceApiObservationKind::PromptTokenized,
                format!("prompt tokenized into {} tokens", result.token_count),
                correlation_id,
            );
            Ok(result)
        }
        Err(error) => {
            observer.observe(
                InferenceApiObservationKind::TokenizationFailed,
                error.to_string(),
                correlation_id,
            );
            Err(error)
        }
    }
}

fn encode_text(
    tokenizer: &dyn Tokenizer,
    text: String,
    request: &TokenizationRequest,
) -> Result<TokenizationResult, InferenceApiError> {
    let mut input = EncodeInput::new(text);
    input.add_special_tokens = request.add_special_tokens;
    input.truncation = request.truncation;
    input.max_tokens = request.max_tokens;
    input.return_offsets = request.return_offsets;
    let output = tokenizer.encode(input)?;
    Ok(TokenizationResult {
        token_ids: output.token_ids,
        token_count: output.token_count,
        offsets: output.offsets,
        diagnostics: output.diagnostics,
        correlation_id: request.correlation_id.clone(),
    })
}

/// Decodes token IDs through the Tokenizer Contract. The returned text is
/// caller-owned output, not a Runtime-logged value; default redaction
/// applies to observability, not to the direct decode result.
pub fn decode_tokens(
    tokenizer: &dyn Tokenizer,
    input: DecodeInput,
) -> Result<DecodeOutput, InferenceApiError> {
    tokenizer.decode(input).map_err(InferenceApiError::from)
}

/// A streaming decode request: decode newly generated token IDs
/// incrementally, carrying [`StreamingDecodeState`] forward between calls
/// instead of re-decoding already emitted output.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StreamingDecodeRequest {
    pub token_ids: Vec<TokenId>,
    pub streaming_state: Option<StreamingDecodeState>,
    pub skip_special_tokens: bool,
}

impl StreamingDecodeRequest {
    pub fn new(token_ids: Vec<TokenId>) -> Self {
        Self {
            token_ids,
            streaming_state: None,
            skip_special_tokens: true,
        }
    }
}

/// Decodes tokens incrementally through the Tokenizer Contract's streaming
/// decode support.
pub fn decode_tokens_streaming(
    tokenizer: &dyn Tokenizer,
    request: StreamingDecodeRequest,
) -> Result<DecodeOutput, InferenceApiError> {
    tokenizer
        .decode(DecodeInput {
            token_ids: request.token_ids,
            skip_special_tokens: request.skip_special_tokens,
            clean_up_tokenization_spaces: true,
            streaming_state: request.streaming_state,
        })
        .map_err(InferenceApiError::from)
}

/// Validates that a tokenizer is compatible with model-derived expectations
/// (digest, vocabulary size, family, ...) through the Tokenizer Contract.
pub fn validate_tokenizer_compatibility(
    tokenizer: &dyn Tokenizer,
    compatibility: &TokenizerCompatibility,
) -> Result<(), InferenceApiError> {
    tokenizer
        .metadata()
        .validate_compatibility(compatibility)
        .map_err(InferenceApiError::from)
}

// ---------------------------------------------------------------------
// Generation API
// ---------------------------------------------------------------------

/// Builds a [`GenerationRequest`] from tokenized prompt input, keeping the
/// Prompt Input -> Tokenization -> Generation pipeline explicit at the API
/// boundary. All other generation fields remain caller-supplied because
/// [`GenerationRequest`] already models them fully.
#[allow(clippy::too_many_arguments)]
pub fn build_generation_request(
    request_id: GenerationRequestId,
    session: Option<InferenceSessionId>,
    model: GenerationModelReference,
    tokenizer: GenerationTokenizerReference,
    tokenized: TokenizationResult,
    max_new_tokens: usize,
    parameters: GenerationParameters,
    stop_conditions: StopConditions,
    streaming: StreamingMode,
) -> GenerationRequest {
    GenerationRequest {
        request_id,
        session,
        model,
        tokenizer,
        prompt_token_count: tokenized.token_ids.len(),
        input_token_ids: tokenized.token_ids,
        max_new_tokens,
        max_total_tokens: None,
        model_context_length: None,
        parameters,
        stop_conditions,
        streaming,
        priority: GenerationPriority::default(),
        cancellation: CancellationMetadata::default(),
        memory: GenerationMemoryEstimate::default(),
        correlation_id: tokenized.correlation_id,
        trace_id: None,
    }
}

/// Applies session policy to a generation request (delegating to
/// [`Runtime::apply_session_to_generation`]) and validates the resulting
/// request through the Generation Contract before admission.
pub fn prepare_generation(
    runtime: &Runtime,
    mut request: GenerationRequest,
) -> Result<GenerationRequest, InferenceApiError> {
    runtime.apply_session_to_generation(&mut request)?;
    request.validate()?;
    Ok(request)
}

/// Generation request as exposed through the Runtime Inference API. Wraps
/// the core [`GenerationRequest`] with an explicit privacy/redaction policy
/// for callers -- such as one-shot inference -- that have no
/// [`SessionPolicy`] to inherit a redaction policy from.
#[derive(Clone, Debug, PartialEq)]
pub struct GenerationApiRequest {
    pub core: GenerationRequest,
    pub privacy: SessionRedactionPolicy,
}

impl GenerationApiRequest {
    pub fn new(core: GenerationRequest, privacy: SessionRedactionPolicy) -> Self {
        Self { core, privacy }
    }
}

/// KV cache / Prefix Cache reuse summary attached to a [`GenerationResult`],
/// never exposing raw cache contents -- hit/miss booleans only.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct CacheUsageSummary {
    pub kv_cache_hit: Option<bool>,
    pub prefix_cache_hit: Option<bool>,
}

/// Runtime-produced evidence that one generation step used the architectural
/// execution path instead of a caller-owned logits shortcut.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct RuntimeGenerationExecutionEvidence {
    pub(crate) model_instance_ready: bool,
    pub(crate) graph_validated: bool,
    pub(crate) kernel_selected: bool,
    pub(crate) kernel_dispatched: bool,
    pub(crate) provider_executed: bool,
    pub(crate) tensor_resource_used: bool,
    pub(crate) context: Vec<String>,
}

impl RuntimeGenerationExecutionEvidence {
    pub fn untrusted() -> Self {
        Self {
            model_instance_ready: false,
            graph_validated: false,
            kernel_selected: false,
            kernel_dispatched: false,
            provider_executed: false,
            tensor_resource_used: false,
            context: Vec::new(),
        }
    }

    #[cfg(test)]
    pub(crate) fn complete() -> Self {
        Self {
            model_instance_ready: true,
            graph_validated: true,
            kernel_selected: true,
            kernel_dispatched: true,
            provider_executed: true,
            tensor_resource_used: true,
            context: Vec::new(),
        }
    }

    pub fn validate(self) -> Result<(), InferenceApiError> {
        if !self.model_instance_ready {
            return Err(InferenceApiError::ModelInstanceNotReady {
                reason: "runtime execution did not prove model instance readiness".into(),
            });
        }
        if !self.graph_validated {
            return Err(InferenceApiError::GraphPlanningFailed {
                reason: "runtime execution did not validate the execution graph".into(),
            });
        }
        if !self.kernel_selected {
            return Err(InferenceApiError::KernelUnavailable {
                reason: "runtime execution did not select a kernel".into(),
            });
        }
        if !self.kernel_dispatched {
            return Err(InferenceApiError::KernelUnavailable {
                reason: "runtime execution did not dispatch a kernel".into(),
            });
        }
        if !self.provider_executed {
            return Err(InferenceApiError::ProviderUnavailable {
                reason: "runtime execution did not execute a Provider".into(),
            });
        }
        if !self.tensor_resource_used {
            return Err(InferenceApiError::GenerationFailed {
                reason: "runtime execution did not produce Runtime-owned tensor logits".into(),
            });
        }
        Ok(())
    }

    pub fn from_dispatch_result(
        dispatch: &KernelDispatchResult,
        model_instance_ready: bool,
        graph_validated: bool,
    ) -> Self {
        let dispatch_succeeded = dispatch.status == KernelResultStatus::Succeeded;
        let tensor_resource_used = dispatch
            .updated_resources
            .iter()
            .any(|resource| dispatch.output_readiness.get(resource.id.as_str()) == Some(&true))
            || dispatch.output_readiness.values().any(|ready| *ready);

        Self {
            model_instance_ready,
            graph_validated,
            kernel_selected: true,
            kernel_dispatched: dispatch_succeeded,
            provider_executed: dispatch_succeeded,
            tensor_resource_used,
            context: vec![
                format!("kernel={}", dispatch.selected_kernel.stable_key()),
                format!("provider={}", dispatch.provider),
            ],
        }
    }

    pub fn with_context(mut self, context: impl Into<String>) -> Self {
        let context = context.into();
        if !context.is_empty() && self.context.len() < 16 {
            self.context.push(context);
        }
        self
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct RuntimeModelExecutionStep {
    pub(crate) logits: Vec<f32>,
    pub(crate) evidence: RuntimeGenerationExecutionEvidence,
    pub(crate) kv_commit: Option<RuntimeKvCacheCommit>,
    /// Per-node causal-chain events (Correctif 17 / task group 17) captured
    /// during this step's graph dispatch, turned into redacted
    /// `InferenceApiObservation`s by the generation loop that calls
    /// `execute_generation_step`.
    pub(crate) node_events: Vec<crate::first_native_runtime::PerNodeCausalEvent>,
}

impl RuntimeModelExecutionStep {
    pub(crate) fn new(logits: Vec<f32>, evidence: RuntimeGenerationExecutionEvidence) -> Self {
        Self {
            logits,
            evidence,
            kv_commit: None,
            node_events: Vec::new(),
        }
    }

    pub(crate) fn with_kv_commit(mut self, commit: RuntimeKvCacheCommit) -> Self {
        self.kv_commit = Some(commit);
        self
    }

    pub(crate) fn with_node_events(
        mut self,
        node_events: Vec<crate::first_native_runtime::PerNodeCausalEvent>,
    ) -> Self {
        self.node_events = node_events;
        self
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum RuntimeKvCacheCommit {
    PrefillCompleted { cache: KvCacheId, tokens: u32 },
    DecodeAppended { cache: KvCacheId, tokens: u32 },
}

pub(crate) struct RuntimeGenerationExecutionPlans<'a> {
    pub(crate) prefill: &'a mut PreparedExecutionPlan,
    pub(crate) decode: &'a mut PreparedExecutionPlan,
}

fn runtime_generation_plan_error(error: PreparedExecutionPlanError) -> InferenceApiError {
    InferenceApiError::KernelUnavailable {
        reason: format!("prepared execution plan unavailable: {error}"),
    }
}

fn runtime_generation_plan_context(
    phase: PreparedExecutionPhase,
    request: &GenerationRequest,
    generated_tokens: &[TokenId],
) -> PlanGuardContext {
    let token_count = match phase {
        PreparedExecutionPhase::Prefill => request.input_token_ids.len() as u64,
        PreparedExecutionPhase::Decode => 1,
        _ => request
            .input_token_ids
            .len()
            .saturating_add(generated_tokens.len())
            .max(1) as u64,
    };
    let mut context = PlanGuardContext::for_phase(phase);
    context.sequence_length = Some(token_count.max(1));
    context.total_tokens = Some(
        request
            .input_token_ids
            .len()
            .saturating_add(generated_tokens.len())
            .saturating_add(1)
            .max(1) as u64,
    );
    context.affinity = Some(ResourceAffinity::new(FallbackClass::Transparent));
    context.provider_ready = true;
    context.device_ready = true;
    context.memory_feasible = true;
    context
}

/// Runtime-owned execution hook used by the Runtime Inference API to produce
/// logits. Callers configure this when constructing a Runtime; normal
/// generation does not accept per-request callbacks or readiness booleans.
pub(crate) trait RuntimeModelExecutionEngine: Send + Sync {
    fn execute_generation_step(
        &self,
        runtime: &mut Runtime,
        request: &GenerationRequest,
        generated_tokens: &[TokenId],
        execution_plan: Option<&mut PreparedExecutionPlan>,
    ) -> Result<RuntimeModelExecutionStep, InferenceApiError>;

    fn commit_generation_step(
        &self,
        _runtime: &mut Runtime,
        _request: &GenerationRequest,
        _generated_tokens_before_step: &[TokenId],
        _accepted_token: TokenId,
        _step: &RuntimeModelExecutionStep,
    ) -> Result<(), InferenceApiError> {
        Ok(())
    }
}

#[derive(Clone)]
pub(crate) struct SharedRuntimeModelExecutionEngine(Arc<dyn RuntimeModelExecutionEngine>);

impl SharedRuntimeModelExecutionEngine {
    pub(crate) fn new(executor: Arc<dyn RuntimeModelExecutionEngine>) -> Self {
        Self(executor)
    }

    pub(crate) fn execute_generation_step(
        &self,
        runtime: &mut Runtime,
        request: &GenerationRequest,
        generated_tokens: &[TokenId],
        execution_plan: Option<&mut PreparedExecutionPlan>,
    ) -> Result<RuntimeModelExecutionStep, InferenceApiError> {
        self.0
            .execute_generation_step(runtime, request, generated_tokens, execution_plan)
    }

    pub(crate) fn commit_generation_step(
        &self,
        runtime: &mut Runtime,
        request: &GenerationRequest,
        generated_tokens_before_step: &[TokenId],
        accepted_token: TokenId,
        step: &RuntimeModelExecutionStep,
    ) -> Result<(), InferenceApiError> {
        self.0.commit_generation_step(
            runtime,
            request,
            generated_tokens_before_step,
            accepted_token,
            step,
        )
    }
}

/// Generation result exposed through the Runtime Inference API. Wraps the
/// existing [`GenerationOutput`] contract with decoded text (where
/// requested), Model Instance metadata, cache usage, redaction status, and
/// error information -- while still never exposing raw internal handles.
#[derive(Clone, Debug, PartialEq)]
pub struct GenerationResult {
    pub output: GenerationOutput,
    pub decoded_text: Option<String>,
    pub model_instance: Option<ModelInstanceId>,
    pub cache_usage: CacheUsageSummary,
    pub redacted: bool,
    pub error: Option<String>,
}

impl GenerationResult {
    pub fn new(output: GenerationOutput) -> Self {
        let error = matches!(
            output.finish_reason,
            FinishReason::ProviderError | FinishReason::ModelError | FinishReason::Error
        )
        .then(|| format!("generation finished with {:?}", output.finish_reason));
        Self {
            output,
            decoded_text: None,
            model_instance: None,
            cache_usage: CacheUsageSummary::default(),
            redacted: true,
            error,
        }
    }

    pub fn with_decoded_text(mut self, text: String) -> Self {
        self.decoded_text = Some(text);
        self
    }

    pub fn with_model_instance(mut self, instance: ModelInstanceId) -> Self {
        self.model_instance = Some(instance);
        self
    }

    pub fn with_cache_usage(mut self, cache_usage: CacheUsageSummary) -> Self {
        self.cache_usage = cache_usage;
        self
    }
}

/// Drives a generation request through the Runtime-owned generation execution
/// boundary, then Sampling and streaming observation emission. Logits come from
/// the [`RuntimeModelExecutionEngine`] attached to the [`Runtime`], so callers do
/// not provide readiness booleans or executable logits callbacks per request.
fn observe_generation_execution_error(
    observer: &mut InferenceApiObserver,
    correlation_id: Option<CorrelationId>,
    error: &InferenceApiError,
) {
    match error {
        InferenceApiError::ProviderUnavailable { .. } => observer.observe(
            InferenceApiObservationKind::ProviderUnavailable,
            "provider unavailable for generation",
            correlation_id.clone(),
        ),
        InferenceApiError::KernelUnavailable { .. } => observer.observe(
            InferenceApiObservationKind::KernelUnavailable,
            "kernel unavailable for generation",
            correlation_id.clone(),
        ),
        _ => observer.observe(
            InferenceApiObservationKind::GenerationFailed,
            "runtime execution failed",
            correlation_id.clone(),
        ),
    }
    observer.observe(
        InferenceApiObservationKind::StreamInterrupted,
        "stream interrupted by runtime execution failure",
        correlation_id,
    );
}

fn observation_message(base: &str, context: &[String]) -> String {
    if context.is_empty() {
        return base.to_string();
    }
    let bounded = context
        .iter()
        .take(8)
        .map(|item| item.as_str())
        .collect::<Vec<_>>()
        .join(" ");
    format!("{base}; {bounded}")
}

pub(crate) fn generation_decode_absolute_position(
    prompt_token_count: usize,
    generated_token_count: usize,
) -> Option<usize> {
    generated_token_count
        .checked_sub(1)
        .map(|generated_index| prompt_token_count + generated_index)
}

pub fn run_generation_loop(
    runtime: &mut Runtime,
    request: &GenerationRequest,
    sampling_policy: SamplingPolicy,
    cache_usage: CacheUsageSummary,
    mut should_cancel: impl FnMut(&[TokenId]) -> bool,
    observer: &mut InferenceApiObserver,
) -> Result<GenerationResult, InferenceApiError> {
    run_generation_loop_inner(
        runtime,
        request,
        sampling_policy,
        cache_usage,
        &mut should_cancel,
        observer,
        None,
    )
}

pub(crate) fn run_generation_loop_with_execution_plans(
    runtime: &mut Runtime,
    request: &GenerationRequest,
    sampling_policy: SamplingPolicy,
    cache_usage: CacheUsageSummary,
    mut should_cancel: impl FnMut(&[TokenId]) -> bool,
    observer: &mut InferenceApiObserver,
    execution_plans: &mut RuntimeGenerationExecutionPlans<'_>,
) -> Result<GenerationResult, InferenceApiError> {
    run_generation_loop_inner(
        runtime,
        request,
        sampling_policy,
        cache_usage,
        &mut should_cancel,
        observer,
        Some(execution_plans),
    )
}

fn run_generation_loop_inner(
    runtime: &mut Runtime,
    request: &GenerationRequest,
    sampling_policy: SamplingPolicy,
    cache_usage: CacheUsageSummary,
    should_cancel: &mut impl FnMut(&[TokenId]) -> bool,
    observer: &mut InferenceApiObserver,
    mut execution_plans: Option<&mut RuntimeGenerationExecutionPlans<'_>>,
) -> Result<GenerationResult, InferenceApiError> {
    let correlation_id = request.correlation_id.clone();
    observer.observe(
        InferenceApiObservationKind::GenerationStarted,
        "generation started",
        correlation_id.clone(),
    );
    observer.observe(
        InferenceApiObservationKind::StreamOpened,
        "stream opened",
        correlation_id.clone(),
    );

    let executor = runtime.model_execution_engine().cloned().ok_or_else(|| {
        InferenceApiError::ProviderUnavailable {
            reason: "no Runtime generation executor is registered".into(),
        }
    });
    let executor = match executor {
        Ok(executor) => executor,
        Err(error) => {
            observer.observe(
                InferenceApiObservationKind::ProviderUnavailable,
                "provider unavailable for generation",
                correlation_id.clone(),
            );
            observer.observe(
                InferenceApiObservationKind::StreamInterrupted,
                "stream interrupted because no Runtime generation executor is registered",
                correlation_id.clone(),
            );
            return Err(error);
        }
    };

    if runtime.kernel_registry().entries().next().is_none() {
        observer.observe(
            InferenceApiObservationKind::KernelUnavailable,
            "kernel unavailable for generation",
            correlation_id.clone(),
        );
        return Err(InferenceApiError::KernelUnavailable {
            reason: "kernel unavailable for generation".into(),
        });
    }

    let admission = memory_admission(request, runtime.memory())?;
    if !admission.is_admitted() {
        observer.observe(
            InferenceApiObservationKind::MemoryAdmissionFailed,
            format!("{admission:?}"),
            correlation_id.clone(),
        );
        observer.observe(
            InferenceApiObservationKind::StreamInterrupted,
            "stream interrupted by memory admission failure",
            correlation_id.clone(),
        );
        return Err(InferenceApiError::MemoryAdmissionFailed {
            reason: format!("{admission:?}"),
        });
    }

    if cache_usage.kv_cache_hit == Some(true) {
        observer.observe(
            InferenceApiObservationKind::KvCacheUsed,
            "kv cache reused",
            correlation_id.clone(),
        );
    }
    match cache_usage.prefix_cache_hit {
        Some(true) => observer.observe(
            InferenceApiObservationKind::PrefixCacheHit,
            "prefix cache hit",
            correlation_id.clone(),
        ),
        Some(false) => observer.observe(
            InferenceApiObservationKind::PrefixCacheMiss,
            "prefix cache miss",
            correlation_id.clone(),
        ),
        None => {}
    }

    if let Some(plans) = execution_plans.as_deref_mut() {
        let context =
            runtime_generation_plan_context(PreparedExecutionPhase::Prefill, request, &[]);
        if let Err(error) = plans
            .prefill
            .execute_ready_path(&context)
            .map_err(runtime_generation_plan_error)
        {
            observe_generation_execution_error(observer, correlation_id.clone(), &error);
            return Err(error);
        }
        let plan_context = vec![
            format!("request={}", request.request_id),
            format!("plan={}", plans.prefill.id),
            format!("plan_generation={}", plans.prefill.generation.value()),
            "phase=prefill".to_string(),
        ];
        observer.observe(
            InferenceApiObservationKind::PlanSelected,
            observation_message("prepared execution plan selected", &plan_context),
            correlation_id.clone(),
        );
        observer.observe(
            InferenceApiObservationKind::PlanGuardAccepted,
            observation_message("prepared execution plan guard accepted", &plan_context),
            correlation_id.clone(),
        );
    }
    prefill(request)?;
    observer.observe(
        InferenceApiObservationKind::PrefillStarted,
        "prefill started",
        correlation_id.clone(),
    );
    observer.observe(
        InferenceApiObservationKind::PrefillCompleted,
        "prefill completed",
        correlation_id.clone(),
    );

    let mut generated: Vec<TokenId> = Vec::new();
    let mut rng_state: Option<SamplingRngState> = None;
    let finish_reason = loop {
        if should_cancel(&generated) {
            observer.observe(
                InferenceApiObservationKind::GenerationCancelled,
                "generation cancelled during decode",
                correlation_id.clone(),
            );
            observer.observe(
                InferenceApiObservationKind::StreamInterrupted,
                "stream interrupted by cancellation",
                correlation_id.clone(),
            );
            break FinishReason::Cancelled;
        }
        observer.observe(
            InferenceApiObservationKind::DecodeStarted,
            format!("decode step {}", generated.len()),
            correlation_id.clone(),
        );
        if let Some(plans) = execution_plans.as_deref_mut() {
            let context = runtime_generation_plan_context(
                PreparedExecutionPhase::Decode,
                request,
                &generated,
            );
            if let Err(error) = plans
                .decode
                .execute_ready_path(&context)
                .map_err(runtime_generation_plan_error)
            {
                observe_generation_execution_error(observer, correlation_id.clone(), &error);
                return Err(error);
            }
            let plan_context = vec![
                format!("request={}", request.request_id),
                format!("plan={}", plans.decode.id),
                format!("plan_generation={}", plans.decode.generation.value()),
                "phase=decode".to_string(),
                format!(
                    "kv_position={}",
                    generation_decode_absolute_position(
                        request.input_token_ids.len(),
                        generated.len()
                    )
                    .unwrap_or(request.input_token_ids.len())
                ),
            ];
            observer.observe(
                InferenceApiObservationKind::PlanSelected,
                observation_message("prepared execution plan selected", &plan_context),
                correlation_id.clone(),
            );
            observer.observe(
                InferenceApiObservationKind::PlanGuardAccepted,
                observation_message("prepared execution plan guard accepted", &plan_context),
                correlation_id.clone(),
            );
        }
        let execution_plan = execution_plans.as_deref_mut().map(|plans| {
            if generated.is_empty() {
                &mut *plans.prefill
            } else {
                &mut *plans.decode
            }
        });
        let runtime_step =
            match executor.execute_generation_step(runtime, request, &generated, execution_plan) {
                Ok(runtime_step) => runtime_step,
                Err(error) => {
                    observe_generation_execution_error(observer, correlation_id.clone(), &error);
                    return Err(error);
                }
            };
        if runtime_step.evidence.model_instance_ready {
            observer.observe(
                InferenceApiObservationKind::ModelInstanceReady,
                observation_message("model instance ready", &runtime_step.evidence.context),
                correlation_id.clone(),
            );
        }
        if runtime_step.evidence.graph_validated {
            observer.observe(
                InferenceApiObservationKind::ExecutionGraphValidated,
                observation_message("execution graph validated", &runtime_step.evidence.context),
                correlation_id.clone(),
            );
            observer.observe(
                InferenceApiObservationKind::GraphValidationCompleted,
                observation_message("graph validation completed", &runtime_step.evidence.context),
                correlation_id.clone(),
            );
        }
        if runtime_step.evidence.kernel_selected {
            observer.observe(
                InferenceApiObservationKind::KernelSelected,
                observation_message("kernel selected", &runtime_step.evidence.context),
                correlation_id.clone(),
            );
            observer.observe(
                InferenceApiObservationKind::KernelResolved,
                observation_message(
                    "kernel registry resolved candidate",
                    &runtime_step.evidence.context,
                ),
                correlation_id.clone(),
            );
            observer.observe(
                InferenceApiObservationKind::KernelPrepared,
                observation_message("prepared kernel accepted", &runtime_step.evidence.context),
                correlation_id.clone(),
            );
        }
        if runtime_step.evidence.kernel_dispatched {
            observer.observe(
                InferenceApiObservationKind::KernelDispatched,
                observation_message("kernel dispatched", &runtime_step.evidence.context),
                correlation_id.clone(),
            );
            observer.observe(
                InferenceApiObservationKind::ProviderSubmitted,
                observation_message(
                    "provider submission accepted",
                    &runtime_step.evidence.context,
                ),
                correlation_id.clone(),
            );
        }
        if runtime_step.evidence.provider_executed {
            observer.observe(
                InferenceApiObservationKind::ProviderExecuted,
                observation_message("provider executed", &runtime_step.evidence.context),
                correlation_id.clone(),
            );
            observer.observe(
                InferenceApiObservationKind::ProviderCompleted,
                observation_message(
                    "provider completion observed",
                    &runtime_step.evidence.context,
                ),
                correlation_id.clone(),
            );
        }
        if runtime_step.evidence.tensor_resource_used {
            observer.observe(
                InferenceApiObservationKind::TensorLogitsProduced,
                observation_message(
                    "Runtime-owned tensor logits produced",
                    &runtime_step.evidence.context,
                ),
                correlation_id.clone(),
            );
            observer.observe(
                InferenceApiObservationKind::LogitsProduced,
                observation_message("logits produced", &runtime_step.evidence.context),
                correlation_id.clone(),
            );
        }
        // Correctif 17 / task group 17: one real `InferenceApiObservation`
        // per captured per-node causal event, correlated by `node=...`
        // (and `resource=...` where the event produces one) in the message
        // -- not just the five global evidence-category booleans above.
        // Emitted even on a later failure (the `?`s above only run once
        // `execute_generation_step` already returned `Ok`, so every event
        // it did capture before any internal failure is still here).
        for event in &runtime_step.node_events {
            let mut context = vec![
                format!("request={}", request.request_id),
                format!("node={}", event.node),
            ];
            if let Some(resource) = &event.resource {
                context.push(format!("resource={resource}"));
            }
            observer.observe(
                event.kind,
                observation_message("per-node causal event", &context),
                correlation_id.clone(),
            );
        }
        if let Err(error) = runtime_step.evidence.clone().validate() {
            observe_generation_execution_error(observer, correlation_id.clone(), &error);
            return Err(error);
        }
        let (sampling, step) = decode_step_from_sampling_with_rng(
            request,
            &generated,
            runtime_step.logits.clone(),
            sampling_policy.clone(),
            rng_state.take(),
        )
        .map_err(|error| {
            let error = InferenceApiError::from(error);
            observe_generation_execution_error(observer, correlation_id.clone(), &error);
            error
        })?;
        if let Err(error) = executor.commit_generation_step(
            runtime,
            request,
            &generated,
            step.token_id,
            &runtime_step,
        ) {
            observe_generation_execution_error(observer, correlation_id.clone(), &error);
            return Err(error);
        }
        if let Some(commit) = &runtime_step.kv_commit {
            let commit_context = match commit {
                RuntimeKvCacheCommit::PrefillCompleted { cache, tokens } => vec![
                    format!("request={}", request.request_id),
                    format!("kv_cache={cache}"),
                    format!("tokens={tokens}"),
                    "phase=prefill".to_string(),
                ],
                RuntimeKvCacheCommit::DecodeAppended { cache, tokens } => vec![
                    format!("request={}", request.request_id),
                    format!("kv_cache={cache}"),
                    format!("tokens={tokens}"),
                    format!(
                        "kv_position={}",
                        generation_decode_absolute_position(
                            request.input_token_ids.len(),
                            generated.len()
                        )
                        .unwrap_or(request.input_token_ids.len())
                    ),
                    "phase=decode".to_string(),
                ],
            };
            observer.observe(
                InferenceApiObservationKind::KvCacheCommitted,
                observation_message("kv cache committed", &commit_context),
                correlation_id.clone(),
            );
            // Correctif 17 / task group 17: one `KvUpdateCommitted` event
            // per committed layer/role resource, not just the one aggregate
            // `KvCacheCommitted` above -- `promote_pending_kv_resources`
            // (task group 9's `KvUpdateTransaction`) already published these
            // bindings to `runtime`'s KV cache by the time this runs, so
            // reading them back here needs no new plumbing through the
            // commit call itself. Correlated by `TensorResourceId`, not
            // `ExecutionNodeId`: by commit time (after sampling, a separate
            // phase from graph dispatch) the specific graph node that
            // originally produced a given layer's pending write is no
            // longer tracked, only the resource it left behind.
            let cache_id = match commit {
                RuntimeKvCacheCommit::PrefillCompleted { cache, .. }
                | RuntimeKvCacheCommit::DecodeAppended { cache, .. } => cache,
            };
            if let Ok(kv_cache) = runtime.kv_cache(cache_id) {
                for (layer, binding) in &kv_cache.layer_resources {
                    for (role, resource) in [("k", &binding.k), ("v", &binding.v)] {
                        observer.observe(
                            InferenceApiObservationKind::KvUpdateCommitted,
                            observation_message(
                                "kv update committed",
                                &[
                                    format!("kv_cache={cache_id}"),
                                    format!("layer={layer}"),
                                    format!("role={role}"),
                                    format!("resource={resource}"),
                                ],
                            ),
                            correlation_id.clone(),
                        );
                    }
                }
            }
        }
        rng_state = sampling.updated_rng_state;
        generated.push(step.token_id);
        let token_context = vec![
            format!("request={}", request.request_id),
            format!("token_index={}", step.token_index),
        ];
        observer.observe(
            InferenceApiObservationKind::SamplingCompleted,
            observation_message("sampling completed", &token_context),
            correlation_id.clone(),
        );
        observer.observe(
            InferenceApiObservationKind::TokenGenerated,
            format!("token index {} generated", step.token_index),
            correlation_id.clone(),
        );
        observer.observe(
            InferenceApiObservationKind::TokenCommitted,
            observation_message("token committed", &token_context),
            correlation_id.clone(),
        );
        if let Some(reason) = step.finish_reason {
            break reason;
        }
    };

    let output = GenerationOutput::new(request, generated, finish_reason);
    if finish_reason != FinishReason::Cancelled {
        observer.observe(
            InferenceApiObservationKind::GenerationCompleted,
            "generation completed",
            correlation_id.clone(),
        );
        observer.observe(
            InferenceApiObservationKind::StreamClosed,
            "stream closed",
            correlation_id,
        );
    }
    Ok(GenerationResult::new(output).with_cache_usage(cache_usage))
}

// ---------------------------------------------------------------------
// Backpressure / Admission
// ---------------------------------------------------------------------

/// Admission/backpressure state surfaced by the Runtime Inference API.
/// Requests are always accepted, queued, rejected, delayed, cancelled, or
/// timed out with structured metadata -- never silently dropped.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum AdmissionState {
    Accepted,
    Queued { reason: String },
    Rejected { reason: String },
    Delayed { reason: String },
    Cancelled,
    TimedOut,
}

impl From<&MemoryAdmissionDecision> for AdmissionState {
    fn from(decision: &MemoryAdmissionDecision) -> Self {
        match decision {
            MemoryAdmissionDecision::Admit { .. } => Self::Accepted,
            MemoryAdmissionDecision::Queue { reason } => Self::Queued {
                reason: reason.clone(),
            },
            MemoryAdmissionDecision::Reject { reason } => Self::Rejected {
                reason: reason.clone(),
            },
            MemoryAdmissionDecision::RetryLater { reason } => Self::Delayed {
                reason: reason.clone(),
            },
        }
    }
}

/// Admits a validated generation request into a Continuous Batch, surfacing
/// admission/backpressure as a structured [`AdmissionState`] rather than a
/// hard failure where the underlying batching error indicates admission
/// was rejected rather than internally broken.
pub fn submit_generation(
    runtime: &mut Runtime,
    batch: &BatchId,
    request: &GenerationRequest,
) -> Result<(AdmissionState, Option<BatchSlotId>), InferenceApiError> {
    match runtime.admit_generation_to_batch(batch, request) {
        Ok(slot) => {
            let state = match runtime.batching().slot(&slot) {
                Ok(admitted) if admitted.state == BatchedOperationState::Queued => {
                    AdmissionState::Queued {
                        reason: "continuous batch admission policy enqueues admitted operations"
                            .into(),
                    }
                }
                _ => AdmissionState::Accepted,
            };
            Ok((state, Some(slot)))
        }
        Err(BatchingError::BatchAdmissionRejected { reason }) => {
            Ok((AdmissionState::Rejected { reason }, None))
        }
        Err(BatchingError::StreamingBackpressure { reason }) => {
            Ok((AdmissionState::Delayed { reason }, None))
        }
        Err(BatchingError::OperationTimedOut) => Ok((AdmissionState::TimedOut, None)),
        Err(BatchingError::OperationCancelled) => Ok((AdmissionState::Cancelled, None)),
        Err(other) => Err(other.into()),
    }
}

/// [`submit_generation`] plus [`InferenceApiObserver`] emission of the
/// matching `Generation*` observation for the resulting [`AdmissionState`].
pub fn submit_generation_observed(
    runtime: &mut Runtime,
    batch: &BatchId,
    request: &GenerationRequest,
    observer: &mut InferenceApiObserver,
) -> Result<(AdmissionState, Option<BatchSlotId>), InferenceApiError> {
    let correlation_id = request.correlation_id.clone();
    let outcome = submit_generation(runtime, batch, request)?;
    let (kind, message) = match &outcome.0 {
        AdmissionState::Accepted => (
            InferenceApiObservationKind::GenerationAccepted,
            "generation accepted".to_string(),
        ),
        AdmissionState::Queued { reason } | AdmissionState::Delayed { reason } => (
            InferenceApiObservationKind::GenerationQueued,
            reason.clone(),
        ),
        AdmissionState::Rejected { reason } => (
            InferenceApiObservationKind::GenerationFailed,
            reason.clone(),
        ),
        AdmissionState::Cancelled => (
            InferenceApiObservationKind::GenerationCancelled,
            "generation cancelled".to_string(),
        ),
        AdmissionState::TimedOut => (
            InferenceApiObservationKind::GenerationFailed,
            "generation timed out".to_string(),
        ),
    };
    observer.observe(kind, message, correlation_id);
    Ok(outcome)
}

// ---------------------------------------------------------------------
// Streaming API
// ---------------------------------------------------------------------

/// Opaque handle identifying an open generation event stream. Carries no
/// Provider/Device/Kernel handle -- only the stable IDs needed to
/// correlate [`GenerationEvent`]s already produced by
/// [`crate::generation::token_stream_events`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StreamingHandle {
    pub request: GenerationRequestId,
    pub session: Option<InferenceSessionId>,
}

impl StreamingHandle {
    pub fn for_request(request: &GenerationRequest) -> Self {
        Self {
            request: request.request_id.clone(),
            session: request.session.clone(),
        }
    }
}

// ---------------------------------------------------------------------
// Cancellation API
// ---------------------------------------------------------------------

/// Opaque cancellation token for a generation request. Converts into the
/// [`CancellationMetadata`] already carried on [`GenerationRequest`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CancellationToken {
    pub request: GenerationRequestId,
    pub cancellation_id: Option<String>,
}

impl CancellationToken {
    pub fn new(request: GenerationRequestId) -> Self {
        Self {
            request,
            cancellation_id: None,
        }
    }

    pub fn into_metadata(self) -> CancellationMetadata {
        CancellationMetadata {
            cancellation_id: self.cancellation_id,
            requested: true,
        }
    }
}

/// Outcome of requesting cancellation. When a Provider or Kernel does not
/// support interruption after dispatch, Runtime SHALL report the
/// limitation rather than silently ignoring the request.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CancellationOutcome {
    Cancelled,
    LimitationReported { reason: String },
}

pub fn request_cancellation(
    _token: &CancellationToken,
    supports_interruption_after_dispatch: bool,
) -> CancellationOutcome {
    if supports_interruption_after_dispatch {
        CancellationOutcome::Cancelled
    } else {
        CancellationOutcome::LimitationReported {
            reason: "Provider/Kernel does not support interruption after dispatch".into(),
        }
    }
}

/// Execution stage a cancellation request targets. Cancellation SHALL
/// propagate through queued generation, tokenization, prefill, decode,
/// sampling, batching, graph execution, and Kernel Dispatch unconditionally;
/// only Provider execution depends on Provider/Kernel interruption support.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CancellationStage {
    Queued,
    Tokenization,
    Prefill,
    Decode,
    Sampling,
    Batching,
    GraphExecution,
    KernelDispatch,
    ProviderExecution,
}

impl CancellationStage {
    /// Whether Runtime can cancel this stage unconditionally, without
    /// depending on Provider/Kernel interruption support.
    pub const fn always_cancellable(self) -> bool {
        !matches!(self, Self::ProviderExecution)
    }
}

/// Stage-aware cancellation: every stage before Provider dispatch is always
/// cancellable; Provider execution reports a limitation when the Provider
/// or Kernel does not support interruption after dispatch.
pub fn request_cancellation_at_stage(
    token: &CancellationToken,
    stage: CancellationStage,
    supports_interruption_after_dispatch: bool,
) -> CancellationOutcome {
    if stage.always_cancellable() || supports_interruption_after_dispatch {
        CancellationOutcome::Cancelled
    } else {
        CancellationOutcome::LimitationReported {
            reason: format!(
                "Provider/Kernel does not support interruption after dispatch (stage: {stage:?}, request: {})",
                token.request
            ),
        }
    }
}

/// [`request_cancellation_at_stage`] plus [`InferenceApiObserver`] emission
/// of `GenerationCancelled` when cancellation succeeds.
pub fn request_cancellation_at_stage_observed(
    token: &CancellationToken,
    stage: CancellationStage,
    supports_interruption_after_dispatch: bool,
    observer: &mut InferenceApiObserver,
) -> CancellationOutcome {
    let outcome = request_cancellation_at_stage(token, stage, supports_interruption_after_dispatch);
    if outcome == CancellationOutcome::Cancelled {
        observer.observe(
            InferenceApiObservationKind::GenerationCancelled,
            format!(
                "generation '{}' cancelled at stage {stage:?}",
                token.request
            ),
            None,
        );
    }
    outcome
}

pub fn cancel_inference_session(
    runtime: &mut Runtime,
    session: &InferenceSessionId,
) -> Result<(), InferenceApiError> {
    runtime
        .cancel_inference_session(session)
        .map_err(InferenceApiError::from)
}

// ---------------------------------------------------------------------
// Adapter Activation API
// ---------------------------------------------------------------------

/// Activates an inference-scoped adapter, enforcing that the activation
/// scope names only an inference responsibility before delegating to
/// [`validate_adapter_activation`].
pub fn activate_adapter(
    residency: &AdapterResidency,
    request: &AdapterActivationRequest,
    session_policy: Option<&AdapterSessionPolicy>,
    batch: Option<&AdapterBatchCompatibility>,
) -> Result<(), InferenceApiError> {
    if let AdapterActivationScope::Operation(scope) = &request.scope {
        validate_inference_scope(scope)?;
    }
    validate_adapter_activation(residency, request, session_policy, batch)
        .map_err(InferenceApiError::from)
}

/// [`activate_adapter`] plus [`InferenceApiObserver`] emission of
/// `AdapterActivated`.
pub fn activate_adapter_observed(
    residency: &AdapterResidency,
    request: &AdapterActivationRequest,
    session_policy: Option<&AdapterSessionPolicy>,
    batch: Option<&AdapterBatchCompatibility>,
    observer: &mut InferenceApiObserver,
) -> Result<(), InferenceApiError> {
    activate_adapter(residency, request, session_policy, batch)?;
    observer.observe(
        InferenceApiObservationKind::AdapterActivated,
        format!("adapter set '{}' activated", request.adapter_set.as_str()),
        None,
    );
    Ok(())
}

// ---------------------------------------------------------------------
// Diagnostics API
// ---------------------------------------------------------------------

/// Resolution status for a model reference, as tracked by a caller-owned
/// [`ModelRegistry`] (Runtime itself does not retain resolution history).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ModelResolutionStatus {
    Resolved,
    Failed,
    NotAttempted,
}

/// Caller-supplied diagnostic inputs that are not derivable from `&Runtime`
/// alone, because model resolution and model loading are driven by
/// caller-owned objects ([`ModelRegistry`], [`ModelLoadingCoordinator`])
/// rather than Runtime-owned state.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct RuntimeDiagnosticsInputs {
    pub model_resolution_status: Option<ModelResolutionStatus>,
    pub model_loading_status: Option<ModelLoadingPhase>,
    pub operator_missing_count: usize,
    pub tokenizer_compatible: Option<bool>,
    pub queued_admission_count: usize,
}

/// Structured, redacted-by-default Runtime diagnostics summary. Contains
/// only counts and stable state enums -- no raw Provider/Device/Kernel
/// handles, no raw prompt text, and no filesystem paths.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RuntimeDiagnostics {
    pub model_instance_count: usize,
    pub ready_model_instance_count: usize,
    pub active_session_count: usize,
    pub provider_count: usize,
    pub device_count: usize,
    pub kernel_advertisement_count: usize,
    pub memory_pressure: MemoryPressureLevel,
    pub kv_cache_count: usize,
    pub prefix_cache_entry_count: usize,
    pub model_resolution_status: Option<ModelResolutionStatus>,
    pub model_loading_status: Option<ModelLoadingPhase>,
    pub operator_missing_count: usize,
    pub tokenizer_compatible: Option<bool>,
    pub queued_admission_count: usize,
    pub redacted: bool,
}

pub fn runtime_diagnostics(runtime: &Runtime) -> RuntimeDiagnostics {
    runtime_diagnostics_with(runtime, RuntimeDiagnosticsInputs::default())
}

/// [`runtime_diagnostics`] enriched with caller-supplied
/// [`RuntimeDiagnosticsInputs`] for the state Runtime does not itself own
/// (model resolution/loading status, operator coverage, tokenizer
/// compatibility, queued admission count).
pub fn runtime_diagnostics_with(
    runtime: &Runtime,
    inputs: RuntimeDiagnosticsInputs,
) -> RuntimeDiagnostics {
    RuntimeDiagnostics {
        model_instance_count: runtime.model_instances().instances().count(),
        ready_model_instance_count: runtime
            .model_instances()
            .instances()
            .filter(|instance| instance.status().readiness == ModelInstanceReadiness::Ready)
            .count(),
        active_session_count: runtime.sessions().count(),
        provider_count: runtime.providers().provider_names().count(),
        device_count: runtime.devices().count(),
        kernel_advertisement_count: runtime.kernel_registry().entries().count(),
        memory_pressure: runtime.memory().pressure_snapshot().runtime,
        kv_cache_count: runtime.kv_caches().caches().count(),
        prefix_cache_entry_count: runtime.prefix_caches().entries().count(),
        model_resolution_status: inputs.model_resolution_status,
        model_loading_status: inputs.model_loading_status,
        operator_missing_count: inputs.operator_missing_count,
        tokenizer_compatible: inputs.tokenizer_compatible,
        queued_admission_count: inputs.queued_admission_count,
        redacted: true,
    }
}

// ---------------------------------------------------------------------
// Usage Reporting
// ---------------------------------------------------------------------

/// Structured usage report. Never exposes raw prompt text -- only token
/// counts, timing, and cache/memory summaries.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct UsageReport {
    pub prompt_token_count: usize,
    pub generated_token_count: usize,
    pub total_token_count: usize,
    pub prefill_duration_millis: Option<u64>,
    pub decode_duration_millis: Option<u64>,
    pub tokens_per_second: Option<u64>,
    pub cache_hit: Option<bool>,
    pub memory_estimate_bytes: Option<u64>,
    pub queued_millis: Option<u64>,
    pub cancelled: bool,
}

impl UsageReport {
    pub fn from_generation(
        usage: &GenerationUsage,
        memory: &GenerationMemoryEstimate,
        cache_hit: Option<bool>,
        queued_millis: Option<u64>,
    ) -> Self {
        let memory_estimate_bytes = memory
            .input_token_buffer_bytes
            .saturating_add(memory.output_token_buffer_bytes)
            .saturating_add(memory.logits_buffer_bytes)
            .saturating_add(memory.sampling_workspace_bytes);
        Self {
            prompt_token_count: usage.prompt_tokens,
            generated_token_count: usage.generated_tokens,
            total_token_count: usage.total_tokens,
            prefill_duration_millis: usage.prefill_duration_millis,
            decode_duration_millis: usage.decode_duration_millis,
            tokens_per_second: usage.tokens_per_second,
            cache_hit,
            memory_estimate_bytes: Some(memory_estimate_bytes),
            queued_millis,
            cancelled: matches!(usage.finish_reason, FinishReason::Cancelled),
        }
    }
}

// ---------------------------------------------------------------------
// Browser Target
// ---------------------------------------------------------------------

/// Runtime features that browser targets are not required to support.
pub const UNSUPPORTED_BROWSER_FEATURES: &[&str] = &[
    "wasmtime",
    "native-provider-loading",
    "arbitrary-filesystem-access",
    "process-execution",
    "shell-execution",
    "native-memory-mapping",
];

/// Returns a structured [`InferenceApiError::BrowserFeatureUnsupported`]
/// when running on a wasm32 target and the named feature is one of the
/// features browser targets are not required to support.
pub fn require_browser_supported(feature: &str) -> Result<(), InferenceApiError> {
    if cfg!(target_arch = "wasm32") && UNSUPPORTED_BROWSER_FEATURES.contains(&feature) {
        return Err(InferenceApiError::BrowserFeatureUnsupported {
            feature: feature.into(),
        });
    }
    Ok(())
}

/// Reduced inference paths a browser (wasm32) target may still support even
/// without Wasmtime, native Provider loading, arbitrary filesystem access,
/// process/shell execution, or native memory mapping.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BrowserInferenceCapabilities {
    pub tokenization: bool,
    pub generation: bool,
    pub streaming: bool,
    pub kv_cache: bool,
}

impl BrowserInferenceCapabilities {
    /// The reduced capability set Runtime Inference API targets for
    /// browser: tokenization, generation, and streaming remain available
    /// through a browser-compatible Provider; KV cache reuse does not,
    /// since it depends on native memory residency.
    pub const fn reduced() -> Self {
        Self {
            tokenization: true,
            generation: true,
            streaming: true,
            kv_cache: false,
        }
    }
}

// ---------------------------------------------------------------------
// Observability
// ---------------------------------------------------------------------

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum InferenceApiObservationKind {
    InferenceRequestReceived,
    ModelResolved,
    ModelResolutionFailed,
    ModelLoadingRequested,
    ModelLoaded,
    ModelLoadingFailed,
    ModelInstanceSelected,
    ModelInstanceReady,
    ComponentValidated,
    ComponentInstantiated,
    SessionCreated,
    SessionClosed,
    PromptTokenized,
    TokenizationFailed,
    GenerationAccepted,
    GenerationQueued,
    GenerationStarted,
    PrefillStarted,
    PrefillCompleted,
    DecodeStarted,
    TokenGenerated,
    GenerationCompleted,
    GenerationFailed,
    GenerationCancelled,
    AdapterActivated,
    KvCacheUsed,
    PrefixCacheHit,
    PrefixCacheMiss,
    MemoryAdmissionFailed,
    ProviderUnavailable,
    KernelUnavailable,
    ExecutionGraphValidated,
    GraphValidationCompleted,
    PlanSelected,
    PlanGuardAccepted,
    KernelSelected,
    KernelResolved,
    KernelPrepared,
    KernelDispatched,
    ProviderSubmitted,
    ProviderExecuted,
    ProviderCompleted,
    TensorLogitsProduced,
    LogitsProduced,
    SamplingCompleted,
    StreamOpened,
    StreamClosed,
    StreamInterrupted,
    KvCacheCommitted,
    TokenCommitted,
    /// Correctif 17 / task group 17: a graph node's inputs are all resolved
    /// (from `bindings` or a weight edge) and it is about to be dispatched.
    /// The first per-node event in a node's causal chain.
    GraphNodeReady,
    /// The node's `PlanNodeBinding` was resolved from a published
    /// `PreparedExecutionPlan` via `PreparedExecutionPlanExecutor::prepare_node_execution`.
    PlanBindingResolved,
    /// The node's bound `PreparedKernelId` resolved to a currently-active
    /// `KernelAdvertisement` in the Kernel Registry.
    PreparedKernelResolved,
    /// The node's Kernel output was produced and written into the
    /// registered Provider's storage under a `TensorResourceId`. The last
    /// per-node event in a node's causal chain.
    TensorResourceProduced,
    /// A node's KV-cache-bearing output was written under a *pending*
    /// resource id (not yet Runtime-owned).
    KvUpdatePrepared,
    /// A pending KV update was promoted to the KV cache's committed,
    /// Runtime-owned state.
    KvUpdateCommitted,
}

/// A redacted-by-default observation. `message` MUST NOT contain raw
/// prompt text, raw model weights, raw tensor values, raw KV cache
/// contents, Provider/Device/Kernel handles, memory pointers, filesystem
/// paths, secrets, or external service credentials.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct InferenceApiObservation {
    pub kind: InferenceApiObservationKind,
    pub message: String,
    pub correlation_id: Option<CorrelationId>,
}

impl InferenceApiObservation {
    pub fn new(
        kind: InferenceApiObservationKind,
        message: impl Into<String>,
        correlation_id: Option<CorrelationId>,
    ) -> Self {
        Self {
            kind,
            message: message.into(),
            correlation_id,
        }
    }
}

/// Maximum [`InferenceApiObservation`]s an [`InferenceApiObserver`] retains.
/// A long-running or many-step generation must not grow this buffer without
/// bound; once full, the oldest observation is dropped to admit the newest
/// one, so the buffer always reflects the most recent causal evidence.
pub(crate) const INFERENCE_API_OBSERVATION_BUFFER_CAPACITY: usize = 4096;

/// Collects [`InferenceApiObservation`]s emitted by the `*_observed`
/// variants of the Runtime Inference API functions. Mirrors
/// [`crate::tokenizer::TokenizerObserver`]'s pattern of a caller-owned,
/// explicitly-threaded observer rather than a global sink.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct InferenceApiObserver {
    observations: Vec<InferenceApiObservation>,
}

impl InferenceApiObserver {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn observe(
        &mut self,
        kind: InferenceApiObservationKind,
        message: impl Into<String>,
        correlation_id: Option<CorrelationId>,
    ) {
        if self.observations.len() >= INFERENCE_API_OBSERVATION_BUFFER_CAPACITY {
            self.observations.remove(0);
        }
        self.observations
            .push(InferenceApiObservation::new(kind, message, correlation_id));
    }

    pub fn observations(&self) -> &[InferenceApiObservation] {
        &self.observations
    }
}

// ---------------------------------------------------------------------
// Error Model
// ---------------------------------------------------------------------

/// Structured Runtime Inference API error, unifying the underlying
/// contract-level error types behind a stable, caller-facing category.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum InferenceApiError {
    InferenceApiUnavailable { reason: String },
    ModelReferenceInvalid { reason: String },
    ModelResolutionFailed { reason: String },
    ModelLoadingFailed { reason: String },
    ModelInstanceNotReady { reason: String },
    ModelInstanceUnavailable { reason: String },
    ModelComponentUnavailable { reason: String },
    TokenizerUnavailable { reason: String },
    TokenizerIncompatible { reason: String },
    TokenizationFailed { reason: String },
    SessionCreationFailed { reason: String },
    SessionNotFound,
    SessionClosed,
    GenerationRejected { reason: String },
    GenerationQueued,
    GenerationTimeout,
    GenerationCancelled,
    GenerationFailed { reason: String },
    SamplingFailed { reason: String },
    StopConditionInvalid { reason: String },
    AdapterActivationFailed { reason: String },
    KvCacheUnavailable { reason: String },
    PrefixCacheUnavailable { reason: String },
    MemoryAdmissionFailed { reason: String },
    ProviderUnavailable { reason: String },
    DeviceUnavailable { reason: String },
    KernelUnavailable { reason: String },
    OperatorUnsupported { reason: String },
    GraphPlanningFailed { reason: String },
    PolicyDenied { reason: String },
    CancellationUnsupported { reason: String },
    StreamingUnavailable { reason: String },
    StreamingInterrupted { reason: String },
    DiagnosticsRedacted,
    BrowserFeatureUnsupported { feature: String },
    InternalInferenceApiError { reason: String },
}

impl fmt::Display for InferenceApiError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InferenceApiUnavailable { reason } => {
                write!(f, "inference api unavailable: {reason}")
            }
            Self::ModelReferenceInvalid { reason } => {
                write!(f, "model reference invalid: {reason}")
            }
            Self::ModelResolutionFailed { reason } => {
                write!(f, "model resolution failed: {reason}")
            }
            Self::ModelLoadingFailed { reason } => write!(f, "model loading failed: {reason}"),
            Self::ModelInstanceNotReady { reason } => {
                write!(f, "model instance not ready: {reason}")
            }
            Self::ModelInstanceUnavailable { reason } => {
                write!(f, "model instance unavailable: {reason}")
            }
            Self::ModelComponentUnavailable { reason } => {
                write!(f, "model component unavailable: {reason}")
            }
            Self::TokenizerUnavailable { reason } => {
                write!(f, "tokenizer unavailable: {reason}")
            }
            Self::TokenizerIncompatible { reason } => {
                write!(f, "tokenizer incompatible: {reason}")
            }
            Self::TokenizationFailed { reason } => write!(f, "tokenization failed: {reason}"),
            Self::SessionCreationFailed { reason } => {
                write!(f, "session creation failed: {reason}")
            }
            Self::SessionNotFound => f.write_str("session not found"),
            Self::SessionClosed => f.write_str("session closed"),
            Self::GenerationRejected { reason } => write!(f, "generation rejected: {reason}"),
            Self::GenerationQueued => f.write_str("generation queued"),
            Self::GenerationTimeout => f.write_str("generation timeout"),
            Self::GenerationCancelled => f.write_str("generation cancelled"),
            Self::GenerationFailed { reason } => write!(f, "generation failed: {reason}"),
            Self::SamplingFailed { reason } => write!(f, "sampling failed: {reason}"),
            Self::StopConditionInvalid { reason } => {
                write!(f, "stop condition invalid: {reason}")
            }
            Self::AdapterActivationFailed { reason } => {
                write!(f, "adapter activation failed: {reason}")
            }
            Self::KvCacheUnavailable { reason } => write!(f, "kv cache unavailable: {reason}"),
            Self::PrefixCacheUnavailable { reason } => {
                write!(f, "prefix cache unavailable: {reason}")
            }
            Self::MemoryAdmissionFailed { reason } => {
                write!(f, "memory admission failed: {reason}")
            }
            Self::ProviderUnavailable { reason } => write!(f, "provider unavailable: {reason}"),
            Self::DeviceUnavailable { reason } => write!(f, "device unavailable: {reason}"),
            Self::KernelUnavailable { reason } => write!(f, "kernel unavailable: {reason}"),
            Self::OperatorUnsupported { reason } => write!(f, "operator unsupported: {reason}"),
            Self::GraphPlanningFailed { reason } => write!(f, "graph planning failed: {reason}"),
            Self::PolicyDenied { reason } => write!(f, "policy denied: {reason}"),
            Self::CancellationUnsupported { reason } => {
                write!(f, "cancellation unsupported: {reason}")
            }
            Self::StreamingUnavailable { reason } => {
                write!(f, "streaming unavailable: {reason}")
            }
            Self::StreamingInterrupted { reason } => {
                write!(f, "streaming interrupted: {reason}")
            }
            Self::DiagnosticsRedacted => f.write_str("diagnostics redacted"),
            Self::BrowserFeatureUnsupported { feature } => {
                write!(f, "browser feature unsupported: {feature}")
            }
            Self::InternalInferenceApiError { reason } => {
                write!(f, "internal inference api error: {reason}")
            }
        }
    }
}

impl Error for InferenceApiError {}

impl From<SessionError> for InferenceApiError {
    fn from(error: SessionError) -> Self {
        match error {
            SessionError::SessionNotFound => Self::SessionNotFound,
            SessionError::SessionClosed => Self::SessionClosed,
            SessionError::MemoryAdmissionFailed { reason }
            | SessionError::MemoryBudgetExceeded { reason } => {
                Self::MemoryAdmissionFailed { reason }
            }
            SessionError::GenerationFailed { reason } => Self::GenerationFailed { reason },
            SessionError::CancellationFailed { reason } => Self::CancellationUnsupported { reason },
            SessionError::StreamingFailed { reason } => Self::StreamingUnavailable { reason },
            SessionError::Unauthorized => Self::PolicyDenied {
                reason: "session access unauthorized".into(),
            },
            SessionError::SessionPolicyDenied { reason } => Self::PolicyDenied { reason },
            other => Self::SessionCreationFailed {
                reason: other.to_string(),
            },
        }
    }
}

impl From<GenerationError> for InferenceApiError {
    fn from(error: GenerationError) -> Self {
        match error {
            GenerationError::StopConditionInvalid { message } => {
                Self::StopConditionInvalid { reason: message }
            }
            GenerationError::ParameterInvalid { message, .. }
            | GenerationError::SamplingModeUnsupported { message }
            | GenerationError::LogitsProcessorUnsupported { message } => {
                Self::SamplingFailed { reason: message }
            }
            GenerationError::MemoryAdmissionFailed { message } => {
                Self::MemoryAdmissionFailed { reason: message }
            }
            GenerationError::TokenizerIncompatible { message } => {
                Self::TokenizerIncompatible { reason: message }
            }
            other => Self::GenerationFailed {
                reason: other.to_string(),
            },
        }
    }
}

impl From<TokenizerError> for InferenceApiError {
    fn from(error: TokenizerError) -> Self {
        match error {
            TokenizerError::TokenizerIncompatibleWithModel => Self::TokenizerIncompatible {
                reason: error.to_string(),
            },
            TokenizerError::ImplementationUnavailable
            | TokenizerError::TokenizerArtifactMissing { .. }
            | TokenizerError::InvalidTokenizerArtifact { .. } => Self::TokenizerUnavailable {
                reason: error.to_string(),
            },
            other => Self::TokenizationFailed {
                reason: other.to_string(),
            },
        }
    }
}

impl From<ModelInstanceError> for InferenceApiError {
    fn from(error: ModelInstanceError) -> Self {
        match error {
            ModelInstanceError::ModelInstanceNotReady
            | ModelInstanceError::ModelInstanceLoading
            | ModelInstanceError::ModelInstanceWarming => Self::ModelInstanceNotReady {
                reason: error.to_string(),
            },
            ModelInstanceError::ModelInstanceProviderUnavailable
            | ModelInstanceError::ModelInstanceProviderNotReady
            | ModelInstanceError::ModelInstanceProviderFailed => Self::ProviderUnavailable {
                reason: error.to_string(),
            },
            ModelInstanceError::ModelInstanceDeviceUnavailable
            | ModelInstanceError::ModelInstanceDeviceLost => Self::DeviceUnavailable {
                reason: error.to_string(),
            },
            ModelInstanceError::ModelInstanceMemoryPressure => Self::MemoryAdmissionFailed {
                reason: error.to_string(),
            },
            ModelInstanceError::ModelInstanceBrowserFeatureUnsupported => {
                Self::BrowserFeatureUnsupported {
                    feature: "model-instance".into(),
                }
            }
            other => Self::ModelInstanceUnavailable {
                reason: other.to_string(),
            },
        }
    }
}

impl From<ModelArtifactError> for InferenceApiError {
    fn from(error: ModelArtifactError) -> Self {
        Self::ModelResolutionFailed {
            reason: error.to_string(),
        }
    }
}

impl From<ModelLoadingError> for InferenceApiError {
    fn from(error: ModelLoadingError) -> Self {
        match error.code {
            ModelLoadingErrorCode::ProviderCapabilityUnavailable
            | ModelLoadingErrorCode::ProviderNotReady
            | ModelLoadingErrorCode::ProviderSaturated
            | ModelLoadingErrorCode::ProviderInitializationFailed => Self::ProviderUnavailable {
                reason: error.to_string(),
            },
            ModelLoadingErrorCode::DeviceUnavailable
            | ModelLoadingErrorCode::DeviceMemoryInsufficient => Self::DeviceUnavailable {
                reason: error.to_string(),
            },
            ModelLoadingErrorCode::MemoryFeasibilityFailed
            | ModelLoadingErrorCode::MemoryAllocationFailed => Self::MemoryAdmissionFailed {
                reason: error.to_string(),
            },
            ModelLoadingErrorCode::TokenizerIncompatible => Self::TokenizerIncompatible {
                reason: error.to_string(),
            },
            ModelLoadingErrorCode::BrowserFeatureUnsupported => Self::BrowserFeatureUnsupported {
                feature: "model-loading".into(),
            },
            _ => Self::ModelLoadingFailed {
                reason: error.to_string(),
            },
        }
    }
}

impl From<AdapterError> for InferenceApiError {
    fn from(error: AdapterError) -> Self {
        match error {
            AdapterError::ProviderCapabilityUnavailable
            | AdapterError::ProviderAdapterUnsupported
            | AdapterError::ProviderNotReady
            | AdapterError::ProviderSaturated => Self::ProviderUnavailable {
                reason: error.to_string(),
            },
            AdapterError::DeviceUnavailable | AdapterError::DeviceMemoryInsufficient => {
                Self::DeviceUnavailable {
                    reason: error.to_string(),
                }
            }
            AdapterError::BrowserFeatureUnsupported => Self::BrowserFeatureUnsupported {
                feature: "adapter-activation".into(),
            },
            other => Self::AdapterActivationFailed {
                reason: other.to_string(),
            },
        }
    }
}

impl From<KvCacheError> for InferenceApiError {
    fn from(error: KvCacheError) -> Self {
        Self::KvCacheUnavailable {
            reason: error.to_string(),
        }
    }
}

impl From<PrefixCacheError> for InferenceApiError {
    fn from(error: PrefixCacheError) -> Self {
        Self::PrefixCacheUnavailable {
            reason: error.to_string(),
        }
    }
}

impl From<BatchingError> for InferenceApiError {
    fn from(error: BatchingError) -> Self {
        match error {
            BatchingError::ProviderUnavailable
            | BatchingError::ProviderNotReady
            | BatchingError::ProviderSaturated => Self::ProviderUnavailable {
                reason: error.to_string(),
            },
            BatchingError::DeviceUnavailable | BatchingError::DeviceMemoryInsufficient => {
                Self::DeviceUnavailable {
                    reason: error.to_string(),
                }
            }
            BatchingError::MemoryAdmissionFailed => Self::MemoryAdmissionFailed {
                reason: error.to_string(),
            },
            BatchingError::KvCacheUnavailable | BatchingError::KvCacheIncompatible => {
                Self::KvCacheUnavailable {
                    reason: error.to_string(),
                }
            }
            BatchingError::PrefixCacheReuseDenied => Self::PrefixCacheUnavailable {
                reason: error.to_string(),
            },
            BatchingError::OperationTimedOut => Self::GenerationTimeout,
            BatchingError::OperationCancelled => Self::GenerationCancelled,
            BatchingError::BrowserFeatureUnsupported => Self::BrowserFeatureUnsupported {
                feature: "batching".into(),
            },
            other => Self::GenerationRejected {
                reason: other.to_string(),
            },
        }
    }
}
