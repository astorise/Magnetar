//! Post-baseline Server API roadmap contract (see
//! `openspec/changes/define-post-baseline-server-api-roadmap`).
//!
//! Magnetar's first baseline exposes inference only through the in-process
//! Runtime Inference API (see [`crate::inference_api`]) and the
//! `magnetar-cli` boundary (see [`crate::cli_boundary`]). This module does
//! not implement `magnetar serve`, an HTTP server, TLS, production
//! authentication, or a finalized wire schema -- the change's proposal
//! "Non-Goals" section rules all of that out explicitly. Instead it defines,
//! as executable Rust types and validation functions, the roadmap
//! **contract** any future server/API implementation must satisfy:
//!
//! - [`ServerApiEndpoint`][]: the ten illustrative endpoint categories from
//!   the proposal's "Initial Serve Scope" (`specs/server-api-roadmap/spec.md`
//!   "Post-Baseline Server API Roadmap"). Endpoint names/paths are
//!   illustrative only ([`ServerApiEndpoint::is_illustrative`] is always
//!   `true`); final endpoint syntax is not defined by this change.
//! - [`ServerHealthStatus`] / [`ServerReadinessStatus`] /
//!   [`healthy_but_not_ready_is_representable`][]: two structurally
//!   independent types (no `From` impl or shared constructor exists between
//!   them) implementing "Health Is Not Readiness"
//!   (`specs/server-api-roadmap/spec.md`): health reports process liveness
//!   only and never implies Runtime readiness or model availability.
//! - [`ServerModelEndpointOperation`] / [`ModelEndpointLoadingProof`] /
//!   [`validate_model_endpoint_request`] / [`reject_server_arbitrary_model_path`][]:
//!   implementing "Model Endpoints Preserve Loading Validation" -- a model
//!   load/unload request is only accepted with an explicit proof that Model
//!   Source, Cache, Model Artifact, Model Loading, trust, integrity,
//!   compatibility, and policy validation all ran (mirroring how
//!   [`crate::provider_roadmap::validate_fused_kernel_declaration`] requires
//!   an explicit declaration rather than a boolean rubber stamp), and
//!   arbitrary local paths are rejected via the existing
//!   [`crate::model_format_roadmap::validate_local_file_boundary`].
//! - [`ServerSessionRequest`] / [`reject_server_session_owned_authority`][]:
//!   implementing "Session Endpoints Preserve Inference Session Scope" by
//!   composing [`crate::inference_api::validate_inference_scope`] over
//!   [`SessionCreationRequest::allowed_capabilities`], analogous to
//!   [`crate::cli_boundary::reject_cli_owned_authority`].
//! - [`ServerGenerationRequest`] / [`ServerGenerationRuntimeContext`] /
//!   [`build_runtime_generation_request`][]: the illustrative Generation
//!   Endpoint request surface from the proposal (model/session reference,
//!   prompt/chat/tokenized input via the existing [`PromptInput`],
//!   generation+sampling parameters via the existing [`GenerationParameters`],
//!   stop conditions, streaming flag, cache policy, adapter policy, timeout,
//!   correlation id), converted into a real [`GenerationRequest`] only with
//!   Runtime-resolved context (tokenizer reference, tokenized input, model
//!   reference), implementing "Server Generation Uses Generation Contract".
//! - [`ServerGeneratedTextHandling`] / [`reject_tool_execution_from_generated_output`][]:
//!   implementing "Generation Endpoint Does Not Execute Tools" /
//!   "Server Generation Does Not Execute Side Effects".
//! - [`ServerStreamingTransport`] / [`validate_stream_event_ordering`] /
//!   [`ServerStreamEvent`] / [`reject_raw_stream_payload`][]: implementing
//!   "Streaming Preserves Runtime Event Ordering" (order-preserving
//!   subsequence check over [`GenerationEventKind`]) and the raw-data
//!   exclusion ([`ServerStreamEvent`] structurally carries only a kind and
//!   redacted metadata; [`reject_raw_stream_payload`] denies raw logits,
//!   tensor, KV cache, and handle payload kinds explicitly).
//! - [`server_cancellation_calls_runtime_cancellation`][]: implementing
//!   "Cancellation Calls Runtime Cancellation" by composing
//!   [`crate::inference_api::request_cancellation_at_stage`].
//! - [`ServerDiagnosticsSummary`] / [`server_diagnostics_summary`][]:
//!   implementing "Diagnostics Are Redacted" -- summary/count fields only,
//!   built from the existing [`RuntimeDiagnostics`].
//! - [`OpenAiCompatibilityPolicy`] / [`handle_openai_unsupported_field`] /
//!   [`reject_openai_tool_call_execution`] /
//!   [`openai_facade_maps_to_generation_api_request`][]: implementing
//!   "OpenAI-Compatible Facade Is Optional".
//! - [`AuthenticatedServerRequest`] / [`reject_credential_in_server_diagnostics`]
//!   / [`redact_server_diagnostic`][]: implementing "Authentication Is Server
//!   Boundary" -- an opaque, credential-free authentication marker plus
//!   reuse of [`crate::model_source_cache_roadmap::reject_credential_in_metadata`]
//!   and [`crate::compute::redact_backend_diagnostic`].
//! - [`ServerAuthorizationScope`] / [`ServerAuthorizationDecision`] /
//!   [`authorize_server_request`][]: implementing "Authorization Does Not
//!   Bypass Runtime Policy" -- server authorization and Runtime policy are
//!   both required; either alone is insufficient.
//! - [`ServerAdmissionLimits`] / [`ServerAdmissionState`] /
//!   [`evaluate_server_admission`][]: implementing "Admission And Rate
//!   Policy", deny-by-default like
//!   [`crate::provider_roadmap::ProviderRoadmapFallbackContext::deny_by_default`].
//! - [`reject_arbitrary_download_during_generation`][]: implementing "Source
//!   And Cache Boundary" by reusing
//!   [`crate::model_format_roadmap::reject_raw_network_model_reference`].
//! - [`reject_arbitrary_filesystem_path`][]: implementing "Filesystem
//!   Boundary".
//! - [`reject_server_tool_shell_git_execution`][]: implementing "Tool/Shell/Git
//!   Boundary" by composing [`validate_inference_scope`].
//! - [`ServerApiRoadmapError`][]: the 20 structured error categories from the
//!   proposal's "Error Model" section, preserving a wrapped
//!   [`InferenceApiError`] Runtime cause on the model-load and generation
//!   failure variants.
//! - [`ServerApiRoadmapObservationKind`] / [`ServerApiRoadmapObservation`][]:
//!   the 18 observation categories from the proposal's "Observability"
//!   section, with redacted metadata only.
//! - [`ServerApiRoadmapConformanceReport`] / [`run_server_api_roadmap_conformance`][]:
//!   a conformance report, in the shape of
//!   [`crate::CliBoundaryConformanceReport`], asserting the "Conformance"
//!   section's checks hold.

use crate::adapter::AdapterSetId;
use crate::compute::redact_backend_diagnostic;
use crate::generation::{
    CancellationMetadata, GenerationEventKind, GenerationMemoryEstimate, GenerationModelReference,
    GenerationParameters, GenerationPriority, GenerationRequest, GenerationRequestId,
    GenerationTokenizerReference, StopConditions, StreamingMode,
};
use crate::inference_api::RuntimeDiagnostics;
use crate::inference_api::{
    CancellationOutcome, CancellationStage, CancellationToken, GenerationApiRequest,
    InferenceApiError, ModelRef, PromptInput, request_cancellation_at_stage,
    validate_inference_scope,
};
use crate::kv_cache::KvCachePolicy;
use crate::memory::MemoryPressureLevel;
use crate::model::{ModelArtifactSource, ModelDigest};
use crate::model_format_roadmap::{
    reject_raw_network_model_reference, validate_local_file_boundary,
};
use crate::model_source_cache_roadmap::reject_credential_in_metadata;
use crate::observability::{CorrelationId, TraceId};
use crate::session::{InferenceSessionId, SessionCreationRequest, SessionRedactionPolicy};
use crate::tokenizer::{
    SpecialToken, SpecialTokenKind, TokenId, TokenIdRange, TokenizerArtifactId, TokenizerFamily,
    TokenizerId, TokenizerMetadata, TokenizerRevision,
};
use std::{collections::BTreeMap, error::Error, fmt};

pub const SERVER_API_ROADMAP_VERSION: &str = "0.1.0";

// ---------------------------------------------------------------------
// Initial endpoint scope
// ---------------------------------------------------------------------

/// The ten illustrative endpoints from the proposal's "Initial Serve Scope".
/// Endpoint names are illustrative; final endpoint syntax is not defined by
/// this change.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum ServerApiEndpoint {
    Health,
    Readiness,
    ModelsList,
    ModelInspect,
    SessionCreate,
    SessionClose,
    Generate,
    GenerateStream,
    Cancel,
    Diagnostics,
}

impl ServerApiEndpoint {
    pub const fn id(self) -> &'static str {
        match self {
            Self::Health => "health",
            Self::Readiness => "readiness",
            Self::ModelsList => "models-list",
            Self::ModelInspect => "model-inspect",
            Self::SessionCreate => "session-create",
            Self::SessionClose => "session-close",
            Self::Generate => "generate",
            Self::GenerateStream => "generate-stream",
            Self::Cancel => "cancel",
            Self::Diagnostics => "diagnostics",
        }
    }

    /// Endpoint names/paths in this roadmap are illustrative only: this is
    /// always `true`. "Final endpoint syntax is not defined by this
    /// change."
    pub const fn is_illustrative(self) -> bool {
        true
    }
}

/// All ten roadmap endpoints, in the proposal's declared order.
pub const SERVER_API_ENDPOINTS: &[ServerApiEndpoint] = &[
    ServerApiEndpoint::Health,
    ServerApiEndpoint::Readiness,
    ServerApiEndpoint::ModelsList,
    ServerApiEndpoint::ModelInspect,
    ServerApiEndpoint::SessionCreate,
    ServerApiEndpoint::SessionClose,
    ServerApiEndpoint::Generate,
    ServerApiEndpoint::GenerateStream,
    ServerApiEndpoint::Cancel,
    ServerApiEndpoint::Diagnostics,
];

// ---------------------------------------------------------------------
// Health vs Readiness
// ---------------------------------------------------------------------

/// Server process liveness only. Implements "Health Is Not Readiness":
/// there is no `From<ServerReadinessStatus>` impl and no shared constructor
/// with [`ServerReadinessStatus`], so a caller can never derive one from the
/// other.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ServerHealthStatus {
    pub alive: bool,
    pub redacted_message: Option<String>,
}

impl ServerHealthStatus {
    pub fn alive() -> Self {
        Self {
            alive: true,
            redacted_message: None,
        }
    }

    pub fn not_alive(reason: &str) -> Self {
        Self {
            alive: false,
            redacted_message: Some(redact_backend_diagnostic(reason)),
        }
    }
}

/// Whether Runtime can accept inference requests under current policy.
/// Redacted by default: `model_registry_state_summary` is always passed
/// through [`redact_backend_diagnostic`] and no field exposes raw handles.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ServerReadinessStatus {
    pub runtime_initialized: bool,
    pub cache_accessible: Option<bool>,
    pub providers_registered: Option<bool>,
    pub memory_policy_available: Option<bool>,
    pub admission_policy_available: Option<bool>,
    pub model_registry_state_summary: Option<String>,
    pub ready: bool,
}

impl ServerReadinessStatus {
    pub fn not_ready(reason: &str) -> Self {
        Self {
            runtime_initialized: false,
            cache_accessible: None,
            providers_registered: None,
            memory_policy_available: None,
            admission_policy_available: None,
            model_registry_state_summary: Some(redact_backend_diagnostic(reason)),
            ready: false,
        }
    }

    pub fn ready() -> Self {
        Self {
            runtime_initialized: true,
            cache_accessible: Some(true),
            providers_registered: Some(true),
            memory_policy_available: Some(true),
            admission_policy_available: Some(true),
            model_registry_state_summary: None,
            ready: true,
        }
    }
}

/// Proves that "server alive, Runtime/model not ready" is a representable
/// and valid combination, implementing the scenario "Server alive but model
/// unavailable" from "Health Is Not Readiness".
pub fn healthy_but_not_ready_is_representable(
    health: &ServerHealthStatus,
    readiness: &ServerReadinessStatus,
) -> bool {
    health.alive && !readiness.ready
}

// ---------------------------------------------------------------------
// Model endpoints
// ---------------------------------------------------------------------

/// Model endpoint operations from the proposal's "Model Endpoints" section.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum ServerModelEndpointOperation {
    ListKnownModels,
    InspectModelMetadata,
    InspectLoadedInstance,
    RequestModelLoad,
    RequestModelUnload,
}

impl ServerModelEndpointOperation {
    pub const fn id(self) -> &'static str {
        match self {
            Self::ListKnownModels => "list-known-models",
            Self::InspectModelMetadata => "inspect-model-metadata",
            Self::InspectLoadedInstance => "inspect-loaded-instance",
            Self::RequestModelLoad => "request-model-load",
            Self::RequestModelUnload => "request-model-unload",
        }
    }

    /// Only operations that mutate Model Instance residency (load/unload)
    /// require an explicit [`ModelEndpointLoadingProof`]; read-only
    /// list/inspect operations do not load or unload anything.
    pub const fn requires_loading_proof(self) -> bool {
        matches!(self, Self::RequestModelLoad | Self::RequestModelUnload)
    }
}

/// Explicit proof that Model Source, Cache, Model Artifact, Model Loading,
/// trust, integrity, compatibility, and policy validation all ran for a
/// server-initiated model load/unload request. Deny-by-default: every field
/// defaults to `false`, mirroring
/// [`crate::provider_roadmap::ProviderRoadmapFallbackContext::deny_by_default`].
/// Implements "Model Endpoints Preserve Loading Validation".
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ModelEndpointLoadingProof {
    pub source_validated: bool,
    pub cache_validated: bool,
    pub artifact_validated: bool,
    pub model_loading_validated: bool,
    pub trust_validated: bool,
    pub integrity_validated: bool,
    pub compatibility_validated: bool,
    pub policy_validated: bool,
}

impl ModelEndpointLoadingProof {
    pub const fn deny_by_default() -> Self {
        Self {
            source_validated: false,
            cache_validated: false,
            artifact_validated: false,
            model_loading_validated: false,
            trust_validated: false,
            integrity_validated: false,
            compatibility_validated: false,
            policy_validated: false,
        }
    }

    pub const fn all_validated(&self) -> bool {
        self.source_validated
            && self.cache_validated
            && self.artifact_validated
            && self.model_loading_validated
            && self.trust_validated
            && self.integrity_validated
            && self.compatibility_validated
            && self.policy_validated
    }
}

/// Validates a server model endpoint request, implementing "Model Endpoints
/// Preserve Loading Validation": a load/unload operation is only accepted
/// with a complete [`ModelEndpointLoadingProof`]; read-only operations
/// always pass (they perform no loading).
pub fn validate_model_endpoint_request(
    operation: ServerModelEndpointOperation,
    proof: &ModelEndpointLoadingProof,
) -> Result<(), ServerApiRoadmapError> {
    if !operation.requires_loading_proof() {
        return Ok(());
    }
    if proof.all_validated() {
        Ok(())
    } else {
        Err(ServerApiRoadmapError::ServerModelLoadFailed {
            reason: format!(
                "{}: Model Source/Cache/Artifact/Loading/trust/integrity/compatibility/policy \
                 validation is incomplete",
                operation.id()
            ),
            runtime_cause: None,
        })
    }
}

/// "Server Does Not Load From Arbitrary Paths": reuses the existing
/// [`validate_local_file_boundary`] rather than a parallel filesystem check.
pub fn reject_server_arbitrary_model_path(
    source: &ModelArtifactSource,
    authorized: bool,
) -> Result<(), ServerApiRoadmapError> {
    validate_local_file_boundary(source, authorized).map_err(|error| {
        ServerApiRoadmapError::ServerSourcePolicyDenied {
            reason: error.to_string(),
        }
    })
}

// ---------------------------------------------------------------------
// Session endpoints
// ---------------------------------------------------------------------

/// Rejects a session capability that names a responsibility outside the
/// Runtime Inference API boundary (workspace, Git, shell, tool, network,
/// secret, ...), implementing "Session Endpoints Preserve Inference Session
/// Scope" by delegating to [`validate_inference_scope`], analogous to
/// [`crate::cli_boundary::reject_cli_owned_authority`].
pub fn reject_server_session_owned_authority(
    capability: &str,
) -> Result<(), ServerApiRoadmapError> {
    validate_inference_scope(capability).map_err(|_| {
        ServerApiRoadmapError::ServerBoundaryViolation {
            capability: capability.to_string(),
        }
    })
}

/// A server session request, proving by construction that a server session
/// is a Runtime Inference Session only: [`ServerSessionRequest::new`]
/// rejects any `allowed_capabilities` entry that names a
/// `magnetar-cli`-owned responsibility before the wrapped
/// [`SessionCreationRequest`] can be built.
#[derive(Clone, Debug, PartialEq)]
pub struct ServerSessionRequest {
    pub creation: SessionCreationRequest,
}

impl ServerSessionRequest {
    pub fn new(creation: SessionCreationRequest) -> Result<Self, ServerApiRoadmapError> {
        for capability in &creation.allowed_capabilities {
            reject_server_session_owned_authority(capability)?;
        }
        Ok(Self { creation })
    }
}

/// An opaque server transport connection identity, structurally distinct
/// from [`InferenceSessionId`]. Implements "Server Connection State Is
/// Separate": no `From` impl exists between the two identity types.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ServerConnectionId(String);

impl ServerConnectionId {
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// Server transport connection state, optionally bound to a Runtime
/// [`InferenceSessionId`] but never storing Runtime session internals.
/// Implements "Server Connection State Is Separate": a transport
/// disconnect only ever mutates this type, never Runtime Session state
/// directly -- [`server_disconnect_policy`] is the explicit boundary
/// between the two.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ServerConnectionState {
    pub connection: ServerConnectionId,
    pub bound_session: Option<InferenceSessionId>,
}

/// Policy outcome for a client transport disconnect, implementing the
/// scenario "Client disconnects": Runtime cancellation or session cleanup
/// follows policy explicitly rather than happening implicitly as a side
/// effect of the transport closing.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ServerDisconnectPolicy {
    LeaveSessionOpen,
    CancelActiveGeneration,
    CloseSession,
}

/// Resolves what a transport disconnect does to Runtime state, explicitly
/// and never implicitly: closing a connection never silently mutates
/// Runtime Session state unless `policy` says so.
pub fn server_disconnect_policy(
    connection: &ServerConnectionState,
    policy: ServerDisconnectPolicy,
) -> (
    ServerConnectionId,
    ServerDisconnectPolicy,
    Option<InferenceSessionId>,
) {
    (
        connection.connection.clone(),
        policy,
        connection.bound_session.clone(),
    )
}

// ---------------------------------------------------------------------
// Generation endpoint
// ---------------------------------------------------------------------

/// A generation request targets either a bare model reference or an
/// existing Runtime Inference Session.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ServerModelOrSessionRef {
    Model(ModelRef),
    Session(InferenceSessionId),
}

/// Illustrative Generation Endpoint request surface from the proposal's
/// "Generation Endpoint" section. Prompt text, chat messages, and
/// already-tokenized input are all represented by the existing
/// [`PromptInput`] rather than duplicated fields; generation and sampling
/// parameters are both carried by the existing [`GenerationParameters`]
/// (Runtime does not separate them either).
#[derive(Clone, Debug, PartialEq)]
pub struct ServerGenerationRequest {
    pub model_or_session: ServerModelOrSessionRef,
    pub prompt: PromptInput,
    pub parameters: GenerationParameters,
    pub max_new_tokens: usize,
    pub max_total_tokens: Option<usize>,
    pub stop_conditions: StopConditions,
    pub streaming: bool,
    pub cache_policy: KvCachePolicy,
    pub adapter_policy: Option<AdapterSetId>,
    pub timeout_millis: Option<u64>,
    pub correlation_id: Option<CorrelationId>,
}

/// Runtime-resolved context a server cannot supply from the caller request
/// alone (tokenized input, tokenizer reference, model reference, request
/// id): these come from the Tokenizer Contract and Model Loading/Model
/// Instance lifecycle, not from the client.
#[derive(Clone, Debug, PartialEq)]
pub struct ServerGenerationRuntimeContext {
    pub request_id: GenerationRequestId,
    pub model: GenerationModelReference,
    pub tokenizer: GenerationTokenizerReference,
    pub input_token_ids: Vec<TokenId>,
    pub model_context_length: Option<usize>,
    pub trace_id: Option<TraceId>,
}

/// Builds a real Runtime [`GenerationRequest`] from a server-facing
/// [`ServerGenerationRequest`] plus Runtime-resolved context, implementing
/// "Server Generation Uses Generation Contract": the server never invents
/// its own generation execution path, it only assembles Runtime's own
/// contract type and validates it through [`GenerationRequest::validate`].
pub fn build_runtime_generation_request(
    request: &ServerGenerationRequest,
    context: ServerGenerationRuntimeContext,
) -> Result<GenerationRequest, ServerApiRoadmapError> {
    let prompt_token_count = context.input_token_ids.len();
    let session = match &request.model_or_session {
        ServerModelOrSessionRef::Session(session) => Some(session.clone()),
        ServerModelOrSessionRef::Model(_) => None,
    };
    let streaming = if request.streaming {
        StreamingMode::TokenIds
    } else {
        StreamingMode::Disabled
    };
    let core = GenerationRequest {
        request_id: context.request_id,
        session,
        model: context.model,
        tokenizer: context.tokenizer,
        input_token_ids: context.input_token_ids,
        prompt_token_count,
        max_new_tokens: request.max_new_tokens,
        max_total_tokens: request.max_total_tokens,
        model_context_length: context.model_context_length,
        parameters: request.parameters.clone(),
        stop_conditions: request.stop_conditions.clone(),
        streaming,
        priority: GenerationPriority::default(),
        cancellation: CancellationMetadata::default(),
        memory: GenerationMemoryEstimate::default(),
        correlation_id: request.correlation_id.clone(),
        trace_id: context.trace_id,
    };
    core.validate()
        .map_err(|error| ServerApiRoadmapError::ServerGenerationFailed {
            reason: error.to_string(),
            runtime_cause: None,
        })?;
    Ok(core)
}

/// Generated text, plus whether it was executed as a tool call. Implements
/// "Generation Endpoint Does Not Execute Tools" /
/// "Server Generation Does Not Execute Side Effects": the field exists only
/// so [`reject_tool_execution_from_generated_output`] can assert it is
/// always `false` for core inference server behavior.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ServerGeneratedTextHandling {
    pub text: String,
    pub executed_as_tool_call: bool,
}

/// Rejects generated output that was executed as a tool call, shell
/// command, Git operation, or network call, implementing "Generation
/// Endpoint Does Not Execute Tools", "Tool Shell Git Boundary", and "Server
/// Generation Does Not Execute Side Effects".
pub fn reject_tool_execution_from_generated_output(
    handling: &ServerGeneratedTextHandling,
) -> Result<(), ServerApiRoadmapError> {
    if handling.executed_as_tool_call {
        Err(ServerApiRoadmapError::ServerBoundaryViolation {
            capability: "tool-execution-from-generated-output".into(),
        })
    } else {
        Ok(())
    }
}

// ---------------------------------------------------------------------
// Streaming endpoint
// ---------------------------------------------------------------------

/// Streaming transport options from the proposal's "Streaming Endpoint"
/// section. The exact transport is implementation-defined; these are
/// placeholders only.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum ServerStreamingTransport {
    ServerSentEvents,
    ChunkedHttp,
    WebSocketPlaceholder,
    LocalIpcStreamPlaceholder,
}

impl ServerStreamingTransport {
    pub const fn id(self) -> &'static str {
        match self {
            Self::ServerSentEvents => "server-sent-events",
            Self::ChunkedHttp => "chunked-http",
            Self::WebSocketPlaceholder => "websocket-placeholder",
            Self::LocalIpcStreamPlaceholder => "local-ipc-stream-placeholder",
        }
    }
}

/// Validates that `forwarded` preserves the relative ordering of
/// `source`: every forwarded event must appear, in order, within `source`
/// (dropping an event -- for example a redacted one -- is permitted;
/// reordering is not). Implements "Streaming Preserves Runtime Event
/// Ordering".
pub fn validate_stream_event_ordering(
    source: &[GenerationEventKind],
    forwarded: &[GenerationEventKind],
) -> Result<(), ServerApiRoadmapError> {
    let mut cursor = 0usize;
    for event in forwarded {
        match source[cursor..]
            .iter()
            .position(|candidate| candidate == event)
        {
            Some(offset) => cursor += offset + 1,
            None => {
                return Err(ServerApiRoadmapError::ServerStreamInterrupted {
                    reason: format!("event {event:?} was forwarded out of Runtime event order"),
                });
            }
        }
    }
    Ok(())
}

/// Payload kinds streaming SHALL NOT expose by default, implementing
/// "Streaming SHALL not expose raw logits, raw tensor values, raw KV cache
/// contents, Provider handles, Device handles, Kernel handles, or memory
/// pointers by default."
pub const SERVER_STREAM_FORBIDDEN_PAYLOAD_KINDS: &[&str] = &[
    "raw-logits",
    "raw-tensor-values",
    "raw-kv-cache-contents",
    "provider-handle",
    "device-handle",
    "kernel-handle",
    "memory-pointer",
];

/// Denies a streaming payload kind that would expose raw model internals.
pub fn reject_raw_stream_payload(payload_kind: &str) -> Result<(), ServerApiRoadmapError> {
    let normalized = payload_kind.trim().to_ascii_lowercase();
    if SERVER_STREAM_FORBIDDEN_PAYLOAD_KINDS
        .iter()
        .any(|forbidden| normalized.contains(forbidden))
    {
        Err(ServerApiRoadmapError::ServerStreamUnavailable {
            reason: format!("payload kind '{payload_kind}' is excluded from streaming by default"),
        })
    } else {
        Ok(())
    }
}

/// A single forwarded streaming event. Structurally guaranteed to never
/// carry raw logits, raw tensor values, raw KV cache contents, or native
/// handles by default: the only fields are a [`GenerationEventKind`] and a
/// `redacted_metadata` string map whose values always pass through
/// [`redact_backend_diagnostic`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ServerStreamEvent {
    pub kind: GenerationEventKind,
    pub redacted_metadata: BTreeMap<String, String>,
}

impl ServerStreamEvent {
    pub fn new(kind: GenerationEventKind) -> Self {
        Self {
            kind,
            redacted_metadata: BTreeMap::new(),
        }
    }

    pub fn with_redacted_metadata(
        mut self,
        key: impl Into<String>,
        value: impl AsRef<str>,
    ) -> Self {
        self.redacted_metadata
            .insert(key.into(), redact_backend_diagnostic(value.as_ref()));
        self
    }
}

// ---------------------------------------------------------------------
// Cancellation endpoint
// ---------------------------------------------------------------------

/// Cancellation endpoint calls Runtime cancellation, implementing
/// "Cancellation Calls Runtime Cancellation" by composing
/// [`request_cancellation_at_stage`] rather than a parallel cancellation
/// mechanism.
pub fn server_cancellation_calls_runtime_cancellation(
    token: &CancellationToken,
    stage: CancellationStage,
    supports_interruption_after_dispatch: bool,
) -> CancellationOutcome {
    request_cancellation_at_stage(token, stage, supports_interruption_after_dispatch)
}

// ---------------------------------------------------------------------
// Diagnostics endpoint
// ---------------------------------------------------------------------

/// Redacted-by-default server diagnostics summary from the proposal's
/// "Diagnostics Endpoint" section. Contains only counts and stable
/// summaries -- no raw Provider/Device/Kernel handles, prompts, or
/// filesystem paths.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ServerDiagnosticsSummary {
    pub server_alive: bool,
    pub runtime_ready: bool,
    pub provider_readiness_summary: Option<String>,
    pub memory_pressure: MemoryPressureLevel,
    pub queued_admission_count: usize,
    pub loaded_model_count: usize,
    pub active_session_count: usize,
    pub cache_hit_count: u64,
    pub cache_miss_count: u64,
    pub recent_structured_error_ids: Vec<String>,
    pub redacted: bool,
}

/// Builds a [`ServerDiagnosticsSummary`] from Runtime diagnostics plus
/// server-owned health/readiness/cache state, redacting the Provider
/// readiness summary text. Implements "Diagnostics Are Redacted".
pub fn server_diagnostics_summary(
    runtime_diagnostics: &RuntimeDiagnostics,
    health: &ServerHealthStatus,
    readiness: &ServerReadinessStatus,
    provider_readiness_summary: Option<&str>,
    cache_hit_count: u64,
    cache_miss_count: u64,
    recent_structured_error_ids: Vec<String>,
) -> ServerDiagnosticsSummary {
    ServerDiagnosticsSummary {
        server_alive: health.alive,
        runtime_ready: readiness.ready,
        provider_readiness_summary: provider_readiness_summary.map(redact_backend_diagnostic),
        memory_pressure: runtime_diagnostics.memory_pressure,
        queued_admission_count: runtime_diagnostics.queued_admission_count,
        loaded_model_count: runtime_diagnostics.ready_model_instance_count,
        active_session_count: runtime_diagnostics.active_session_count,
        cache_hit_count,
        cache_miss_count,
        recent_structured_error_ids,
        redacted: true,
    }
}

// ---------------------------------------------------------------------
// OpenAI-compatible facade placeholder
// ---------------------------------------------------------------------

/// Documented compatibility policy for an unsupported OpenAI-compatible
/// field, implementing "Unsupported OpenAI fields SHALL fail explicitly or
/// be ignored only according to documented compatibility policy."
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OpenAiCompatibilityPolicy {
    RejectUnsupportedField,
    IgnoreUnsupportedField,
}

/// Applies the documented compatibility policy to an unsupported field.
pub fn handle_openai_unsupported_field(
    policy: OpenAiCompatibilityPolicy,
    field_name: &str,
) -> Result<(), ServerApiRoadmapError> {
    match policy {
        OpenAiCompatibilityPolicy::RejectUnsupportedField => {
            Err(ServerApiRoadmapError::ServerRequestInvalid {
                reason: format!("unsupported OpenAI-compatible field '{field_name}'"),
            })
        }
        OpenAiCompatibilityPolicy::IgnoreUnsupportedField => Ok(()),
    }
}

/// "Tool-call fields SHALL not cause Runtime tool execution."
pub fn reject_openai_tool_call_execution(
    tool_call_fields_present: bool,
    executed: bool,
) -> Result<(), ServerApiRoadmapError> {
    if tool_call_fields_present && executed {
        Err(ServerApiRoadmapError::ServerBoundaryViolation {
            capability: "openai-facade-tool-call-execution".into(),
        })
    } else {
        Ok(())
    }
}

/// The OpenAI-compatible facade maps to Runtime Inference API and cannot
/// redefine Runtime semantics: this function's only possible output is the
/// existing [`GenerationApiRequest`] type, never a parallel execution path.
pub fn openai_facade_maps_to_generation_api_request(
    core: GenerationRequest,
    privacy: SessionRedactionPolicy,
) -> GenerationApiRequest {
    GenerationApiRequest::new(core, privacy)
}

// ---------------------------------------------------------------------
// Authentication boundary
// ---------------------------------------------------------------------

/// An opaque marker proving a server request was authenticated, without
/// carrying any credential type. Runtime Inference API SHALL not receive
/// ambient network credentials: nothing about this type could hand Runtime
/// a credential, because it has no credential field to hand over.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AuthenticatedServerRequest {
    authenticated: (),
}

impl AuthenticatedServerRequest {
    /// Constructs the marker only when server-side authentication already
    /// succeeded. The resulting value carries no credential authority --
    /// it is proof of a completed server-side check, not a bearer of
    /// secrets.
    pub fn from_authenticated(
        server_authentication_succeeded: bool,
    ) -> Result<Self, ServerApiRoadmapError> {
        if server_authentication_succeeded {
            Ok(Self { authenticated: () })
        } else {
            Err(ServerApiRoadmapError::ServerAuthenticationRequired)
        }
    }
}

/// Rejects a credential-shaped key in server diagnostics metadata, reusing
/// [`reject_credential_in_metadata`] rather than a parallel check.
pub fn reject_credential_in_server_diagnostics(
    metadata: &BTreeMap<String, String>,
) -> Result<(), ServerApiRoadmapError> {
    reject_credential_in_metadata(metadata).map_err(|error| {
        ServerApiRoadmapError::ServerAuthenticationFailed {
            reason: error.to_string(),
        }
    })
}

/// Redacts a server-facing diagnostic message, reusing
/// [`redact_backend_diagnostic`] rather than a parallel redaction path.
pub fn redact_server_diagnostic(message: &str) -> String {
    redact_backend_diagnostic(message)
}

// ---------------------------------------------------------------------
// Authorization boundary
// ---------------------------------------------------------------------

/// Server authorization scopes from the proposal's "Authorization Boundary"
/// section.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum ServerAuthorizationScope {
    Models,
    SourceKinds,
    SessionCreation,
    GenerationLimits,
    StreamingPermission,
    DiagnosticsAccess,
    CacheInspection,
    ModelLoadingUnloading,
    AdapterActivation,
}

impl ServerAuthorizationScope {
    pub const fn id(self) -> &'static str {
        match self {
            Self::Models => "models",
            Self::SourceKinds => "source-kinds",
            Self::SessionCreation => "session-creation",
            Self::GenerationLimits => "generation-limits",
            Self::StreamingPermission => "streaming-permission",
            Self::DiagnosticsAccess => "diagnostics-access",
            Self::CacheInspection => "cache-inspection",
            Self::ModelLoadingUnloading => "model-loading-unloading",
            Self::AdapterActivation => "adapter-activation",
        }
    }
}

/// A server-side authorization decision for a scope. Deliberately does not
/// by itself grant the request: [`authorize_server_request`] additionally
/// requires the Runtime policy gate to pass.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ServerAuthorizationDecision {
    pub scope: ServerAuthorizationScope,
    pub server_authorized: bool,
}

/// Authorizes a server request, implementing "Authorization Does Not Bypass
/// Runtime Policy": both the server-side decision AND the Runtime policy
/// gate must pass. A server user authorized for generation whose Runtime
/// policy denies memory admission still fails.
pub fn authorize_server_request(
    decision: &ServerAuthorizationDecision,
    runtime_policy_allows: bool,
) -> Result<(), ServerApiRoadmapError> {
    if !decision.server_authorized {
        return Err(ServerApiRoadmapError::ServerAuthorizationDenied {
            scope: decision.scope.id().to_string(),
        });
    }
    if !runtime_policy_allows {
        return Err(ServerApiRoadmapError::ServerAuthorizationDenied {
            scope: format!("{}: runtime policy denied", decision.scope.id()),
        });
    }
    Ok(())
}

// ---------------------------------------------------------------------
// Admission and rate policy
// ---------------------------------------------------------------------

/// Admission/rate policy limit placeholders from the proposal's "Admission
/// And Rate Policy" and "Request Size And Prompt Limits" sections.
/// Deny-by-default: [`ServerAdmissionLimits::deny_by_default`] sets every
/// limit to zero capacity, mirroring
/// [`crate::provider_roadmap::ProviderRoadmapFallbackContext::deny_by_default`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ServerAdmissionLimits {
    pub max_concurrent_requests: u32,
    pub max_queued_requests: u32,
    pub max_tokens_per_request: u32,
    pub max_sessions: u32,
    pub max_loaded_models: u32,
    pub memory_budget_bytes: u64,
    pub max_streaming_connections: u32,
    pub max_request_body_bytes: u64,
    pub max_prompt_bytes: u64,
    pub max_source_cache_operations: u32,
}

impl ServerAdmissionLimits {
    pub const fn deny_by_default() -> Self {
        Self {
            max_concurrent_requests: 0,
            max_queued_requests: 0,
            max_tokens_per_request: 0,
            max_sessions: 0,
            max_loaded_models: 0,
            memory_budget_bytes: 0,
            max_streaming_connections: 0,
            max_request_body_bytes: 0,
            max_prompt_bytes: 0,
            max_source_cache_operations: 0,
        }
    }
}

/// Current server admission state, compared against [`ServerAdmissionLimits`]
/// by [`evaluate_server_admission`].
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ServerAdmissionState {
    pub concurrent_requests: u32,
    pub queued_requests: u32,
    pub requested_tokens: u32,
    pub active_sessions: u32,
    pub loaded_models: u32,
    pub memory_used_bytes: u64,
    pub streaming_connections: u32,
    pub request_body_bytes: u64,
    pub prompt_bytes: u64,
    pub source_cache_operations_in_flight: u32,
}

/// Evaluates server-side admission before a Runtime call, implementing
/// "Admission And Rate Policy" and "Request Size And Prompt Limits": server
/// policy may reject before Runtime is ever called, but Runtime still owns
/// inference admission independently (see [`crate::memory::MemoryManager`]
/// admission).
pub fn evaluate_server_admission(
    limits: &ServerAdmissionLimits,
    state: &ServerAdmissionState,
) -> Result<(), ServerApiRoadmapError> {
    let checks: &[(bool, &str)] = &[
        (
            state.concurrent_requests >= limits.max_concurrent_requests,
            "concurrent request limit reached",
        ),
        (
            state.queued_requests >= limits.max_queued_requests,
            "queued request limit reached",
        ),
        (
            state.requested_tokens > limits.max_tokens_per_request,
            "max token limit exceeded",
        ),
        (
            state.active_sessions >= limits.max_sessions,
            "max session limit reached",
        ),
        (
            state.loaded_models >= limits.max_loaded_models,
            "max loaded model limit reached",
        ),
        (
            state.memory_used_bytes > limits.memory_budget_bytes,
            "memory budget exceeded",
        ),
        (
            state.streaming_connections >= limits.max_streaming_connections,
            "streaming connection limit reached",
        ),
        (
            state.request_body_bytes > limits.max_request_body_bytes,
            "request body size limit exceeded",
        ),
        (
            state.prompt_bytes > limits.max_prompt_bytes,
            "prompt size limit exceeded",
        ),
        (
            state.source_cache_operations_in_flight >= limits.max_source_cache_operations,
            "source/cache operation limit reached",
        ),
    ];
    for (violated, reason) in checks {
        if *violated {
            return Err(ServerApiRoadmapError::ServerAdmissionRejected {
                reason: (*reason).to_string(),
            });
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------
// Source/cache boundary
// ---------------------------------------------------------------------

/// "Server SHALL not perform arbitrary model downloads during generation",
/// implementing "Source And Cache Boundary" by reusing the existing
/// [`reject_raw_network_model_reference`] rather than a parallel check.
pub fn reject_arbitrary_download_during_generation(
    model_reference: &str,
) -> Result<(), ServerApiRoadmapError> {
    reject_raw_network_model_reference(model_reference).map_err(|error| {
        ServerApiRoadmapError::ServerSourcePolicyDenied {
            reason: error.to_string(),
        }
    })
}

// ---------------------------------------------------------------------
// Filesystem boundary
// ---------------------------------------------------------------------

/// "Server generation endpoints SHALL not read arbitrary server files",
/// implementing "Filesystem Boundary": a requested path is only permitted
/// when it is already wrapped in an authorized source contract.
pub fn reject_arbitrary_filesystem_path(
    requested_path: &str,
    authorized_source: bool,
) -> Result<(), ServerApiRoadmapError> {
    if authorized_source {
        Ok(())
    } else {
        Err(ServerApiRoadmapError::ServerBoundaryViolation {
            capability: format!("filesystem-path:{requested_path}"),
        })
    }
}

// ---------------------------------------------------------------------
// Tool/Shell/Git boundary
// ---------------------------------------------------------------------

/// "Core Server API SHALL not execute tools, shell commands, processes, or
/// Git operations", implementing "Tool Shell Git Boundary" by delegating to
/// [`validate_inference_scope`] (`tool-call`, `shell`, `process`,
/// `process-execution`, and `git` are already forbidden inference scopes).
pub fn reject_server_tool_shell_git_execution(
    capability: &str,
) -> Result<(), ServerApiRoadmapError> {
    validate_inference_scope(capability).map_err(|_| {
        ServerApiRoadmapError::ServerBoundaryViolation {
            capability: capability.to_string(),
        }
    })
}

// ---------------------------------------------------------------------
// Error Model
// ---------------------------------------------------------------------

/// Structured Server API error, covering all 20 categories from the
/// proposal's "Error Model" section. `ServerModelLoadFailed` and
/// `ServerGenerationFailed` can preserve a wrapped [`InferenceApiError`]
/// Runtime cause, implementing "Runtime errors SHALL be preserved or
/// wrapped with structured cause metadata."
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ServerApiRoadmapError {
    ServerApiUnavailable {
        reason: String,
    },
    ServerRequestInvalid {
        reason: String,
    },
    ServerRequestTooLarge {
        reason: String,
    },
    ServerAuthenticationRequired,
    ServerAuthenticationFailed {
        reason: String,
    },
    ServerAuthorizationDenied {
        scope: String,
    },
    ServerRateLimited {
        reason: String,
    },
    ServerAdmissionRejected {
        reason: String,
    },
    ServerStreamUnavailable {
        reason: String,
    },
    ServerStreamInterrupted {
        reason: String,
    },
    ServerCancellationFailed {
        reason: String,
    },
    ServerModelNotFound {
        model: String,
    },
    ServerModelLoadFailed {
        reason: String,
        runtime_cause: Option<Box<InferenceApiError>>,
    },
    ServerSessionNotFound {
        session: String,
    },
    ServerGenerationFailed {
        reason: String,
        runtime_cause: Option<Box<InferenceApiError>>,
    },
    ServerDiagnosticsRedacted,
    ServerSourcePolicyDenied {
        reason: String,
    },
    ServerCachePolicyDenied {
        reason: String,
    },
    ServerBoundaryViolation {
        capability: String,
    },
    InternalServerApiError {
        reason: String,
    },
}

impl ServerApiRoadmapError {
    pub const fn id(&self) -> &'static str {
        match self {
            Self::ServerApiUnavailable { .. } => "server-api-unavailable",
            Self::ServerRequestInvalid { .. } => "server-request-invalid",
            Self::ServerRequestTooLarge { .. } => "server-request-too-large",
            Self::ServerAuthenticationRequired => "server-authentication-required",
            Self::ServerAuthenticationFailed { .. } => "server-authentication-failed",
            Self::ServerAuthorizationDenied { .. } => "server-authorization-denied",
            Self::ServerRateLimited { .. } => "server-rate-limited",
            Self::ServerAdmissionRejected { .. } => "server-admission-rejected",
            Self::ServerStreamUnavailable { .. } => "server-stream-unavailable",
            Self::ServerStreamInterrupted { .. } => "server-stream-interrupted",
            Self::ServerCancellationFailed { .. } => "server-cancellation-failed",
            Self::ServerModelNotFound { .. } => "server-model-not-found",
            Self::ServerModelLoadFailed { .. } => "server-model-load-failed",
            Self::ServerSessionNotFound { .. } => "server-session-not-found",
            Self::ServerGenerationFailed { .. } => "server-generation-failed",
            Self::ServerDiagnosticsRedacted => "server-diagnostics-redacted",
            Self::ServerSourcePolicyDenied { .. } => "server-source-policy-denied",
            Self::ServerCachePolicyDenied { .. } => "server-cache-policy-denied",
            Self::ServerBoundaryViolation { .. } => "server-boundary-violation",
            Self::InternalServerApiError { .. } => "internal-server-api-error",
        }
    }

    /// Builds [`Self::ServerModelLoadFailed`] preserving `error` as the
    /// structured Runtime cause.
    pub fn model_load_failed_from_runtime(error: InferenceApiError) -> Self {
        Self::ServerModelLoadFailed {
            reason: error.to_string(),
            runtime_cause: Some(Box::new(error)),
        }
    }

    /// Builds [`Self::ServerGenerationFailed`] preserving `error` as the
    /// structured Runtime cause.
    pub fn generation_failed_from_runtime(error: InferenceApiError) -> Self {
        Self::ServerGenerationFailed {
            reason: error.to_string(),
            runtime_cause: Some(Box::new(error)),
        }
    }

    /// Returns the preserved Runtime structured error category, or `None`
    /// for every variant that does not wrap one. Lets callers inspect the
    /// Runtime category without matching the whole enum, mirroring
    /// [`crate::cli_boundary::CliBoundaryError::runtime_category`].
    pub fn runtime_cause(&self) -> Option<&InferenceApiError> {
        match self {
            Self::ServerModelLoadFailed { runtime_cause, .. }
            | Self::ServerGenerationFailed { runtime_cause, .. } => runtime_cause.as_deref(),
            _ => None,
        }
    }
}

impl fmt::Display for ServerApiRoadmapError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ServerApiUnavailable { reason }
            | Self::ServerRequestInvalid { reason }
            | Self::ServerRequestTooLarge { reason }
            | Self::ServerAuthenticationFailed { reason }
            | Self::ServerRateLimited { reason }
            | Self::ServerAdmissionRejected { reason }
            | Self::ServerStreamUnavailable { reason }
            | Self::ServerStreamInterrupted { reason }
            | Self::ServerCancellationFailed { reason }
            | Self::ServerSourcePolicyDenied { reason }
            | Self::ServerCachePolicyDenied { reason }
            | Self::InternalServerApiError { reason } => write!(f, "{}: {reason}", self.id()),
            Self::ServerAuthenticationRequired | Self::ServerDiagnosticsRedacted => {
                f.write_str(self.id())
            }
            Self::ServerAuthorizationDenied { scope } => write!(f, "{}: {scope}", self.id()),
            Self::ServerModelNotFound { model } => write!(f, "{}: {model}", self.id()),
            Self::ServerModelLoadFailed {
                reason,
                runtime_cause,
            }
            | Self::ServerGenerationFailed {
                reason,
                runtime_cause,
            } => {
                write!(f, "{}: {reason}", self.id())?;
                if let Some(cause) = runtime_cause {
                    write!(f, " (runtime cause: {cause})")?;
                }
                Ok(())
            }
            Self::ServerSessionNotFound { session } => write!(f, "{}: {session}", self.id()),
            Self::ServerBoundaryViolation { capability } => {
                write!(f, "{}: {capability}", self.id())
            }
        }
    }
}

impl Error for ServerApiRoadmapError {}

// ---------------------------------------------------------------------
// Observability
// ---------------------------------------------------------------------

/// The 18 observation categories from the proposal's "Observability"
/// section.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum ServerApiRoadmapObservationKind {
    ServerStarted,
    ServerStopped,
    RequestReceived,
    RequestRejected,
    RequestAuthorized,
    RuntimeRequestSubmitted,
    StreamOpened,
    StreamClosed,
    StreamInterrupted,
    GenerationCompleted,
    GenerationFailed,
    CancellationRequested,
    DiagnosticsRequested,
    ModelEndpointUsed,
    SessionEndpointUsed,
    RateLimitHit,
    AdmissionRejected,
    BoundaryViolationDetected,
}

/// A single Server API observation. Structurally guaranteed to never carry
/// a raw prompt, raw model weight, raw tensor value, raw KV cache content,
/// secret, credential, raw file content, raw cache path, or native handle
/// by default: the only fields are an enum `kind`, an optional endpoint
/// name, and a `redacted_metadata` string map whose values always pass
/// through [`redact_backend_diagnostic`]. Implements "Server Observability".
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ServerApiRoadmapObservation {
    pub kind: ServerApiRoadmapObservationKind,
    pub endpoint: Option<String>,
    pub redacted_metadata: BTreeMap<String, String>,
}

impl ServerApiRoadmapObservation {
    pub fn new(kind: ServerApiRoadmapObservationKind) -> Self {
        Self {
            kind,
            endpoint: None,
            redacted_metadata: BTreeMap::new(),
        }
    }

    pub fn with_endpoint(mut self, endpoint: impl Into<String>) -> Self {
        self.endpoint = Some(endpoint.into());
        self
    }

    pub fn with_redacted_metadata(
        mut self,
        key: impl Into<String>,
        value: impl AsRef<str>,
    ) -> Self {
        self.redacted_metadata
            .insert(key.into(), redact_backend_diagnostic(value.as_ref()));
        self
    }
}

// ---------------------------------------------------------------------
// Conformance
// ---------------------------------------------------------------------

/// A single Server API roadmap conformance check result, mirroring
/// [`crate::CliBoundaryConformanceResult`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ServerApiRoadmapConformanceResult {
    pub requirement: String,
    pub passed: bool,
    pub diagnostic: Option<String>,
}

/// A collected set of [`ServerApiRoadmapConformanceResult`]s.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ServerApiRoadmapConformanceReport {
    pub results: Vec<ServerApiRoadmapConformanceResult>,
}

impl ServerApiRoadmapConformanceReport {
    pub fn is_conformant(&self) -> bool {
        self.results.iter().all(|result| result.passed)
    }
}

fn record(
    results: &mut Vec<ServerApiRoadmapConformanceResult>,
    requirement: impl Into<String>,
    passed: bool,
    diagnostic: impl Into<String>,
) {
    let diagnostic = diagnostic.into();
    results.push(ServerApiRoadmapConformanceResult {
        requirement: requirement.into(),
        passed,
        diagnostic: (!passed).then_some(diagnostic),
    });
}

/// A minimal, self-contained [`GenerationTokenizerReference`] fixture used
/// only by [`run_server_api_roadmap_conformance`] to exercise
/// [`build_runtime_generation_request`] end-to-end.
fn conformance_tokenizer_reference() -> GenerationTokenizerReference {
    let metadata = TokenizerMetadata {
        id: TokenizerId::new("server-api-roadmap-fixture").expect("valid tokenizer id"),
        artifact: TokenizerArtifactId::new("server-api-roadmap-fixture-artifact")
            .expect("valid tokenizer artifact id"),
        digest: ModelDigest::sha256(b"server-api-roadmap-fixture"),
        family: TokenizerFamily::new("fixture").expect("valid tokenizer family"),
        revision: TokenizerRevision::new("1.0.0").expect("valid tokenizer revision"),
        vocabulary_size: 256,
        added_token_count: 2,
        token_id_range: TokenIdRange::new(1, 300),
        model_max_length: Some(64),
        special_tokens: vec![SpecialToken::new(SpecialTokenKind::Eos, "<eos>", 299)],
        additional_special_tokens: Vec::new(),
        byte_fallback: false,
        normalization: None,
        pre_tokenizer: None,
        supports_offsets: true,
        supports_token_type_ids: false,
        supports_browser: true,
    };
    GenerationTokenizerReference {
        tokenizer_id: metadata.id.clone(),
        metadata,
    }
}

/// Runs the Server API roadmap conformance checks described in this
/// module's doc comment: server accepts ordinary inference scopes but
/// rejects CLI/tool/shell/Git-owned capabilities; health/readiness stay
/// structurally independent and a "healthy but not ready" combination is
/// representable; a model load/unload request without a complete loading
/// proof is rejected; a server session request carrying a workspace/Git/
/// shell/tool/network/secret capability is rejected; a server generation
/// request builds and validates through the real Runtime
/// [`GenerationRequest`]; generated output executed as a tool call is
/// rejected; stream event reordering is rejected while in-order forwarding
/// (including drops) is accepted; raw stream payload kinds are rejected;
/// cancellation composes Runtime cancellation; authorization requires both
/// the server decision and Runtime policy; admission is denied by default;
/// arbitrary downloads/filesystem paths are rejected; and a wrapped Runtime
/// error round-trips through `runtime_cause()`.
pub fn run_server_api_roadmap_conformance() -> ServerApiRoadmapConformanceReport {
    let mut results = Vec::new();

    {
        let outcome = validate_inference_scope("generation");
        record(
            &mut results,
            "server accepts an ordinary Runtime inference scope",
            outcome.is_ok(),
            format!("unexpected outcome: {outcome:?}"),
        );
    }

    for capability in [
        "workspace",
        "git",
        "shell",
        "tool-call",
        "secrets",
        "process",
    ] {
        let session_outcome = reject_server_session_owned_authority(capability);
        let tool_outcome = reject_server_tool_shell_git_execution(capability);
        record(
            &mut results,
            format!("server session endpoint rejects capability '{capability}'"),
            matches!(
                session_outcome,
                Err(ServerApiRoadmapError::ServerBoundaryViolation { .. })
            ),
            format!("unexpected outcome: {session_outcome:?}"),
        );
        record(
            &mut results,
            format!("server tool/shell/Git boundary rejects capability '{capability}'"),
            matches!(
                tool_outcome,
                Err(ServerApiRoadmapError::ServerBoundaryViolation { .. })
            ),
            format!("unexpected outcome: {tool_outcome:?}"),
        );
    }

    {
        let health = ServerHealthStatus::alive();
        let readiness = ServerReadinessStatus::not_ready("no model loaded");
        record(
            &mut results,
            "server can be healthy while Runtime/model is not ready",
            healthy_but_not_ready_is_representable(&health, &readiness),
            "healthy-but-not-ready combination was not representable",
        );
    }

    {
        let incomplete = validate_model_endpoint_request(
            ServerModelEndpointOperation::RequestModelLoad,
            &ModelEndpointLoadingProof::deny_by_default(),
        );
        record(
            &mut results,
            "model load endpoint denies an incomplete loading proof",
            matches!(
                incomplete,
                Err(ServerApiRoadmapError::ServerModelLoadFailed { .. })
            ),
            format!("unexpected outcome: {incomplete:?}"),
        );

        let complete = validate_model_endpoint_request(
            ServerModelEndpointOperation::RequestModelLoad,
            &ModelEndpointLoadingProof {
                source_validated: true,
                cache_validated: true,
                artifact_validated: true,
                model_loading_validated: true,
                trust_validated: true,
                integrity_validated: true,
                compatibility_validated: true,
                policy_validated: true,
            },
        );
        record(
            &mut results,
            "model load endpoint accepts a complete loading proof",
            complete.is_ok(),
            format!("unexpected outcome: {complete:?}"),
        );

        let read_only = validate_model_endpoint_request(
            ServerModelEndpointOperation::ListKnownModels,
            &ModelEndpointLoadingProof::deny_by_default(),
        );
        record(
            &mut results,
            "read-only model endpoints do not require a loading proof",
            read_only.is_ok(),
            format!("unexpected outcome: {read_only:?}"),
        );
    }

    {
        let request = ServerGenerationRequest {
            model_or_session: ServerModelOrSessionRef::Model(
                ModelRef::new("server-roadmap-fixture-model").expect("valid model ref"),
            ),
            prompt: PromptInput::PlainText("hello".into()),
            parameters: GenerationParameters::greedy(),
            max_new_tokens: 4,
            max_total_tokens: Some(32),
            stop_conditions: StopConditions::default(),
            streaming: false,
            cache_policy: KvCachePolicy {
                enabled: false,
                max_cache_tokens: None,
                max_cache_memory_bytes: None,
                sharing: crate::kv_cache::KvCacheSharingPolicy::Deny,
                retention: crate::kv_cache::KvCacheRetentionPolicy::ReleaseOnSessionClose,
                prefix_reuse_allowed: false,
                privacy_redaction_required: true,
            },
            adapter_policy: None,
            timeout_millis: Some(5_000),
            correlation_id: Some(CorrelationId::new("server-roadmap-fixture")),
        };
        let context = ServerGenerationRuntimeContext {
            request_id: GenerationRequestId::new("server-roadmap-fixture-request")
                .expect("valid generation request id"),
            model: GenerationModelReference::LoadedModelContext("fixture-model-context".into()),
            tokenizer: conformance_tokenizer_reference(),
            input_token_ids: vec![2, 3, 4],
            model_context_length: Some(64),
            trace_id: None,
        };
        let built = build_runtime_generation_request(&request, context);
        record(
            &mut results,
            "server generation request builds a valid Runtime GenerationRequest",
            built.is_ok(),
            format!("unexpected outcome: {built:?}"),
        );
    }

    {
        let executed = reject_tool_execution_from_generated_output(&ServerGeneratedTextHandling {
            text: "```bash\nrm -rf /\n```".into(),
            executed_as_tool_call: true,
        });
        record(
            &mut results,
            "generated tool-call-like output is not executed",
            matches!(
                executed,
                Err(ServerApiRoadmapError::ServerBoundaryViolation { .. })
            ),
            format!("unexpected outcome: {executed:?}"),
        );
        let not_executed =
            reject_tool_execution_from_generated_output(&ServerGeneratedTextHandling {
                text: "```bash\nrm -rf /\n```".into(),
                executed_as_tool_call: false,
            });
        record(
            &mut results,
            "generated tool-call-like output that was never executed is accepted",
            not_executed.is_ok(),
            format!("unexpected outcome: {not_executed:?}"),
        );
    }

    {
        let source = [
            GenerationEventKind::GenerationStarted,
            GenerationEventKind::PrefillStarted,
            GenerationEventKind::PrefillCompleted,
            GenerationEventKind::DecodeStarted,
            GenerationEventKind::TokenGenerated,
            GenerationEventKind::GenerationCompleted,
        ];
        let in_order = [
            GenerationEventKind::PrefillCompleted,
            GenerationEventKind::TokenGenerated,
        ];
        record(
            &mut results,
            "in-order (possibly sparse) forwarded events preserve Runtime ordering",
            validate_stream_event_ordering(&source, &in_order).is_ok(),
            "in-order forwarding was unexpectedly rejected",
        );
        let reordered = [
            GenerationEventKind::TokenGenerated,
            GenerationEventKind::PrefillCompleted,
        ];
        let reorder_outcome = validate_stream_event_ordering(&source, &reordered);
        record(
            &mut results,
            "reordered forwarded events are rejected",
            matches!(
                reorder_outcome,
                Err(ServerApiRoadmapError::ServerStreamInterrupted { .. })
            ),
            format!("unexpected outcome: {reorder_outcome:?}"),
        );
    }

    for payload_kind in SERVER_STREAM_FORBIDDEN_PAYLOAD_KINDS {
        let outcome = reject_raw_stream_payload(payload_kind);
        record(
            &mut results,
            format!("streaming rejects raw payload kind '{payload_kind}' by default"),
            matches!(
                outcome,
                Err(ServerApiRoadmapError::ServerStreamUnavailable { .. })
            ),
            format!("unexpected outcome: {outcome:?}"),
        );
    }

    {
        let token = CancellationToken::new(
            GenerationRequestId::new("server-roadmap-cancel-fixture")
                .expect("valid generation request id"),
        );
        let outcome = server_cancellation_calls_runtime_cancellation(
            &token,
            CancellationStage::Decode,
            false,
        );
        record(
            &mut results,
            "cancellation endpoint calls Runtime cancellation for an always-cancellable stage",
            outcome == CancellationOutcome::Cancelled,
            format!("unexpected outcome: {outcome:?}"),
        );
    }

    {
        let decision = ServerAuthorizationDecision {
            scope: ServerAuthorizationScope::GenerationLimits,
            server_authorized: true,
        };
        let denied_by_runtime = authorize_server_request(&decision, false);
        record(
            &mut results,
            "server authorization alone does not bypass a Runtime policy denial",
            matches!(
                denied_by_runtime,
                Err(ServerApiRoadmapError::ServerAuthorizationDenied { .. })
            ),
            format!("unexpected outcome: {denied_by_runtime:?}"),
        );
        let allowed = authorize_server_request(&decision, true);
        record(
            &mut results,
            "authorization succeeds only when both server and Runtime policy allow it",
            allowed.is_ok(),
            format!("unexpected outcome: {allowed:?}"),
        );
    }

    {
        let outcome = evaluate_server_admission(
            &ServerAdmissionLimits::deny_by_default(),
            &ServerAdmissionState::default(),
        );
        record(
            &mut results,
            "admission is denied by default",
            matches!(
                outcome,
                Err(ServerApiRoadmapError::ServerAdmissionRejected { .. })
            ),
            format!("unexpected outcome: {outcome:?}"),
        );
    }

    {
        let outcome = reject_arbitrary_download_during_generation("https://example.com/model.gguf");
        record(
            &mut results,
            "server rejects an arbitrary remote model URL during generation",
            matches!(
                outcome,
                Err(ServerApiRoadmapError::ServerSourcePolicyDenied { .. })
            ),
            format!("unexpected outcome: {outcome:?}"),
        );
    }

    {
        let outcome = reject_arbitrary_filesystem_path("/etc/passwd", false);
        record(
            &mut results,
            "server rejects an unauthorized arbitrary filesystem path",
            matches!(
                outcome,
                Err(ServerApiRoadmapError::ServerBoundaryViolation { .. })
            ),
            format!("unexpected outcome: {outcome:?}"),
        );
    }

    {
        let source = InferenceApiError::ModelLoadingFailed {
            reason: "example".into(),
        };
        let wrapped = ServerApiRoadmapError::model_load_failed_from_runtime(source.clone());
        let passed = wrapped.runtime_cause() == Some(&source);
        record(
            &mut results,
            "ServerApiRoadmapError preserves the wrapped Runtime error category",
            passed,
            "runtime_cause() did not round-trip",
        );
    }

    ServerApiRoadmapConformanceReport { results }
}
