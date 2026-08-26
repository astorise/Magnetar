# Post-Baseline Server API Roadmap

Magnetar's first baseline exposes inference only through the in-process
Runtime Inference API and the `magnetar-cli` boundary. This roadmap defines
how Magnetar later exposes inference over a server or local IPC boundary --
`magnetar serve`, health/readiness, model/session/generation/streaming/
cancellation/diagnostics endpoints, an optional OpenAI-compatible facade,
authentication/authorization boundaries, and admission/rate policy --
without weakening the core architecture.

This document, and the `magnetar-runtime::server_api_roadmap` module it
describes, do **not** implement `magnetar serve`, an HTTP server, TLS,
production authentication, or a finalized wire schema. They define the
roadmap **contract** -- endpoint categories, structural boundaries,
structured errors, observability categories, and conformance checks -- that
any future server/API implementation must satisfy.

## Server API Principle

```text
HTTP/local request
    |
    v
Server request validation
    |
    v
Runtime Inference API
    |
    v
Runtime inference contracts
    |
    v
Server response / stream
```

Server API SHALL be a network or local IPC facade around Runtime Inference
API. It SHALL not call Providers, Kernels, Memory Manager, or Model
Components directly.

## Serve Mode Ownership

`magnetar serve` MAY live in `magnetar-cli`, a companion binary, or a
dedicated server crate. Regardless of location, serve mode SHALL call
Runtime Inference API and SHALL not become a separate inference path. This
module contains no serve-mode implementation; it only defines the contract
serve mode must satisfy, mirroring how `cli_boundary.rs` defines the
`magnetar-cli` boundary without implementing the CLI.

## Initial Endpoint Scope

`ServerApiEndpoint` enumerates the ten illustrative endpoints from the
proposal's "Initial Serve Scope" (`SERVER_API_ENDPOINTS`): health, readiness,
models-list, model-inspect, session-create, session-close, generate,
generate-stream, cancel, diagnostics. `ServerApiEndpoint::is_illustrative()`
is always `true` -- endpoint names/paths are illustrative only; final
endpoint syntax is not defined by this change.

## Health And Readiness

`ServerHealthStatus` and `ServerReadinessStatus` are two structurally
independent types: there is no `From` impl or shared constructor between
them, so a caller can never derive readiness from health or vice versa.
`healthy_but_not_ready_is_representable` demonstrates the "Server alive but
model unavailable" scenario is representable and valid: a server can be
`alive` while Runtime is not `ready`.

Health reports process liveness only and SHALL not imply Runtime readiness
or model availability. Readiness SHALL be redacted:
`ServerReadinessStatus::not_ready` and `::ready` always pass their summary
text through `redact_backend_diagnostic`, so no raw Provider handle can ever
reach a readiness response.

## Model Endpoints

`ServerModelEndpointOperation` names the five model endpoint operations
(list known models, inspect model metadata, inspect loaded instance, request
model load, request model unload). Only load/unload operations require an
explicit `ModelEndpointLoadingProof` (`requires_loading_proof()`); read-only
operations never bypass anything because they load nothing.

`ModelEndpointLoadingProof` is deny-by-default (`deny_by_default()`, every
field `false`): a model load/unload request is only accepted once the
caller explicitly attests that Model Source, Cache, Model Artifact, Model
Loading, trust, integrity, compatibility, and policy validation *all* ran
(`validate_model_endpoint_request`). This mirrors how
`provider_roadmap::validate_fused_kernel_declaration` requires an explicit
declaration rather than a boolean rubber stamp -- a caller cannot simply
flip one flag to "true" and skip the underlying validation categories.

`reject_server_arbitrary_model_path` reuses the existing
`model_format_roadmap::validate_local_file_boundary` rather than a parallel
filesystem check, implementing "Server Does Not Load From Arbitrary Paths".

## Session Endpoints

`reject_server_session_owned_authority` rejects a session capability that
names a `magnetar-cli`-owned responsibility (workspace, Git, shell, tool,
network, secret, ...) by delegating to
`inference_api::validate_inference_scope`, exactly like
`cli_boundary::reject_cli_owned_authority`. `ServerSessionRequest::new`
applies this check to every entry in the wrapped
`SessionCreationRequest::allowed_capabilities` before construction succeeds,
implementing "Session Endpoints Preserve Inference Session Scope": a server
session is a Runtime Inference Session, never a workspace/Git/shell/tool/
network/secret-carrying object.

`ServerConnectionId` and `ServerConnectionState` keep server transport
connection identity structurally separate from `InferenceSessionId` (no
`From` impl exists between them). `server_disconnect_policy` makes what
happens on a client disconnect (`ServerDisconnectPolicy`: leave the session
open, cancel active generation, or close the session) an explicit decision
rather than an implicit side effect of the transport closing.

## Generation Endpoint

`ServerGenerationRequest` is the illustrative Generation Endpoint request
surface from the proposal: a `ServerModelOrSessionRef` (model or session),
prompt/chat/tokenized input via the existing `PromptInput` (no duplicated
prompt fields), generation and sampling parameters via the existing
`GenerationParameters` (Runtime does not separate the two either), stop
conditions, a streaming flag, a `KvCachePolicy` cache policy, an optional
`AdapterSetId` adapter policy, a timeout, and a correlation id.

`build_runtime_generation_request` converts a `ServerGenerationRequest`,
plus Runtime-resolved context (`ServerGenerationRuntimeContext`: tokenized
input, tokenizer reference, model reference, request id -- none of which a
client can supply directly), into a real `GenerationRequest` and validates
it through `GenerationRequest::validate`. This implements "Server Generation
Uses Generation Contract": the server never invents its own generation
execution path, it only assembles Runtime's own contract type.

`ServerGeneratedTextHandling` and `reject_tool_execution_from_generated_output`
implement "Generation Endpoint Does Not Execute Tools": generated text that
was executed as a tool call, shell command, or Git operation is always
rejected as a boundary violation.

## Streaming Endpoint

`ServerStreamingTransport` names the four transport placeholders (SSE,
chunked HTTP, WebSocket placeholder, local IPC stream placeholder); the
exact transport remains implementation-defined.

`validate_stream_event_ordering` implements "Streaming Preserves Runtime
Event Ordering" as an order-preserving subsequence check over
`GenerationEventKind`: every forwarded event must appear, in the same
relative order, within the Runtime event sequence. Dropping an event (for
example a redacted one) is permitted; reordering is rejected as
`ServerStreamInterrupted`.

`SERVER_STREAM_FORBIDDEN_PAYLOAD_KINDS` and `reject_raw_stream_payload`
implement "Streaming SHALL not expose raw logits, raw tensor values, raw KV
cache contents, Provider handles, Device handles, Kernel handles, or memory
pointers by default" -- every one of those payload kinds is denied
unconditionally. `ServerStreamEvent` structurally carries only a
`GenerationEventKind` and a `redacted_metadata` string map (values always
passed through `redact_backend_diagnostic`): there is no field through
which a raw payload could reach a forwarded event.

## Cancellation Endpoint

`server_cancellation_calls_runtime_cancellation` implements "Cancellation
Calls Runtime Cancellation" by composing the existing
`inference_api::request_cancellation_at_stage` rather than a parallel
cancellation mechanism. Cancellation applies to inference-owned work;
closing a server transport stream is a separate, server-owned concern.

## Diagnostics Endpoint

`ServerDiagnosticsSummary` carries only counts and stable summaries --
server liveness, Runtime readiness, a redacted Provider readiness summary,
memory pressure, queued admission count, loaded model count, active session
count, cache hit/miss counts, and recent structured error ids. No raw
Provider/Device/Kernel handle, prompt, or filesystem path field exists on
the type. `server_diagnostics_summary` builds one from the existing
`RuntimeDiagnostics` plus server-owned health/readiness/cache state,
redacting the Provider readiness summary text, implementing "Diagnostics Are
Redacted".

## OpenAI-Compatible Facade Placeholder

`OpenAiCompatibilityPolicy` (`RejectUnsupportedField` /
`IgnoreUnsupportedField`) and `handle_openai_unsupported_field` implement
"Unsupported OpenAI fields SHALL fail explicitly or be ignored only
according to documented compatibility policy".
`reject_openai_tool_call_execution` implements "Tool-call fields SHALL not
cause Runtime tool execution".
`openai_facade_maps_to_generation_api_request` proves the facade's only
possible output is the existing `GenerationApiRequest` type -- it cannot
redefine Runtime semantics because it has no other return type to redefine
them with.

## Authentication Boundary

`AuthenticatedServerRequest` is an opaque marker proving a server request
was authenticated, without carrying any credential type: it has no
credential field, so nothing about it could hand Runtime a credential.
`AuthenticatedServerRequest::from_authenticated` only constructs the marker
once server-side authentication already succeeded, otherwise it returns
`ServerAuthenticationRequired`. `reject_credential_in_server_diagnostics`
reuses `model_source_cache_roadmap::reject_credential_in_metadata` rather
than a parallel check, and `redact_server_diagnostic` reuses
`redact_backend_diagnostic`.

## Authorization Boundary

`ServerAuthorizationScope` names the nine authorization scopes from the
proposal (models, source kinds, session creation, generation limits,
streaming permission, diagnostics access, cache inspection, model
loading/unloading, adapter activation). `authorize_server_request`
implements "Authorization Does Not Bypass Runtime Policy": both the
server-side `ServerAuthorizationDecision` AND an independent Runtime policy
gate must pass -- a server user authorized for generation whose Runtime
policy denies memory admission still fails.

## Admission And Rate Policy

`ServerAdmissionLimits` carries the ten limit placeholders from the
proposal (concurrent requests, queued requests, max tokens per request, max
sessions, max loaded models, memory budget, streaming connections, request
body size, prompt size, source/cache operations). `deny_by_default()` sets
every limit to zero capacity, mirroring
`provider_roadmap::ProviderRoadmapFallbackContext::deny_by_default`.
`evaluate_server_admission` compares a `ServerAdmissionState` against the
limits and rejects on the first violated limit. Runtime still owns
inference admission independently; server policy may reject before Runtime
is ever called, but never replaces Runtime's own admission check.

## Source And Cache Boundary

`reject_arbitrary_download_during_generation` reuses the existing
`model_format_roadmap::reject_raw_network_model_reference` rather than a
parallel network check, implementing "Server SHALL not perform arbitrary
model downloads during generation". Cached and authorized-source model
references still flow through the full source/cache and Model Loading
contracts; nothing in this module bypasses them.

## Filesystem Boundary

`reject_arbitrary_filesystem_path` implements "Server generation endpoints
SHALL not read arbitrary server files": a requested path is only permitted
when it is already wrapped in an authorized source contract
(`authorized_source: true`); otherwise it is denied as a boundary
violation, matching the proposal's "request asks server to read
`/etc/passwd`" example.

## Tool/Shell/Git Boundary

`reject_server_tool_shell_git_execution` implements "Core Server API SHALL
not execute tools, shell commands, processes, or Git operations" by
delegating to the existing `inference_api::validate_inference_scope`
(`tool-call`, `shell`, `process`, `process-execution`, and `git` are already
forbidden inference scopes) rather than a parallel forbidden-capability
list. A future agent server, if it exists, is explicitly out of scope for
this core inference Server API.

## Error Model

`ServerApiRoadmapError` covers all 20 categories from the proposal's "Error
Model" section: `server-api-unavailable`, `server-request-invalid`,
`server-request-too-large`, `server-authentication-required`,
`server-authentication-failed`, `server-authorization-denied`,
`server-rate-limited`, `server-admission-rejected`,
`server-stream-unavailable`, `server-stream-interrupted`,
`server-cancellation-failed`, `server-model-not-found`,
`server-model-load-failed`, `server-session-not-found`,
`server-generation-failed`, `server-diagnostics-redacted`,
`server-source-policy-denied`, `server-cache-policy-denied`,
`server-boundary-violation`, and `internal-server-api-error`. Each `id()`
matches its category string exactly.

`ServerModelLoadFailed` and `ServerGenerationFailed` can preserve a wrapped
`InferenceApiError` Runtime cause (`runtime_cause: Option<Box<InferenceApiError>>`).
`ServerApiRoadmapError::model_load_failed_from_runtime` and
`::generation_failed_from_runtime` build these, and `runtime_cause()` lets a
caller inspect the preserved category without matching the whole enum --
implementing "Runtime errors SHALL be preserved or wrapped with structured
cause metadata", mirroring `cli_boundary::CliBoundaryError::runtime_category`.

## Observability

`ServerApiRoadmapObservationKind` covers all 18 categories from the
proposal's "Observability" section (server started/stopped, request
received/rejected/authorized, Runtime request submitted, stream
opened/closed/interrupted, generation completed/failed, cancellation
requested, diagnostics requested, model endpoint used, session endpoint
used, rate limit hit, admission rejected, boundary violation detected).
`ServerApiRoadmapObservation` carries only an observation kind, an optional
endpoint name, and a `redacted_metadata` string map whose values always pass
through `redact_backend_diagnostic` -- there is no field through which a raw
prompt, model weight, tensor value, KV cache content, secret, credential,
file content, cache path, or native handle could reach the observation.

## Conformance Report

`run_server_api_roadmap_conformance` produces a
`ServerApiRoadmapConformanceReport` (mirroring `CliBoundaryConformanceReport`)
asserting: the server accepts ordinary Runtime inference scopes while
rejecting CLI/tool/shell/Git-owned capabilities; a "healthy but not ready"
combination is representable; a model load/unload request without a
complete loading proof is rejected while a complete one is accepted;
read-only model endpoints never require a loading proof; a server session
request carrying a workspace/Git/shell/tool/network/secret capability is
rejected; a server generation request builds and validates through the real
Runtime `GenerationRequest`; generated output executed as a tool call is
rejected while unexecuted output is accepted; in-order (possibly sparse)
forwarded stream events are accepted while reordered events are rejected;
every raw stream payload kind is rejected by default; cancellation composes
Runtime cancellation; authorization requires both the server decision and
Runtime policy; admission is denied by default; arbitrary downloads and
unauthorized filesystem paths are rejected; and a wrapped Runtime error
round-trips through `runtime_cause()`.

## Non-Goals

This roadmap, and the module it describes, do not finalize HTTP endpoint
schemas, implement server mode, define production authentication, define
TLS configuration, define the OpenAI-compatible wire schema, define a model
pull endpoint, define an agent/tool-execution/workspace server, define a
distributed Tachyon API, bypass Runtime Inference API, or expose Runtime
internals.

## Local Commands

Run the Server API roadmap tests:

```powershell
cargo test -p magnetar-runtime server_api_roadmap -- --nocapture
```

Run the full Runtime suite:

```powershell
cargo test --workspace --all-targets
```

Validate the OpenSpec change:

```powershell
openspec validate define-post-baseline-server-api-roadmap --strict
```

## Compatibility Versioning

The current roadmap contract version is `0.1.0`, exposed as
`SERVER_API_ROADMAP_VERSION`. Passing this contract's conformance checks
does not imply any server, HTTP framework, or transport has been
implemented -- it only confirms the roadmap's structural guarantees (server
uses Runtime Inference API, health is not readiness, model/session/
generation endpoints preserve their underlying contracts, streaming
preserves ordering and excludes raw payloads, cancellation composes Runtime
cancellation, diagnostics are redacted, authorization cannot bypass Runtime
policy, admission is deny-by-default, and the filesystem/tool/shell/Git/
source boundaries hold) are satisfied in this Runtime revision.
