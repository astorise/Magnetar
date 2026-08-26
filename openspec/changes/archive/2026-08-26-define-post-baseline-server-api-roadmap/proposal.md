# Define Post-Baseline Server API Roadmap

## Why

Magnetar now has a local Runtime Inference API, CLI boundary, model source/cache
roadmap, and end-to-end local inference conformance target.

After the first local baseline, Magnetar needs a controlled server/API roadmap.

The goal is to expose inference over a server boundary without weakening the
core architecture.

Server mode must remain a wrapper over Runtime Inference API.

It must not bypass:

- Model Loading
- Model Instance lifecycle
- Session lifecycle
- Tokenizer
- Generation
- Sampling
- Kernel Registry
- Memory Manager
- Provider selection
- observability redaction
- source/cache validation

## What Changes

This change defines the post-baseline Server API roadmap.

It introduces roadmap scope for:

- `magnetar serve`
- local-only serve mode
- HTTP API placeholder
- streaming API
- health/readiness endpoints
- model inspection endpoints
- session endpoints
- generation endpoints
- diagnostics endpoints
- OpenAI-compatible facade placeholder
- authentication boundary
- admission and rate policy
- request redaction
- server conformance

This change does not finalize HTTP schemas.

It defines boundaries and required behavior before server implementation.

## Server API Principle

Server API SHALL be a network or local IPC facade around Runtime Inference API.

Canonical flow:

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

The Server API SHALL not call Providers, Kernels, Memory Manager, or Model
Components directly.

## Serve Mode Ownership

`magnetar serve` MAY be implemented in `magnetar-cli`, a companion binary, or a
dedicated server crate.

Regardless of implementation location, serve mode SHALL call Runtime Inference
API.

Serve mode SHALL not become a separate inference path.

## Initial Serve Scope

The first server roadmap SHOULD support local development serving.

Initial scope MAY include:

```text
GET  /health
GET  /ready
GET  /models
GET  /models/{id}
POST /sessions
DELETE /sessions/{id}
POST /generate
POST /generate/stream
POST /cancel/{generation_id}
GET  /diagnostics
```

Endpoint names are illustrative.

Final endpoint syntax is not defined by this change.

## Health Endpoint

Server health endpoint SHOULD report whether the server process is alive.

Health SHALL not imply Runtime readiness.

Health SHALL not imply model availability.

Health response SHALL be redacted.

## Readiness Endpoint

Server readiness endpoint SHOULD report whether Runtime can accept inference
requests under current policy.

Readiness MAY include redacted summaries for:

- Runtime initialized
- cache accessible where configured
- configured Providers registered
- memory policy available
- admission policy available
- model registry/source state where safe

Readiness SHALL not expose raw handles or secrets.

## Model Endpoints

Model endpoints MAY expose redacted model metadata.

They MAY support:

- list known models
- inspect model metadata
- inspect cached model metadata
- inspect loaded Model Instances
- request model loading
- request model unloading

Model endpoints SHALL not bypass Model Source, Cache, Model Artifact, Model
Loading, trust, integrity, or policy validation.

## Session Endpoints

Session endpoints MAY expose Runtime Inference Session creation, inspection, and
closure.

Server sessions SHALL remain Runtime inference sessions.

They SHALL not store workspace, Git, shell, tool, network, or secret state.

Server connection state is separate from Runtime session state.

## Generation Endpoint

Generation endpoint SHALL call Runtime Generation through Runtime Inference API.

Request MAY include:

- model reference or session reference
- prompt text
- chat messages
- already-tokenized input where policy allows
- generation parameters
- sampling parameters
- stop conditions
- streaming flag
- cache policy
- adapter policy
- timeout
- request correlation

Generation endpoint SHALL not execute tools from model output.

## Streaming Endpoint

Streaming endpoint SHALL expose Runtime streaming events.

Streaming MAY use:

- server-sent events
- chunked HTTP
- WebSocket placeholder
- local IPC stream placeholder

The exact transport is implementation-defined.

Streaming SHALL preserve event ordering.

Streaming SHALL not expose raw logits, raw tensor values, raw KV cache contents,
Provider handles, Device handles, Kernel handles, or memory pointers by default.

## Cancellation Endpoint

Cancellation endpoint SHALL call Runtime cancellation.

Cancellation SHALL apply to inference-owned work.

Server may also close transport streams.

Cancellation of source downloads, cache operations, or CLI-owned workflows is
outside this core Server API roadmap unless explicitly implemented.

## Diagnostics Endpoint

Diagnostics endpoint MAY expose redacted Runtime and server diagnostics.

Diagnostics MAY include:

- server status
- Runtime readiness
- Provider readiness summary
- memory pressure summary
- queue/admission status
- loaded model summary
- session summary
- cache hit/miss summary
- recent structured errors

Diagnostics SHALL be redacted by default.

## OpenAI-Compatible Facade Placeholder

Post-baseline roadmap MAY include an OpenAI-compatible facade.

This facade MAY map compatible requests to Runtime Inference API.

The facade SHALL not redefine Runtime semantics.

Unsupported OpenAI fields SHALL fail explicitly or be ignored only according to
documented compatibility policy.

Tool-call fields SHALL not cause Runtime tool execution.

## Authentication Boundary

Server authentication MAY be added post-baseline.

Authentication SHALL be a server boundary concern.

Runtime Inference API SHALL not receive ambient network credentials.

Secrets SHALL not be stored in Runtime diagnostics or observability by default.

## Authorization Boundary

Server authorization MAY control:

- allowed models
- allowed source kinds
- session creation
- generation limits
- streaming permission
- diagnostics access
- cache inspection
- model loading/unloading
- adapter activation

Authorization SHALL not bypass Runtime policy.

## Admission And Rate Policy

Server API SHOULD define admission and rate policy placeholders.

Policy MAY limit:

- concurrent requests
- queued requests
- max tokens
- max sessions
- max loaded models
- memory budget
- streaming connections
- request body size
- prompt size
- source/cache operations

Runtime still owns inference admission.

Server policy may reject before Runtime call.

## Request Size And Prompt Limits

Server SHALL validate request size and prompt limits before Runtime submission
where possible.

Runtime SHALL still validate token limits and generation policy.

Server-side validation does not replace Runtime validation.

## Source And Cache Boundary

Server endpoints MAY reference models from cache or authorized sources.

Server SHALL not perform arbitrary model downloads during generation.

Model pull/import endpoints may be future work and SHALL still use source/cache
contracts.

## Filesystem Boundary

Server API SHALL not expose arbitrary filesystem access.

If local artifact import is later supported, it SHALL use authorized source
contracts and policy.

Generation endpoints SHALL not read arbitrary server files.

## Tool/Shell/Git Boundary

Server API SHALL not execute tools, shell commands, processes, or Git operations
as part of inference.

If future agent server exists, it SHALL be separate from core inference server
and explicitly scoped.

## Error Model

Server API errors SHALL be structured.

Error categories SHOULD include:

- server-api-unavailable
- server-request-invalid
- server-request-too-large
- server-authentication-required
- server-authentication-failed
- server-authorization-denied
- server-rate-limited
- server-admission-rejected
- server-stream-unavailable
- server-stream-interrupted
- server-cancellation-failed
- server-model-not-found
- server-model-load-failed
- server-session-not-found
- server-generation-failed
- server-diagnostics-redacted
- server-source-policy-denied
- server-cache-policy-denied
- server-boundary-violation
- internal-server-api-error

Runtime errors SHALL be preserved or wrapped with structured cause metadata.

## Observability

Server SHOULD emit observations for:

- server started
- server stopped
- request received
- request rejected
- request authorized
- Runtime request submitted
- stream opened
- stream closed
- stream interrupted
- generation completed
- generation failed
- cancellation requested
- diagnostics requested
- model endpoint used
- session endpoint used
- rate limit hit
- admission rejected
- boundary violation detected

Observability SHALL not expose raw prompts, raw model weights, raw tensor values,
raw KV cache contents, secrets, credentials, raw file contents, raw cache paths,
Provider handles, Device handles, Kernel handles, or memory pointers by default.

## Conformance

Server API conformance SHALL validate:

- server uses Runtime Inference API
- health does not imply readiness
- readiness is redacted
- generation endpoint uses Runtime Generation
- streaming preserves order
- cancellation calls Runtime cancellation
- diagnostics are redacted
- model endpoints do not bypass loading
- server does not execute tools/shell/Git
- server does not read arbitrary files
- server does not perform arbitrary downloads during generation
- Runtime structured errors are preserved

## Non-Goals

This change does not:

- finalize HTTP endpoint schema
- implement server mode
- define production authentication
- define TLS configuration
- define OpenAI-compatible schema
- define model pull endpoint
- define agent server
- define tool execution server
- define workspace server
- define distributed Tachyon API
- bypass Runtime Inference API
- expose Runtime internals

## Impact

Magnetar gains a safe post-baseline server/API roadmap.

The server path becomes:

```text
client request
  -> server validation
  -> Runtime Inference API
  -> Runtime inference contracts
  -> redacted response or stream
```

without turning Runtime or server mode into an agent/tool/workspace runtime.