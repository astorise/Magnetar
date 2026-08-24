# Define Runtime Inference API

## Why

Magnetar now has the internal contracts required for inference:

- Model Artifact
- Model Loading
- Model Instance lifecycle
- Model Component
- Qwen Model Component baseline
- Tokenizer
- Generation
- Sampling
- Inference Session
- KV Cache
- Prefix Cache
- Continuous Batching
- Tensor Resource
- Execution Graph
- Operator
- Kernel Registry and Dispatch
- Reference CPU Provider

The next step is to expose a stable Runtime Inference API.

This API is the boundary used by first-party and external callers to ask
Magnetar to perform inference.

It must be high-level enough to avoid leaking internal handles, but explicit
enough to preserve Runtime policy, resource usage, cancellation, streaming,
diagnostics, and model/session lifecycle.

It must also preserve the primary boundary:

```text
Magnetar = inference runtime
magnetar-cli = workspace / file / network / secret / tool / agent runtime
```

The Runtime Inference API SHALL not grow into an agent or tool API.

## What Changes

This change introduces a Runtime Inference API.

The API SHALL expose inference operations such as:

- resolve model
- load model
- create model instance
- inspect model instance
- create inference session
- prepare prompt
- tokenize input
- generate
- stream generation events
- cancel generation
- close session
- unload model instance
- inspect diagnostics

The exact Rust type names are implementation-defined.

## API Scope

Runtime Inference API SHALL be limited to inference.

Allowed API areas include:

```text
model resolution
model loading
model instance lifecycle
tokenizer execution
prompt/chat-template formatting where Runtime-owned
session creation
generation execution
sampling configuration
streaming output
KV cache policy
Prefix Cache policy
adapter activation where inference-scoped
runtime diagnostics
observability metadata
cancellation
resource usage reporting
```

Forbidden API areas include:

```text
workspace filesystem
arbitrary filesystem
Git
network tools
shell
process execution
secrets
external services
agent orchestration
source editing
task automation
```

## API Boundary

Runtime Inference API SHALL not expose raw internal handles.

It SHALL not expose:

- Provider handles
- Device handles
- Kernel handles
- raw tensor pointers
- raw memory pointers
- raw KV cache contents
- raw model weights
- raw prompt text in diagnostics by default
- Provider-owned opaque resource internals

All identifiers exposed by the API SHALL be Runtime-issued opaque identifiers.

## Primary Concepts

Runtime Inference API SHOULD expose or reference:

```text
ModelRef
ModelArtifactRef
ModelInstanceRef
InferenceSessionRef
GenerationRequest
GenerationHandle
GenerationEvent
GenerationResult
TokenizationRequest
TokenizationResult
StreamingHandle
CancellationToken
RuntimeDiagnostics
UsageReport
```

Names are illustrative.

The implementation may choose different type names.

## Model Resolution

The API SHALL support resolving a model reference into Runtime-known model
metadata.

Resolution may target:

- local Runtime registry
- client-provided artifact reference
- trusted cache
- development fixture
- future external source through validated distribution contract
- future Tachyon-provided source through neutral source contract

Model resolution SHALL not grant arbitrary filesystem or network access.

## Model Loading API

The API SHALL expose explicit model loading or policy-controlled implicit loading.

Model loading request SHOULD include:

- model reference
- optional tokenizer reference
- optional adapter references
- target usage
- dtype policy
- layout policy
- memory budget
- cache policy
- provider policy preferences
- timeout
- observability correlation

Provider and Device preferences SHALL be non-authoritative policy inputs.

Runtime SHALL own Provider and Device selection.

## Model Instance API

The API SHALL expose Model Instance lifecycle operations.

Operations MAY include:

- create instance
- inspect instance
- warm instance
- suspend instance
- resume instance
- drain instance
- unload instance

Lifecycle operations SHALL respect active sessions, active generations, memory
pressure, Provider/Device readiness, adapter state, and policy.

## Session API

The API SHALL expose Inference Session creation and closure.

Session creation request SHOULD include:

- Model Instance reference or model reference
- tokenizer reference or policy
- generation defaults override
- memory budget
- KV cache policy
- Prefix Cache policy
- adapter activation policy
- streaming policy
- cancellation policy
- timeout and idle TTL
- privacy/redaction policy
- observability correlation

A session SHALL not own workspace, files, tools, Git, shell, network, or secrets.

## One-Shot Inference

The API MAY support one-shot inference.

One-shot inference SHALL be modeled as policy-controlled implicit session
creation, generation execution, and session close.

One-shot inference SHALL not bypass Model Instance, Session, Generation,
Tokenizer, Sampling, Memory Manager, Provider, or Kernel contracts.

## Tokenization API

The API SHALL expose tokenization through the Tokenizer Contract.

Tokenization MAY include:

- encode prompt text
- apply chat template where Runtime-owned and authorized
- decode tokens
- streaming decode
- return token usage metadata
- validate tokenizer/model compatibility

Tokenization SHALL not expose tokenizer internals beyond stable metadata.

Raw prompt logging SHALL be disabled by default.

## Prompt Input

Runtime Inference API may accept prompt input forms such as:

```text
plain text
chat messages
already-tokenized input
test token sequence
```

If chat messages are accepted, chat-template formatting SHALL occur through
Runtime-authorized prompt/template contracts.

The API SHALL not perform external retrieval, file reading, workspace scanning,
or tool execution.

Those responsibilities belong to clients such as `magnetar-cli`.

## Generation API

The API SHALL expose generation requests.

Generation request SHOULD include:

- session or model reference
- input token IDs or prompt input
- generation parameters
- sampling parameters
- stop conditions
- streaming mode
- max new tokens
- max total tokens
- priority
- timeout/deadline
- cancellation token
- observability correlation
- privacy/redaction policy

Generation SHALL execute through the Generation Contract.

## Streaming API

The API SHALL expose streaming generation events.

Streaming events SHOULD include:

```text
generation-started
prefill-started
prefill-completed
decode-token
decoded-text
stop-reached
generation-completed
generation-failed
generation-cancelled
usage-updated
diagnostic
```

Events SHALL have stable ordering guarantees.

Raw logits, raw KV cache contents, raw tensor values, raw model weights, and raw
Provider handles SHALL not be streamed by default.

## Generation Result

Generation result SHOULD include:

- generated token IDs where policy allows
- decoded text where requested
- finish reason
- usage accounting
- timing metadata
- cache usage metadata
- model instance metadata
- structured diagnostics
- redaction status
- error information

Result SHALL not include raw internal handles.

## Cancellation API

The API SHALL support cancellation.

Cancellation SHALL propagate through:

- queued generation
- tokenization where applicable
- prefill
- decode
- sampling
- batching
- graph execution
- Kernel Dispatch
- Provider execution where supported

If a Provider or Kernel does not support cancellation after dispatch, Runtime
SHALL report the limitation.

## Backpressure

The API SHALL expose admission and backpressure behavior.

Requests may be:

```text
accepted
queued
rejected
delayed
cancelled
timed-out
```

Backpressure SHALL be structured and observable.

The API SHALL not hide queueing or resource admission failures.

## Adapter Activation API

The API MAY expose inference-scoped adapter activation.

Adapter activation SHALL be explicit and policy-controlled.

Adapter activation MAY be scoped to:

- operation
- session
- model instance

Adapter activation SHALL respect Model Component compatibility, cache
compatibility, memory budget, and policy.

## KV Cache API

The API SHALL expose KV cache policy inputs without exposing raw cache contents.

KV cache policy MAY include:

- enabled or disabled
- scope
- budget
- reuse policy
- eviction policy
- privacy policy
- persistence policy placeholder

The API SHALL not allow clients to mutate raw KV cache contents directly.

## Prefix Cache API

The API SHALL expose Prefix Cache policy inputs without exposing raw prompts or
raw KV cache contents.

Prefix Cache policy MAY include:

- enabled or disabled
- scope
- sharing policy
- TTL
- budget
- privacy policy
- reuse policy

Prefix Cache matching remains Runtime-owned.

## Diagnostics API

The API SHALL expose structured diagnostics.

Diagnostics MAY include:

- model resolution status
- model loading status
- Model Instance status
- Provider readiness summary
- Device readiness summary
- memory pressure summary
- Kernel availability summary
- operator missing summary
- tokenizer compatibility status
- cache status summary
- queue/admission status
- error traces with redaction

Diagnostics SHALL be redacted by default.

## Usage Reporting

The API SHOULD expose usage reporting.

Usage MAY include:

- prompt token count
- generated token count
- total token count
- prefill duration
- decode duration
- tokens per second where available
- cache hit/miss metadata
- memory estimate
- queued time
- cancellation status

Usage reporting SHALL not expose raw prompts by default.

## Error Model

Runtime Inference API errors SHALL be structured.

Error categories SHOULD include:

- inference api unavailable
- model reference invalid
- model resolution failed
- model loading failed
- model instance not ready
- model instance unavailable
- model component unavailable
- tokenizer unavailable
- tokenizer incompatible
- tokenization failed
- session creation failed
- session not found
- session closed
- generation rejected
- generation queued
- generation timeout
- generation cancelled
- generation failed
- sampling failed
- stop condition invalid
- adapter activation failed
- KV cache unavailable
- Prefix Cache unavailable
- memory admission failed
- Provider unavailable
- Device unavailable
- Kernel unavailable
- operator unsupported
- graph planning failed
- policy denied
- cancellation unsupported
- streaming unavailable
- streaming interrupted
- diagnostics redacted
- browser feature unsupported
- internal inference api error

## Observability

Runtime SHOULD emit observations for:

- inference request received
- model resolved
- model resolution failed
- model loading requested
- model loaded
- model loading failed
- model instance selected
- session created
- session closed
- prompt tokenized
- tokenization failed
- generation accepted
- generation queued
- generation started
- prefill started
- prefill completed
- decode started
- token generated
- generation completed
- generation failed
- generation cancelled
- adapter activated
- KV cache used
- Prefix Cache hit
- Prefix Cache miss
- memory admission failed
- Provider unavailable
- Kernel unavailable
- stream opened
- stream closed
- stream interrupted

Observability SHALL not expose raw prompts, raw model weights, raw tensor values,
raw KV cache contents, Provider handles, Device handles, Kernel handles, memory
pointers, filesystem paths, secrets, or external service credentials by default.

## Browser Target

Runtime Inference API SHALL be platform-neutral.

Browser targets may support reduced inference paths.

Browser targets SHALL not require:

- Wasmtime
- native Provider loading
- arbitrary filesystem access
- process execution
- shell execution
- native memory mapping

Unsupported browser features SHALL return structured errors.

## Tachyon Relationship

Tachyon may call Runtime Inference API through an adapter boundary.

Tachyon remains responsible for distributed service orchestration.

Magnetar remains responsible for local inference validation and execution.

Tachyon SHALL not bypass Magnetar Runtime validation, Model Instance lifecycle,
Kernel Registry, Memory Manager, or Provider contracts.

## magnetar-cli Relationship

`magnetar-cli` may call Runtime Inference API.

`magnetar-cli` owns:

- workspace
- files
- Git
- network
- secrets
- shell/process execution
- tools
- agent orchestration
- user interaction beyond inference

Runtime Inference API SHALL not absorb those responsibilities.

## Non-Goals

This change does not:

- define CLI commands
- define workspace management
- define Git integration
- define tool calling
- define agent orchestration
- define external retrieval
- define model download UX
- define HTTP server API
- define Tachyon distribution protocol
- define remote execution
- expose raw Provider handles
- expose raw Device handles
- expose raw Kernel handles
- expose raw tensor pointers
- require GPU hardware
- require browser implementation

## Impact

Magnetar gains a stable public Runtime-level inference façade.

The first end-to-end path becomes expressible as:

```text
resolve model
  -> load/create Model Instance
  -> create Session
  -> tokenize prompt
  -> generate
  -> stream tokens
  -> close Session
```

This prepares:

- magnetar-cli inference boundary
- end-to-end local inference conformance
- future server API