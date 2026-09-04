# inference-api Specification

## Purpose
This specification defines the Runtime Inference API boundary for model loading, sessions, generation, streaming, diagnostics, and CLI/server callers.
## Requirements
### Requirement: Runtime Inference API

Magnetar SHALL define a Runtime Inference API for model resolution, loading, session management, tokenization, generation, streaming, cancellation, diagnostics, and usage reporting.

#### Scenario: Run inference

Given a valid model reference and prompt input

When a caller invokes Runtime Inference API

Then Runtime resolves or loads the model, creates or uses a session, tokenizes
input, runs generation, and returns or streams generated output.

---

### Requirement: Inference-Only Scope

Runtime Inference API SHALL be limited to inference responsibilities. It SHALL NOT own workspace filesystem, arbitrary filesystem, Git, network tools, shell, process execution, secrets, external services, source editing, or agent orchestration.

#### Scenario: File read request

Given a caller asks Runtime Inference API to read a workspace file

When Runtime validates the request

Then the request is rejected as outside inference scope.

---

### Requirement: API Handle Safety

Runtime Inference API SHALL not expose raw Provider handles, Device handles, Kernel handles, tensor pointers, memory pointers, raw KV cache contents, raw model weights, or Provider-owned opaque internals.

#### Scenario: Inspect generation result

Given generation completes

When result is returned

Then it contains stable metadata and no raw internal handles.

---

### Requirement: Model Resolution API

Runtime Inference API SHALL support resolving model references into Runtime-known model metadata without granting arbitrary filesystem or network access.

#### Scenario: Resolve local model

Given model reference points to Runtime registry entry

When resolution runs

Then Runtime returns validated model metadata.

---

### Requirement: Model Loading API

Runtime Inference API SHALL expose explicit model loading or policy-controlled implicit loading, and the trust decision used to authorize that load SHALL be sourced from the Runtime instance performing it rather than accepted as a loading-call parameter.

#### Scenario: Explicit load

Given a valid model reference

When loading request is submitted

Then Runtime validates artifact, component, memory, provider, device, and policy
contracts before creating a ready Model Instance.

---

#### Scenario: Loading API does not accept a caller-supplied trust decision

Given a loading request for a model artifact

When the Model Loading API call is made

Then the trust decision applied is the one the performing Runtime instance was configured with, and the API surface provides no parameter through which a caller can substitute a different trust decision for that call

---

### Requirement: Provider Preferences Are Non-Authoritative

Provider and Device preferences in API requests SHALL be policy inputs only. Runtime SHALL own Provider and Device selection.

#### Scenario: Caller requests CUDA

Given caller prefers CUDA

When Runtime resolves execution

Then Runtime may consider preference but SHALL not violate Resource Affinity,
capability, memory, readiness, or policy constraints.

---

### Requirement: Model Instance API

Runtime Inference API SHALL expose Model Instance lifecycle operations without exposing internal handles.

#### Scenario: Unload instance

Given Model Instance has no active use

When unload is requested

Then Runtime drains or unloads according to policy.

---

### Requirement: Session API

Runtime Inference API SHALL expose Inference Session creation and closure. A session SHALL not own workspace, files, tools, Git, shell, network, or secrets.

#### Scenario: Create session

Given a ready Model Instance

When session creation request is valid

Then Runtime creates an Inference Session with policy metadata.

---

### Requirement: One-Shot Inference

One-shot inference SHALL be modeled as policy-controlled implicit session creation, generation, and session close.

#### Scenario: One-shot prompt

Given one-shot request is submitted

When policy allows implicit session

Then Runtime creates a short-lived session and runs generation through normal
contracts.

---

### Requirement: Tokenization API

Runtime Inference API SHALL expose tokenization through the Tokenizer Contract. Raw prompt logging SHALL be disabled by default.

#### Scenario: Encode text

Given prompt text is submitted

When tokenization runs

Then Tokenizer Contract produces token IDs and usage metadata.

---

### Requirement: Prompt Input Boundary

Runtime Inference API MAY accept plain text, chat messages, already-tokenized input, or test token sequences. It SHALL not perform external retrieval, file reading, workspace scanning, or tool execution.

#### Scenario: Chat messages

Given chat messages are submitted

When Runtime prepares input

Then chat-template formatting occurs only through authorized Runtime prompt
contracts.

---

### Requirement: Generation API

Runtime Inference API SHALL expose generation requests through the Generation Contract.

#### Scenario: Generate tokens

Given session and tokenized input are valid

When generation starts

Then Runtime executes prefill/decode and Sampling according to policy.

---

### Requirement: Streaming API

Runtime Inference API SHALL expose ordered streaming events for generation progress and output. Events SHALL not expose raw logits, KV cache contents, tensor values, model weights, or Provider handles by default.

#### Scenario: Token streamed

Given generation is in decode phase

When a token is produced

Then Runtime emits a decode-token event and decoded-text event where requested.

---

### Requirement: Generation Result

Generation result SHALL include stable output, finish reason, usage, diagnostics, and error metadata without raw internal handles.

#### Scenario: Generation stopped

Given stop condition is reached

When result is returned

Then finish reason and usage metadata are included.

---

### Requirement: Cancellation API

Runtime Inference API SHALL support cancellation and report unsupported cancellation limitations.

#### Scenario: Cancel during Provider execution

Given Kernel does not support interruption after dispatch

When cancellation is requested

Then Runtime reports cancellation limitation and follows policy.

---

### Requirement: Backpressure

Runtime Inference API SHALL expose admission and backpressure states.

#### Scenario: Memory pressure

Given memory admission fails due to pressure

When generation is requested

Then request is rejected, delayed, or queued with structured metadata.

---

### Requirement: Adapter Activation API

Runtime Inference API MAY expose inference-scoped adapter activation. Adapter activation SHALL be explicit and policy-controlled.

#### Scenario: Adapter incompatible

Given adapter is incompatible with active Model Component

When activation is requested

Then Runtime rejects activation.

---

### Requirement: KV Cache Policy API

Runtime Inference API SHALL expose KV cache policy inputs without exposing raw cache contents.

#### Scenario: KV cache enabled

Given session requests KV cache reuse

When session is created

Then Runtime applies KV cache policy without granting raw cache mutation.

---

### Requirement: Prefix Cache Policy API

Runtime Inference API SHALL expose Prefix Cache policy inputs without exposing raw prompts or raw KV cache contents.

#### Scenario: Prefix cache hit

Given Prefix Cache policy allows reuse

When matching prefix exists

Then Runtime may report cache hit metadata without exposing raw prompt.

---

### Requirement: Diagnostics API

Runtime Inference API SHALL expose structured, redacted diagnostics.

#### Scenario: Missing Kernel

Given generation fails because no Kernel supports an Operator

When diagnostics are requested

Then Runtime returns missing Kernel summary without raw handles.

---

### Requirement: Usage Reporting

Runtime Inference API SHALL expose token, timing, cache, memory, queue, and cancellation usage metadata when available.

#### Scenario: Usage returned

Given generation completes

When result is returned

Then prompt token count, generated token count, and finish reason are included.

---

### Requirement: Structured Error Model

Runtime Inference API failures SHALL use structured errors.

#### Scenario: Session closed

Given caller uses a closed session

When generation is requested

Then Runtime returns session-closed.

---

### Requirement: Runtime Inference Observability

Runtime SHOULD emit inference API observations for request, resolution, loading, session, tokenization, generation, streaming, cache, memory, Provider, Kernel, and failure events. Observability SHALL be redacted by default.

#### Scenario: Request observed

Given inference request is received

When observability records it

Then raw prompt and raw model weights are not logged by default.

---

### Requirement: Browser-Compatible Inference API

Runtime Inference API SHALL be platform-neutral and SHALL not require Wasmtime, native Provider loading, arbitrary filesystem access, process execution, shell execution, or native mmap.

#### Scenario: Browser unsupported feature

Given browser Runtime receives request requiring native Provider loading

When validation runs

Then Runtime returns browser-feature-unsupported.

---

### Requirement: Tachyon Boundary

Tachyon may call Runtime Inference API but SHALL not bypass Runtime validation, Model Instance lifecycle, Kernel Registry, Memory Manager, or Provider contracts.

#### Scenario: Tachyon source

Given Tachyon supplies model component source metadata

When Runtime loads it

Then Runtime still validates artifact, trust, authority, and execution contracts.

---

### Requirement: magnetar-cli Boundary

`magnetar-cli` may call Runtime Inference API but SHALL own workspace, file, Git, network, secrets, shell/process, tools, agent orchestration, and user interaction beyond inference.

#### Scenario: CLI prompt from file

Given CLI reads a file and builds prompt context

When it calls Runtime Inference API

Then Runtime receives prompt input and does not read the workspace file itself.

---

### Requirement: Runtime Inference API Is Called By CLI

`magnetar-cli` SHALL call Runtime Inference API for inference operations.

#### Scenario: CLI run

Given user executes `magnetar run`

When generation is needed

Then CLI calls Runtime Inference API.

---

### Requirement: Runtime Inference API Receives Explicit Context

Runtime Inference API SHALL receive explicit prompt/context data from CLI and
not CLI authority.

#### Scenario: File context

Given CLI reads a file

When Runtime request is built

Then request contains selected content or references allowed by contract, not
filesystem authority.

---

### Requirement: Runtime Inference API Does Not Execute CLI Tools

Runtime Inference API SHALL not execute CLI tools or shell commands.

#### Scenario: Tool-like output

Given generated output contains tool syntax

When Runtime returns it

Then Runtime does not execute the tool.

---

### Requirement: E2E Uses Runtime Inference API

End-to-End Local Inference Conformance SHALL enter inference through Runtime
Inference API.

#### Scenario: E2E request

Given local E2E test starts inference

When request is submitted

Then it uses Runtime Inference API, not internal Provider or Kernel APIs.

---

### Requirement: E2E Validates One-Shot Inference

E2E suite MAY validate one-shot inference, but one-shot SHALL use normal Runtime
contracts internally.

#### Scenario: One-shot E2E

Given one-shot request runs

When tracing or observations are inspected

Then implicit session, tokenization, generation, and dispatch contracts were
used.

---

### Requirement: E2E Validates API Errors

E2E suite SHALL validate structured Runtime Inference API errors for failure
cases.

#### Scenario: Invalid model reference

Given invalid model reference is submitted

When API validates it

Then structured model resolution or invalid reference error is returned.

---

### Requirement: Runtime Inference API Implemented After Core Baseline

Runtime Inference API baseline SHALL be implemented after Tensor, Memory,
Operators, Reference CPU, Kernel Registry, Model Loading, Tokenizer, Qwen
baseline, Generation, and Sampling are sufficiently available.

#### Scenario: API success path

Given Runtime Inference API accepts request

When generation completes

Then it uses the implemented core baseline instead of fake responses.

---

### Requirement: Inference API Baseline Is Inference-Only

Runtime Inference API implementation SHALL not add workspace, Git, tool, shell,
network, secret, or agent responsibilities.

#### Scenario: Tool execution request

Given API request asks Runtime to execute a tool

When validation runs

Then Runtime rejects it.

### Requirement: Inference API Accepts Normalized Model References

Runtime Inference API MAY accept model references that resolve to normalized Model Artifacts from supported formats, and Model Loading SHALL apply standard validation to every such reference.

#### Scenario: safetensors model reference

Given caller references a safetensors-based model

When Runtime resolves it

Then Runtime loads the normalized Model Artifact through standard loading.

---

### Requirement: Inference API Does Not Download Formats Arbitrarily

Runtime Inference API SHALL not perform arbitrary model downloads during
inference.

#### Scenario: Remote URL inference

Given inference request contains remote model URL

When Runtime validates it

Then Runtime uses authorized source contracts or rejects arbitrary network
access.

### Requirement: Inference API Uses Source Resolution Safely

Runtime Inference API MAY resolve ModelRefs through authorized source/cache contracts, and resolution SHALL validate the resulting artifact before it is used.

#### Scenario: Cached model reference

Given inference request references cached model

When Runtime resolves it

Then cache entry is validated before loading.

---

### Requirement: Inference API Does Not Gain Download Authority

Runtime Inference API SHALL not perform arbitrary model downloads during
inference.

#### Scenario: Download requested in inference

Given inference request asks Runtime to download model from arbitrary URL

When validation runs

Then Runtime rejects it or delegates only through authorized source contract.

### Requirement: Server Facade Uses Inference API

Server API SHALL use Runtime Inference API for model, session, generation,
streaming, cancellation, diagnostics, and usage operations.

#### Scenario: Server generation

Given server receives generation request

When Runtime work is required

Then Runtime Inference API is called.

---

### Requirement: Server Facade Does Not Expose Runtime Internals

Server API SHALL not expose Runtime internal handles through Inference API
responses.

#### Scenario: Provider diagnostic

Given server returns diagnostics

When response is inspected

Then Provider handles, Device handles, Kernel handles, and memory pointers are
absent.

### Requirement: Inference API Compatibility Status

Release metadata SHALL declare Runtime Inference API compatibility status.

#### Scenario: API status

Given `v0.1` is released

When compatibility notes are inspected

Then Runtime Inference API is marked stable-for-baseline or unstable as
appropriate.

---

### Requirement: Inference API Release Safety

Runtime Inference API release SHALL not expose internal handles.

#### Scenario: Release API audit

Given release API docs are generated

When responses are inspected

Then Provider, Device, Kernel, tensor pointer, memory pointer, KV cache, and raw
weight internals are absent.

### Requirement: Inference API Release Security

Runtime Inference API release SHALL remain inference-only and redacted by
default.

#### Scenario: Release inference diagnostics

Given diagnostics are requested after inference

When response is returned

Then raw prompt, raw weights, raw tensors, raw KV cache, secrets, credentials,
handles, and memory pointers are absent by default.

---

### Requirement: Inference API Rejects Non-Inference Authority

Runtime Inference API SHALL reject requests for filesystem, network, secret,
shell, process, Git, tool, or agent authority.

#### Scenario: Tool execution request

Given inference request asks Runtime to execute tool

When Runtime validates it

Then request is rejected.

### Requirement: Inference API Release Gate Coverage

Runtime Inference API SHALL have release gate coverage for baseline inference.

#### Scenario: One-shot inference

Given one-shot inference is included in `v0.1`

When release gate runs

Then one-shot path is tested through normal Runtime contracts.

### Requirement: Inference API Cutover Status

Cutover SHALL confirm Runtime Inference API status in compatibility matrix.

#### Scenario: Missing status

Given Runtime Inference API has no status

When cutover runs

Then release is blocked.

---

### Requirement: Inference API Baseline Verified Before Tag

Runtime Inference API baseline gates SHALL pass before stable tag creation.

#### Scenario: API gate after tag

Given stable tag is created before API gate passes

When cutover validates sequence

Then release is invalid.

---

### Requirement: Inference API Excludes Optimization Orchestration

Runtime Inference API SHALL NOT expose arbitrary Kernel optimization-agent
execution.

#### Scenario: Generate request contains optimizer URL

Given caller submits optimization-service URL with generation request

When request is validated

Then this authority is rejected as outside inference scope.

---

### Requirement: Inference API Does Not Accept Arbitrary Kernel Source

Normal inference requests SHALL NOT inject executable Kernel source.

#### Scenario: Prompt request carries CUDA source

Given generation request attempts to override MatMul with supplied CUDA code

When Runtime validates request

Then source injection is rejected.

---

### Requirement: Inference Session Does Not Own Optimization Credentials

Inference Session SHALL NOT hold generator/compiler-service credentials.

#### Scenario: External optimizer has API token

Given optimizer token exists

When inference session is inspected

Then token is absent.

### Requirement: Inference API Accepts Model And Generation Intent

RuntimeInferenceApi SHALL expose model/session/generation operations without
requiring model execution implementation from caller.

#### Scenario: Text generation

Given caller selects loaded Qwen fixture and supplies prompt

When generate is invoked

Then Runtime performs tokenization/model execution/sampling.

### Requirement: Caller Cannot Inject Ordinary Forward Function

RuntimeInferenceApi SHALL not expose caller-supplied forward/logits callback as
normal inference authority.

#### Scenario: Caller tries to supply logits closure

Given normal generation API

When request is constructed

Then no such callback is required or authoritative.

### Requirement: Cancellation Is Runtime Owned

RuntimeInferenceApi SHALL allow caller-requested cancellation without exposing
or delegating Provider execution details to the caller.

#### Scenario: Generation cancelled

Given request is active

When caller cancels

Then Runtime propagates cancellation through Session/execution contracts.

### Requirement: Native Handles Are Not Exposed

RuntimeInferenceApi SHALL not expose Kernel, Provider, Tensor native pointer, or
ExecutionStream native handle.

#### Scenario: Client receives token

Given internal execution uses prepared Kernels

When output is returned

Then only high-level inference data/status is exposed.

### Requirement: RuntimeInferenceApi Cutover Removes Model Callback Authority

The final normal inference API SHALL not require a caller-provided model-forward
or logits callback.

#### Scenario: API type inspected

Given caller wants text generation

When constructing request

Then prompt/model/generation configuration is sufficient.

### Requirement: Migration Helper Is Non-Authoritative

If a callback-based helper remains temporarily, it SHALL be clearly outside the
mandatory production first-profile API.

#### Scenario: Legacy test uses helper

Given helper remains for isolated tests

When native profile conformance runs

Then helper is not exercised.

### Requirement: API E2E Precedes CLI E2E

A RuntimeInferenceApi integration test SHALL succeed before CLI integration is
considered complete.

#### Scenario: CLI not yet implemented

Given Runtime API can execute fixture prompt

When API test runs

Then generated result can be verified independently.

### Requirement: Chat Uses Persistent Runtime Session
The chat CLI SHALL execute all turns of a ChatSession through its persistent Runtime and InferenceSession.

#### Scenario: Two chat turns execute
- **WHEN** two turns are submitted through one ChatSession
- **THEN** both turns use the same Runtime InferenceSession identifier.

#### Scenario: Chat is cancelled
- **WHEN** ChatSession cancellation is requested
- **THEN** cancellation targets the Runtime session used by chat turns and blocks new turns.

### Requirement: Production Inference API Rejects Logits Injection
RuntimeInferenceApi SHALL NOT expose a production API that lets callers provide logits, substitute model execution, or install a per-request forward callback.

#### Scenario: Caller cannot inject logits
- **WHEN** production callers build a generation request
- **THEN** the request contains prompt/session/model parameters but no logits array or forward callback.

#### Scenario: Synthetic support is test-only
- **WHEN** tests require synthetic logits
- **THEN** the support is gated by `#[cfg(test)]` or an explicitly non-production conformance feature.

### Requirement: Runtime Owns Model Execution Chain
RuntimeInferenceApi SHALL own the chain from model reference or loaded ModelInstance to graph planning, PreparedExecutionPlan execution, logits production, Sampling, streaming, and cleanup.

#### Scenario: Generation fails without model instance
- **WHEN** generation cannot resolve or access a valid ready ModelInstance
- **THEN** RuntimeInferenceApi rejects generation with a structured model error.

