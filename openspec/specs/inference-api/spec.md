# inference-api Specification

## Purpose
TBD - created by archiving change define-runtime-inference-api. Update Purpose after archive.
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

Runtime Inference API SHALL expose explicit model loading or policy-controlled implicit loading.

#### Scenario: Explicit load

Given a valid model reference

When loading request is submitted

Then Runtime validates artifact, component, memory, provider, device, and policy
contracts before creating a ready Model Instance.

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

