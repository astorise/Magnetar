# server-api-roadmap Specification

## Purpose
TBD - created by archiving change define-post-baseline-server-api-roadmap. Update Purpose after archive.
## Requirements
### Requirement: Post-Baseline Server API Roadmap

Magnetar SHALL define a post-baseline roadmap for serving Runtime inference over
a server or local IPC boundary.

#### Scenario: Roadmap exists

Given Runtime Inference API baseline exists

When server work begins

Then server/API boundaries and conformance requirements are defined.

---

### Requirement: Server API Uses Runtime Inference API

Server API SHALL call Runtime Inference API for inference.

#### Scenario: Generate request

Given server receives generation request

When inference is executed

Then server calls Runtime Inference API rather than Provider or Kernel APIs.

---

### Requirement: Serve Mode Does Not Create Separate Inference Path

`magnetar serve` SHALL not create a separate inference path.

#### Scenario: Serve generation

Given serve mode generates text

When execution is traced

Then Runtime contracts are used.

---

### Requirement: Health Is Not Readiness

Health endpoint SHALL report server process liveness only.

#### Scenario: Server alive but model unavailable

Given server process is alive

And no model is loaded

When health is checked

Then health may be healthy while readiness may be not ready.

---

### Requirement: Readiness Is Redacted

Readiness endpoint responses SHALL be redacted.

Readiness endpoint SHOULD report redacted Runtime readiness metadata.

#### Scenario: Provider ready

Given Provider is ready

When readiness is requested

Then response does not include raw Provider handles.

---

### Requirement: Model Endpoints Preserve Loading Validation

Model endpoints SHALL not bypass Model Source, Cache, Model Artifact, Model
Loading, trust, integrity, compatibility, or policy validation.

#### Scenario: Load model endpoint

Given server receives model load request

When loading runs

Then Model Loading Contract validates the artifact.

---

### Requirement: Session Endpoints Preserve Inference Session Scope

Server session endpoints SHALL manage Runtime Inference Sessions only.

#### Scenario: Create session

Given server creates a session

When Runtime stores session state

Then it does not store workspace, Git, shell, tool, network, or secret state.

---

### Requirement: Generation Endpoint Does Not Execute Tools

Generation endpoint SHALL not execute tools, shell commands, Git operations, or
network calls from model output.

#### Scenario: Tool-call-like output

Given generated text resembles a tool call

When server streams it

Then server does not execute it as part of core inference server behavior.

---

### Requirement: Streaming Preserves Runtime Event Ordering

Streaming endpoint SHALL preserve Runtime streaming event ordering.

#### Scenario: Stream generation

Given Runtime emits prefill-completed before decode-token

When server forwards events

Then ordering is preserved.

---

### Requirement: Cancellation Calls Runtime Cancellation

Cancellation endpoint SHALL call Runtime cancellation for inference work.

#### Scenario: Cancel request

Given active generation exists

When server receives cancel request

Then Runtime cancellation is invoked.

---

### Requirement: Diagnostics Are Redacted

Diagnostics endpoint SHALL be redacted by default.

#### Scenario: Diagnostics requested

Given diagnostics include Provider summary

When server responds

Then raw handles, pointers, secrets, and raw prompts are absent.

---

### Requirement: OpenAI-Compatible Facade Is Optional

If added, the OpenAI-compatible facade SHALL map to Runtime Inference API and
SHALL not redefine Runtime semantics.

It MAY be added as an optional post-baseline layer.

#### Scenario: Unsupported field

Given request includes unsupported compatibility field

When facade validates it

Then field is rejected or handled according to documented compatibility policy.

---

### Requirement: Authentication Is Server Boundary

Authentication SHALL be server concern and SHALL not grant Runtime ambient
credentials.

#### Scenario: Authenticated request

Given server authenticates client

When Runtime request is submitted

Then Runtime receives explicit request data, not credential authority.

---

### Requirement: Authorization Does Not Bypass Runtime Policy

Server authorization SHALL not bypass Runtime policy.

#### Scenario: Authorized server user

Given server user is authorized for generation

When Runtime policy rejects memory admission

Then generation still fails.

---

### Requirement: Admission And Rate Policy

Server API SHALL define admission and rate policy boundaries.

Concrete limits SHOULD be defined as placeholders pending implementation.

#### Scenario: Too many requests

Given concurrent request limit is reached

When new request arrives

Then server rejects or queues according to policy.

---

### Requirement: Source And Cache Boundary

Server SHALL not perform arbitrary downloads during generation.

#### Scenario: Remote model in generate

Given generation request includes arbitrary remote model URL

When server validates it

Then request is rejected or routed only through authorized source contracts.

---

### Requirement: Filesystem Boundary

Server generation endpoints SHALL not read arbitrary server files.

#### Scenario: Prompt asks read file

Given request asks server to read `/etc/passwd`

When generation endpoint validates it

Then request is rejected as outside inference scope.

---

### Requirement: Tool Shell Git Boundary

Core Server API SHALL not execute tools, shell commands, processes, or Git
operations.

#### Scenario: Generated shell command

Given generated text contains shell command

When server streams it

Then server does not execute it.

---

### Requirement: Server Error Categories

Server API failures SHALL use structured error categories and preserve Runtime
structured causes.

#### Scenario: Runtime loading failure

Given Runtime returns model-loading-failed

When server responds

Then structured Runtime cause is preserved.

---

### Requirement: Server Observability

Server observations SHALL be redacted by default.

Server SHOULD emit server observations for key lifecycle events.

#### Scenario: Request observed

Given generation request is received

When observation is emitted

Then raw prompt and credentials are not logged by default.

---

### Requirement: Server Conformance

Server API conformance SHALL validate Runtime API usage, redaction, streaming,
cancellation, model loading boundary, filesystem boundary, source/cache boundary,
and tool/shell/Git boundary.

#### Scenario: Server bypass

Given server implementation calls Kernel Registry directly for generation

When conformance runs

Then conformance fails.

