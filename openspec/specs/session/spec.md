# session Specification

## Purpose
TBD - created by archiving change define-inference-session-model. Update Purpose after archive.
## Requirements
### Requirement: Inference Session

Magnetar SHALL define an Inference Session as a Runtime-owned inference context.

An Inference Session SHALL bind model, tokenizer, policy, memory, streaming,
cancellation, and future cache state for inference operations.

#### Scenario: Create session

Given a valid model reference and compatible tokenizer reference

When Runtime creates an Inference Session

Then the session is created as a Runtime-owned inference context.

---

### Requirement: Session Is Runtime-Owned

Inference Sessions SHALL be created, identified, authorized, and destroyed by
the Runtime.

Clients and Components SHALL NOT forge session identity.

#### Scenario: Forged session ID

Given a caller submits a fabricated session ID

When Runtime resolves the session

Then Runtime rejects it as session-not-found or unauthorized.

---

### Requirement: Session Is Not Client Conversation

An Inference Session SHALL NOT store client conversation, workspace, tool,
filesystem, Git, network, or secret state.

#### Scenario: Client conversation exists

Given a client has conversation messages and workspace files

When Runtime creates an Inference Session

Then only inference-scoped state is held by the session.

---

### Requirement: Session Is Not Model Artifact

A validated Model Artifact SHALL NOT automatically create an Inference Session.

#### Scenario: Model validated

Given a Model Artifact validates and is trusted

When no session creation is requested

Then no Inference Session is created.

---

### Requirement: Session Is Not KV Cache

A session SHALL be able to own or reference future KV cache state.

The KV cache model SHALL be defined separately.

#### Scenario: Session prepared for generation

Given a session is ready

When future KV cache is needed

Then the session may reference cache state without defining cache layout here.

---

### Requirement: Session Lifecycle

An Inference Session SHALL have a lifecycle.

Lifecycle states SHOULD include creating, ready, active, idle, draining,
cancelled, failed, closed, and expired.

#### Scenario: Session becomes ready

Given session creation validation succeeds

When Runtime finishes setup

Then the session transitions to ready.

---

### Requirement: Session Creation Validation

Session creation SHALL validate model availability, tokenizer compatibility,
generation defaults, Runtime policy, memory limits, session TTL, concurrency,
streaming, and cancellation policy.

#### Scenario: Incompatible tokenizer

Given a model requires tokenizer A

And session creation requests tokenizer B

When Runtime validates the session

Then creation fails with tokenizer-incompatible.

---

### Requirement: Session Binding

A session SHALL bind model reference, tokenizer reference, generation defaults,
policy, memory budget, and observability correlation.

#### Scenario: Generate in session

Given a session binds model M and tokenizer T

When generation runs in that session

Then Runtime uses M and T unless explicit policy allows override.

---

### Requirement: Session Policy

A session SHALL carry or reference inference policy.

Policy SHALL remain inference-scoped and SHALL NOT grant filesystem, network,
Git, secrets, workspace, or process authority.

#### Scenario: Policy requests filesystem

Given a session policy attempts to grant filesystem authority

When Runtime validates it

Then validation rejects the policy.

---

### Requirement: Session Resources

A session SHALL track Runtime-owned session resources.

Resources MAY include tokenizer state, streaming decode state, output token
buffers, temporary generation buffers, memory reservations, model residency
references, and future KV cache placeholders.

#### Scenario: Close session

Given a session owns temporary generation buffers

When the session closes

Then Runtime releases or transfers eligible resources according to policy.

---

### Requirement: Session Memory Budget

A session SHALL integrate with Memory Manager for session-scoped memory budget
and usage.

#### Scenario: Budget exceeded

Given a generation operation would exceed session memory budget

When Runtime evaluates admission

Then the operation is rejected, queued, or delayed according to policy.

---

### Requirement: Session Concurrency Policy

A session SHALL define concurrency behavior.

Policies MAY include single-active-operation, allow-parallel-operations,
queue-operations, and reject-while-active.

#### Scenario: Reject while active

Given a session policy is reject-while-active

And one operation is active

When another operation is submitted

Then Runtime rejects it with concurrency-violation.

---

### Requirement: One-Shot Inference Uses Session Semantics

Runtime SHALL support one-shot inference without an explicit persistent session when one-shot generation is enabled by policy.

One-shot inference SHALL still follow session validation, memory, policy, and
cleanup semantics.

#### Scenario: One-shot generation

Given a caller requests one-shot generation

When Runtime executes it

Then Runtime uses an implicit short-lived session and cleans it up after
completion.

---

### Requirement: Session Streaming State

A session SHALL manage streaming state for streaming operations.

Streaming state MAY include token order, tokenizer streaming decode state,
backpressure, cancellation state, finish reason, and partial decode state.

#### Scenario: Streaming decode partial

Given tokenizer streaming decode has pending partial text

When generation continues

Then the session preserves the decode state until it can be flushed or closed.

---

### Requirement: Session Cancellation

A session SHALL support cancellation according to policy.

Cancellation may target current operation, queued operations, or entire session.

#### Scenario: Cancel session

Given a session has an active generation operation

When cancellation is requested

Then Runtime coordinates cancellation with Generation, Scheduler, Provider
execution, Memory Manager, and Tokenizer decode.

---

### Requirement: Session Drain

A draining session SHALL reject new operations while allowing current work to
finish according to policy.

#### Scenario: Drain session

Given Runtime marks a session draining

When a new generation request is submitted

Then Runtime rejects it with session-draining.

---

### Requirement: Session Expiration

A session SHALL be able to expire due to idle TTL, total TTL, model unload, memory pressure, Runtime shutdown, or policy.

#### Scenario: Idle TTL

Given a session is idle beyond its idle TTL

When Runtime evaluates expiration

Then Runtime expires and cleans up the session.

---

### Requirement: Session Status

Runtime SHALL expose session status without exposing raw prompts or raw native
handles by default.

#### Scenario: Inspect status

Given a caller inspects session status

When Runtime returns status

Then it includes lifecycle and usage metadata

And excludes raw Provider handles, Device handles, memory pointers, and raw
prompt text by default.

---

### Requirement: Session ID Is Not Authority

Possessing a session ID SHALL NOT by itself grant access to session resources.

#### Scenario: Unauthorized access

Given a caller knows a valid session ID

But lacks authorization

When it requests session status or operations

Then Runtime denies access.

---

### Requirement: Session Resource Affinity Is Runtime-Derived

Session Resource Affinity SHALL be derived from Runtime-owned resources.

Clients SHALL NOT forge affinity by passing session metadata.

#### Scenario: Forged affinity

Given a client attempts to force a session onto Provider A through arbitrary
session metadata

When Runtime validates the request

Then the metadata is rejected or ignored as non-authoritative.

---

### Requirement: Session Model Residency Reference

A session SHALL be able to reference model residency tracked by Memory Manager.

It SHALL NOT expose raw model memory handles.

#### Scenario: Model resident on Device

Given a session uses a resident model on a Device

When status is inspected

Then Runtime may report stable residency metadata

And not raw device pointers.

---

### Requirement: Browser-Compatible Session Model

The Inference Session model SHALL be platform-neutral.

Browser sessions SHALL not require Wasmtime or native Provider loading.

#### Scenario: Browser target

Given Runtime is built for a browser target

When session behavior is available

Then it uses browser-compatible Component Engine, Memory Manager, and Provider
capability paths.

---

### Requirement: Session Error Categories

Session failures SHALL use structured error categories.

#### Scenario: Closed session

Given a session is closed

When a caller submits generation work to it

Then Runtime returns session-closed.

---

### Requirement: Session Observability

Runtime SHALL define structured observations for session lifecycle, operations, memory pressure, cancellation, expiration, cleanup, and policy rejection.

Observability SHALL not log raw prompts by default.

#### Scenario: Session created

Given a session is created

When observability records the event

Then it may include redacted session metadata and lifecycle state.

### Requirement: Session May Own KV Cache

An Inference Session SHALL define how it may own or reference KV cache resources according to
session policy.

#### Scenario: Session cache

Given session policy enables KV cache

When generation prefill completes

Then the session may reference the created KV cache.

---

### Requirement: Session KV Cache Policy

A session SHALL define or reference policy for KV cache usage, budget, reuse,
sharing, persistence, and eviction.

#### Scenario: Cache budget exceeded

Given a session KV cache budget is exceeded

When generation attempts to append cache state

Then Runtime rejects, evicts, rebuilds, or fails according to policy.

---

### Requirement: Session Close Handles KV Cache

When a session closes, session-owned KV cache resources SHALL be released,
evicted, retained, or transferred to Runtime cache according to policy.

#### Scenario: Close with cache

Given a session owns a KV cache

When the session closes

Then Runtime applies session KV cache cleanup policy.

---

### Requirement: Session Status Redacts KV Cache

Session status SHALL not expose raw KV cache contents.

#### Scenario: Inspect session cache status

Given a session has KV cache state

When status is inspected

Then Runtime may report cache metadata such as size or lifecycle

And not raw key/value tensors.

### Requirement: Session May Require Loaded Model Context

An Inference Session SHALL require an existing loaded model context when policy disables implicit loading.

If implicit loading is disabled, session creation SHALL fail when the model is
not loaded.

#### Scenario: Session with unloaded model

Given session creation references a valid but unloaded Model Artifact

And implicit loading is disabled

When Runtime creates the session

Then creation fails with model-unavailable.

---

### Requirement: Session May Trigger Policy-Controlled Loading

Runtime SHALL allow session creation to trigger implicit model loading only when policy
explicitly permits it.

#### Scenario: Implicit load allowed

Given session creation references an unloaded but valid Model Artifact

And policy permits implicit loading

When session creation runs

Then Runtime performs Model Loading before the session becomes ready.

---

### Requirement: Session Close Does Not Imply Model Unload

Closing a session SHALL not automatically unload a model unless model residency
policy requires it.

#### Scenario: Close session

Given a session references a loaded model context

When the session closes

Then the model may remain resident according to Runtime cache policy.

### Requirement: Session Prefix Cache Policy

An Inference Session SHALL define or reference Prefix Cache policy.

Policy MAY enable, disable, scope, limit, share, retain, or evict prefix cache
entries.

#### Scenario: Prefix cache disabled

Given session policy disables prefix cache

When generation runs inside the session

Then Runtime skips Prefix Cache lookup.

---

### Requirement: Session Close Applies Prefix Cache Policy

When a session closes, session-scoped Prefix Cache entries SHALL be released,
retained, or transferred according to policy.

#### Scenario: Close session

Given a session owns session-scoped prefix entries

When the session closes

Then Runtime applies the configured cleanup policy.

