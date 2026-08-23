# Define Inference Session Model

## Why

Magnetar is an inference Runtime.

Recent contracts define:

- Model Artifacts
- Tokenizer behavior
- Generation behavior
- Memory Manager
- Providers
- Devices
- Component Runtime
- inference-scoped Component authority

The next missing foundation is the Inference Session.

A single generation request can be stateless, but real inference workloads need
a Runtime-owned context that can hold validated bindings and reusable state.

Examples include:

- selected model reference
- tokenizer reference
- generation defaults
- runtime policy
- request limits
- memory admission state
- model residency reference
- future KV cache reference
- future prefix cache reference
- streaming decode state
- cancellation state
- observability correlation
- lifecycle state
- resource ownership

Without a session model, later features such as KV cache, prefix reuse,
multi-turn chat, cancellation, streaming, batching, and model residency will
become tangled with Generation or Scheduler.

This change defines the Inference Session model.

## What Changes

This change introduces `InferenceSession` as a first-class Runtime-owned
inference context.

An Inference Session SHALL represent a bounded Runtime context for one or more
inference operations.

It SHALL bind together:

- model reference or future Model Instance reference
- tokenizer reference
- generation defaults
- Runtime policy
- memory limits
- session-scoped resources
- cancellation state
- streaming state
- observability correlation
- future KV cache and prefix cache ownership

The exact Rust type names are implementation-defined.

## Session Is Runtime-Owned

Inference Sessions SHALL be owned by the Runtime.

Clients may request session creation, use, cancellation, or destruction.

Clients SHALL NOT forge session identity.

Components SHALL NOT create arbitrary sessions outside authorized Runtime
contracts.

Session identity SHALL be opaque and Runtime-issued.

## Session Is Not Client Conversation

An Inference Session SHALL NOT be treated as a full client conversation or
agent state.

Client-side conversation history, files, tools, Git state, workspace state,
network state, and secrets remain outside Magnetar.

Magnetar sessions may hold inference state only.

For example:

```text
client conversation
    - messages
    - UI state
    - workspace files
    - tools
    - secrets
    - Git state

Magnetar Inference Session
    - tokenized prompt context
    - model/tokenizer binding
    - generation state
    - KV cache placeholder
    - memory residency
    - cancellation
    - inference observability
```

## Session Is Not Model Artifact

A Model Artifact is validated model data.

An Inference Session may reference a loaded model context or future Model
Instance derived from a Model Artifact.

Validating a Model Artifact SHALL NOT create an Inference Session.

Creating a session SHALL require explicit Runtime action.

## Session Is Not KV Cache

A session may own or reference future KV cache state.

The KV cache contract is defined later.

This change defines session ownership and lifecycle boundaries.

It does not define KV cache layout, eviction, paging, reuse, or prefix matching.

## Session Lifecycle

An Inference Session SHALL have a lifecycle.

Initial lifecycle states SHOULD include:

```text
creating
ready
active
idle
draining
cancelled
failed
closed
expired
```

Semantics:

- `creating`: Runtime is validating and allocating session resources
- `ready`: session can accept inference operations
- `active`: an operation is currently running
- `idle`: no operation is running but reusable state may exist
- `draining`: session refuses new operations while finishing existing work
- `cancelled`: cancellation requested or completed
- `failed`: session cannot continue due to error
- `closed`: resources released
- `expired`: TTL or policy expiration closed the session

The exact serialized names are implementation-defined.

## Session Creation

Session creation SHALL validate:

- model availability
- tokenizer compatibility
- generation defaults
- Runtime policy
- memory limits
- allowed capabilities
- session TTL
- concurrency policy
- streaming policy
- cancellation policy

A session SHALL NOT become ready until validation succeeds.

## Session Binding

A session SHALL bind to a model reference and tokenizer reference.

The model reference may eventually point to a loaded Model Instance.

The tokenizer reference SHALL be compatible with the model.

Generation requests executed inside the session SHALL use the session binding
unless explicitly overridden by policy.

## Session Policy

A session SHALL carry or reference Runtime policy.

Policy may define:

- maximum prompt tokens
- maximum generated tokens
- maximum total tokens
- allowed generation parameters
- allowed sampling modes
- streaming allowed
- cancellation allowed
- concurrency allowed
- memory budget
- KV cache budget placeholder
- prefix cache allowed placeholder
- observability redaction
- raw prompt logging allowed or denied
- timeout
- idle TTL
- total session TTL

Session policy SHALL not grant filesystem, network, Git, secrets, workspace, or
process authority.

## Session Resources

A session may own Runtime resources.

Resources MAY include:

- tokenizer state
- streaming decode state
- output token buffers
- temporary generation buffers
- memory reservations
- future KV cache resources
- future prefix cache resources
- model residency references
- observability correlation state

Resources SHALL be released when the session closes unless policy preserves
them in a Runtime cache.

## Session Memory

Inference Sessions SHALL integrate with the Memory Manager.

Memory Manager SHALL track session-scoped allocations and budgets.

Session memory may include:

- input token buffers
- output token buffers
- logits buffers
- sampling workspace
- tokenizer streaming state
- future KV cache
- future prefix cache
- temporary workspace
- model residency references

A session SHALL not allocate unbounded memory.

## Session Concurrency

A session SHALL define concurrency behavior.

Possible policies include:

```text
single-active-operation
allow-parallel-operations
queue-operations
reject-while-active
```

Default policy SHOULD be conservative.

For stateful sessions with future KV cache, default policy SHOULD avoid parallel
mutations unless explicitly supported.

## Session Operations

A session may support operations such as:

- generate
- stream-generate
- cancel
- drain
- close
- inspect status
- reset transient state
- future: reuse prefix
- future: fork session
- future: snapshot session

This change defines session boundary and lifecycle.

It does not require all future operations immediately.

## Session And Generation

Generation may run inside a session.

When generation runs inside a session, it SHALL use:

- session model binding
- session tokenizer binding
- session policy
- session memory budget
- session cancellation state
- session observability correlation
- future session KV cache

A GenerationRequest may either reference a session or be executed as an
anonymous one-shot request according to Runtime policy.

## One-Shot Inference

Runtime MAY support one-shot generation without an explicit persistent session.

Conceptually, one-shot generation creates an implicit short-lived session.

That session still follows session validation, memory, policy, and cleanup
semantics.

## Session And Streaming

A session SHALL support streaming state when streaming is enabled.

Streaming state MAY include:

- generated token order
- tokenizer streaming decode state
- consumer backpressure state
- cancellation state
- finish reason
- partial decode state
- output accounting

Streaming state SHALL be cleaned up when operation or session ends.

## Session Cancellation

A session SHALL support cancellation according to policy.

Cancellation may apply to:

- current operation
- queued operations
- entire session

Cancellation SHALL produce structured results.

Cancellation SHALL coordinate with:

- Generation
- Scheduler
- Provider execution
- Memory Manager
- Tokenizer streaming decode

## Session Drain

A session may enter draining state.

A draining session SHALL reject new operations.

It may allow current operation to finish according to policy.

Draining is useful for shutdown, timeout, model unload, memory pressure, or
client close.

## Session Expiration

A session MAY expire due to policy.

Expiration may be based on:

- idle TTL
- total TTL
- memory pressure
- model unload
- Runtime shutdown
- client-specified timeout
- administrative policy

Expired sessions SHALL release resources or move eligible resources to
Runtime-managed caches according to policy.

## Session Status

Runtime SHALL expose session status.

Status SHOULD include:

- session ID
- lifecycle state
- model reference
- tokenizer reference
- active operation count
- queued operation count
- memory usage summary
- future KV cache usage placeholder
- streaming state summary
- cancellation state
- last error
- created timestamp
- last activity timestamp
- expiration metadata

Status SHALL not expose raw prompts by default.

Status SHALL not expose raw Provider, Device, or memory handles.

## Session Identity

Session identity SHALL be opaque.

A session ID may be serializable for clients, but it SHALL not encode raw
memory, Provider, Device, or pointer information.

A session ID SHALL not grant authority by itself.

Runtime policy must still authorize session access.

## Session Authority

Session access SHALL be authorized by Runtime policy.

A Component or client that has a session ID does not automatically gain access
to model artifacts, tokenizer artifacts, raw prompts, KV cache contents, memory
handles, Provider handles, or Device handles.

Session authority remains inference-scoped.

## Session And Provider Resolution

A session SHALL NOT directly select Provider or Device from user input.

Runtime Resolution determines Provider and Device placement for operations.

A session may carry Resource Affinity derived from existing Runtime-owned
resources.

Session Resource Affinity SHALL be Runtime-owned and not forgeable by clients.

## Session And Model Residency

A session may reference model residency.

Model residency is tracked by Memory Manager.

The session does not own raw model memory handles.

When a session closes, model residency MAY remain cached according to Runtime
policy.

## Session And Browser Target

The Inference Session model SHALL be platform-neutral.

Browser targets may support a subset of session behavior depending on available
Component Engine, Memory Manager, and Provider capabilities.

Unsupported session features SHALL produce structured errors.

Browser sessions SHALL not require Wasmtime or native Provider loading.

## Error Model

Session errors SHALL be structured.

Error categories SHOULD include:

- session creation failed
- session not found
- session not ready
- session active
- session closed
- session expired
- session cancelled
- session draining
- session policy denied
- model unavailable
- tokenizer incompatible
- memory admission failed
- memory budget exceeded
- generation failed
- streaming failed
- cancellation failed
- operation queued
- operation rejected
- concurrency violation
- resource cleanup failed
- runtime shutdown
- internal session error

## Observability

Runtime SHOULD emit observations for:

- session create requested
- session created
- session creation failed
- session ready
- session active
- session idle
- session draining
- session cancelled
- session closed
- session expired
- session operation started
- session operation completed
- session operation failed
- session memory pressure
- session cleanup
- session policy rejection

Observability SHALL not log raw prompts by default.

Session identifiers in observability SHALL be redacted or policy-controlled
where needed.

## Non-Goals

This change does not:

- define full KV cache model
- define prefix cache model
- define batching scheduler
- define model loading lifecycle fully
- define Model Instance lifecycle fully
- define chat conversation storage
- define agent memory
- define client workspace state
- define filesystem access
- define Git access
- define network access
- define secrets access
- define remote session protocol
- define Tachyon distributed sessions
- require browser implementation
- require GPU hardware
- define persistence of sessions across Runtime restarts

## Impact

Magnetar gains a stable session boundary.

Future features can attach to the session without polluting Generation,
Tokenizer, Scheduler, or Provider abstractions.

The inference pipeline becomes:

```text
Model Artifact / future Model Instance
        |
        v
Inference Session
        |
        +-- tokenizer binding
        +-- generation policy
        +-- memory budget
        +-- cancellation state
        +-- streaming state
        +-- future KV cache
        |
        v
Generation operations
```

This prepares later changes:

- KV cache model
- model loading contract
- sampling and logits processing contract
- continuous batching contract