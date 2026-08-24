## ADDED Requirements

### Requirement: Model Instance

Magnetar SHALL define Model Instance as a Runtime-owned loaded inference
context.

A Model Instance SHALL represent a loaded model that may be used for inference
when ready.

#### Scenario: Loaded model becomes instance

Given Model Loading succeeds

When Runtime publishes the loaded context

Then a Model Instance may be created or marked ready.

---

### Requirement: Model Instance Is Runtime-Owned

Runtime SHALL own Model Instance identity, lookup, lifecycle, readiness,
authorization, and cleanup.

Clients and Components SHALL NOT forge Model Instance identity.

#### Scenario: Forged instance ID

Given a caller submits a fabricated ModelInstanceId

When Runtime resolves it

Then Runtime rejects it as not found or unauthorized.

---

### Requirement: Model Instance Is Not Model Artifact

A Model Instance SHALL not be treated as immutable Model Artifact data.

#### Scenario: Same artifact loaded twice

Given one Model Artifact is loaded once on CPU and once on GPU

When Runtime records instances

Then both instances reference the same artifact identity but have distinct
instance identities and residency.

---

### Requirement: Model Instance Is Not Model Residency

A Model Instance SHALL reference Model Residency but SHALL also include
lifecycle, readiness, architecture implementation, policy, and inference
coordination.

#### Scenario: Residency exists

Given model weights are resident in Device memory

When no ready Model Instance exists

Then generation cannot use the residency directly.

---

### Requirement: Model Instance Is Not Session

An Inference Session SHALL be able to reference a Model Instance, but the instance remains a
separate Runtime resource.

#### Scenario: Session closes

Given a session references a shared Model Instance

When the session closes

Then the Model Instance may remain loaded according to Runtime policy.

---

### Requirement: Model Instance Lifecycle

A Model Instance SHALL have lifecycle state.

States SHOULD include creating, loading, warming, ready, active, idle, draining,
suspended, reloading, unloading, unloaded, failed, invalid, and removed.

#### Scenario: Instance ready

Given loading and warmup succeed

When Runtime completes readiness checks

Then lifecycle may become ready.

---

### Requirement: Model Instance Readiness

Model Instance lifecycle and readiness SHALL be distinct.

Readiness SHALL consider residency, Provider readiness, Device readiness,
adapter state, memory pressure, Runtime policy, and architecture implementation
readiness.

#### Scenario: Provider not ready

Given an instance lifecycle exists

But Provider is not ready

When Runtime checks readiness

Then the instance is not ready for generation.

---

### Requirement: Model Instance Creation

Model Instance creation SHALL require successful Model Loading or an explicit
policy-controlled loading path.

#### Scenario: Artifact only

Given a valid Model Artifact is not loaded

When a caller requests a Model Instance without implicit loading policy

Then creation fails.

---

### Requirement: Model Instance Warmup

Model Instance warmup MAY be supported and SHALL be policy-controlled.

Warmup failure SHALL prevent ready state.

#### Scenario: Warmup failure

Given Provider warmup fails

When Runtime evaluates instance readiness

Then the instance becomes failed or not-ready according to policy.

---

### Requirement: Model Instance Usage Reference

Generation, Sessions, and Scheduler SHALL acquire Runtime-managed usage
references before using a Model Instance.

#### Scenario: Use instance

Given a ready Model Instance

When generation begins

Then Runtime acquires a usage reference before prefill.

---

### Requirement: Model Instance Unload Protection

Runtime SHALL prevent normal unload while active usage references exist unless
forced policy applies.

#### Scenario: Unload active instance

Given a Model Instance has active generation

When unload is requested

Then Runtime drains, rejects, waits, or forces according to policy.

---

### Requirement: Model Instance Usage Accounting

Runtime SHALL track Model Instance usage without exposing raw prompts, weights,
or handles.

#### Scenario: Inspect usage

Given a caller requests instance status

When Runtime returns usage

Then it includes stable counters and not raw model memory.

---

### Requirement: Model Instance Sharing

Model Instance sharing across sessions SHALL be policy-controlled.

#### Scenario: Unsafe sharing

Given two sessions have incompatible privacy or adapter policy

When they attempt to share an instance

Then Runtime denies sharing.

---

### Requirement: Model Instance Mutability Is Explicit

Semantic mutation of Model Instance state SHALL be explicit.

Silent mutation affecting inference semantics SHALL be forbidden.

#### Scenario: Adapter merge

Given adapter merge mutates base residency

When Runtime applies it

Then Runtime records mutation and invalidates dependent state according to
policy.

---

### Requirement: Model Instance Tracks Adapter State

A Model Instance SHALL track active adapter state where adapters affect
inference.

#### Scenario: Adapter active

Given adapter A is active

When Runtime reports Model Instance metadata

Then active adapter state is represented as redacted metadata.

---

### Requirement: Model Instance KV Cache Compatibility

KV cache compatibility SHALL include Model Instance identity or compatible
instance metadata.

#### Scenario: Instance reload

Given KV cache was created for instance version A

When instance reload creates version B

Then Runtime rejects or invalidates incompatible cache reuse.

---

### Requirement: Model Instance Prefix Cache Compatibility

Prefix Cache entries SHALL bind to Model Instance identity or compatible model
context metadata where needed.

#### Scenario: Prefix reuse after unload

Given a Prefix Cache entry depends on unloaded instance M

When reuse is attempted

Then Runtime rejects reuse.

---

### Requirement: Generation Requires Ready Model Instance

Generation SHALL require a ready Model Instance or a policy-controlled implicit
load path.

#### Scenario: Generate on failed instance

Given a Model Instance is failed

When generation is requested

Then Runtime rejects generation.

---

### Requirement: Batching Uses Model Instance Compatibility

Continuous Batching SHALL consider Model Instance compatibility.

#### Scenario: Different instances

Given two operations use incompatible Model Instances

When Scheduler forms a batch

Then they are not placed in the same execution step.

---

### Requirement: Provider State Affects Model Instance

Provider health, readiness, pressure, admission, and failure SHALL affect Model
Instance readiness and lifecycle.

#### Scenario: Provider failed

Given the Provider backing a Model Instance fails

When Runtime processes status

Then the instance becomes failed, invalid, suspended, or draining according to
policy.

---

### Requirement: Device State Affects Model Instance

Device readiness, loss, reset, pressure, or unavailability SHALL affect Model
Instance readiness and lifecycle.

#### Scenario: Device lost

Given a Model Instance residency is Device-bound

When the Device is lost

Then Runtime suspends, invalidates, reloads, or unloads the instance according
to policy.

---

### Requirement: Memory Manager Tracks Model Instance Residency

All Model Instance residency SHALL be tracked through Memory Manager.

#### Scenario: Instance residency

Given a Model Instance owns Device residency

When memory usage is reported

Then Memory Manager accounts for the residency.

---

### Requirement: Model Instance Suspension

Runtime SHALL define policy-controlled suspension for Model Instances.

Suspended instances SHALL not accept new inference operations.

#### Scenario: Suspend on pressure

Given memory pressure is high

When policy allows suspension

Then Runtime may suspend an idle Model Instance.

---

### Requirement: Model Instance Draining

A draining Model Instance SHALL reject new operations while allowing active
operations to complete according to policy.

#### Scenario: Drain for reload

Given reload is requested

When instance enters draining

Then new generation operations are rejected or routed elsewhere.

---

### Requirement: Model Instance Unload

Unload SHALL release or invalidate dependent runtime state according to policy.

Dependent state MAY include sessions, adapters, KV caches, Prefix Cache entries,
Memory Manager residency, and Provider resources.

#### Scenario: Unload instance

Given a Model Instance has dependent KV cache

When unload completes

Then dependent cache is invalidated or released.

---

### Requirement: Model Instance Reload

Reload SHALL be treated as a new validated loading process and SHALL not
silently mutate active inference semantics.

#### Scenario: Reload to BF16

Given an instance runs with FP16 compute

When reload requests BF16

Then Runtime creates a validated replacement or rejects reload.

---

### Requirement: Model Instance Failure Categories

Model Instance failures SHALL use structured error categories.

#### Scenario: Instance invalid

Given an instance is invalid

When generation attempts to use it

Then Runtime returns model-instance-invalid.

---

### Requirement: Browser-Compatible Model Instance Lifecycle

Model Instance lifecycle SHALL be platform-neutral and SHALL not require
Wasmtime or native Provider loading.

#### Scenario: Browser target

Given Runtime runs on browser target

When unsupported instance lifecycle feature is requested

Then Runtime returns model-instance-browser-feature-unsupported.

---

### Requirement: Model Instance Observability

Runtime SHALL define Model Instance observations for creation, loading, warming,
readiness, usage, draining, suspension, reload, unload, failure, invalidation,
sharing denial, memory pressure, Provider pressure, and Device unavailability.

Observability SHALL not expose raw model weights, raw prompts, raw handles, or
raw cache contents by default.

#### Scenario: Instance ready observation

Given a Model Instance becomes ready

When Runtime emits observability

Then it includes redacted instance metadata and readiness state.
