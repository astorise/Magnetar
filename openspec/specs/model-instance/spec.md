# model-instance Specification

## Purpose
This specification defines Model Instance identity, lifecycle, readiness, warmup, adapter/cache interaction, sharing policy, and runtime-owned usage.
## Requirements
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

Readiness-relevant facts the Runtime can itself observe -- including whether
mandatory weight resources are bound, whether a pinned Provider actually
resolves and offers an execution API, and whether a pinned Device is
available -- SHALL be derived from actual Runtime state, not accepted
outright from a caller-supplied claim. A caller MAY assert a stricter
(`false`) value than Runtime state alone would produce; a caller SHALL NOT
be able to assert a Runtime-observable fact as `true` when the Runtime does
not itself observe it as true.

A bound weight resource SHALL only count toward `weights_materialized` if
it has a corresponding residency record the Runtime itself recorded; a
resource identifier present without one SHALL NOT count. A pinned
Provider SHALL only count toward `provider_ready` if its own status model
reports it as currently accepting new work, not merely that it is
registered and exposes an execution interface in principle.

The public surface for producing a `Ready` Model Instance SHALL NOT permit
an external caller to reach `Ready` other than through a path that
performs this derivation; lifecycle and readiness state SHALL NOT be
directly settable by an external caller.

#### Scenario: Provider not ready

Given an instance lifecycle exists

But Provider is not ready

When Runtime checks readiness

Then the instance is not ready for generation.

#### Scenario: Caller cannot forge a Runtime-observable fact

Given a Model Instance has no weight resources bound

When a caller requests warmup asserting weights are materialized

Then the Runtime's own observation of empty resource bindings overrides the caller's claim and the instance does not become Ready.

#### Scenario: A bound weight without a residency record does not count

Given a Model Instance has a weight resource identifier bound with no corresponding residency record

When a caller requests warmup asserting weights are materialized

Then the Runtime does not treat that resource as materialized and the instance does not become Ready.

#### Scenario: A Provider that rejects new work does not count as ready

Given a Model Instance is pinned to a Provider that is registered and exposes an execution interface

But that Provider's own status model reports it does not currently accept new work

When a caller requests warmup asserting the Provider is ready

Then the Runtime does not treat the Provider as ready and the instance does not become Ready.

---

### Requirement: Model Instance Creation

Model Instance creation SHALL require successful Model Loading or an explicit policy-controlled loading path, and creation SHALL NOT by itself produce a Ready Model Instance.

A newly created Model Instance SHALL be left in a non-Ready lifecycle state (`Loading`). Transition to Ready SHALL happen only through a separate, explicit step performed after every mandatory readiness condition for that instance -- including, where applicable, weight materialization -- has actually been satisfied.

#### Scenario: Artifact only

Given a valid Model Artifact is not loaded

When a caller requests a Model Instance without implicit loading policy

Then creation fails.

#### Scenario: Creation alone does not imply readiness

Given a Model Instance has just been created from a successfully loaded artifact

When no explicit readiness-completing step has run yet for it

Then the instance reports a non-Ready lifecycle and readiness state, and any check that inspects only that state (not a deeper, resource-specific check) correctly rejects it as not usable

### Requirement: Model Instance Warmup

Model Instance warmup MAY be supported and SHALL be policy-controlled.

Warmup failure SHALL prevent ready state.

Regardless of warmup policy, `readiness` SHALL NOT report `Ready` while
`lifecycle` has not itself reached a state that supports inference use. A
warmup policy that does not perform lifecycle transitions SHALL NOT be able
to publish `Ready` readiness as a side effect.

The primitives capable of transitioning a Model Instance to `Ready`
(the underlying lifecycle transition and the raw ready-marking operation)
SHALL NOT be reachable by a caller outside the Runtime's own
implementation. An external caller SHALL only be able to request warmup
through the Runtime-owned entry point that performs readiness derivation
first.

#### Scenario: Warmup failure

Given Provider warmup fails

When Runtime evaluates instance readiness

Then the instance becomes failed or not-ready according to policy.

#### Scenario: Disabled policy cannot forge readiness

Given a Model Instance is in a lifecycle state that does not support inference use

When warmup is invoked with a policy that does not transition the lifecycle

Then readiness does not report Ready even if the supplied checks would otherwise compute Ready

#### Scenario: The raw ready-marking primitive is not externally reachable

Given a caller external to the Runtime's own implementation holds a mutable reference to a Model Instance

When that caller attempts to invoke the underlying lifecycle transition or ready-marking operation directly, bypassing the Runtime-owned warmup entry point

Then no such path is available to that caller

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

Accepting usage (acquiring a generation reference or a usage handle) SHALL
require both that the lifecycle is in a state that supports inference use
and that readiness reports Ready. An internally inconsistent combination
(readiness reporting Ready while lifecycle does not support inference use)
SHALL be rejected, regardless of how that inconsistency arose.

#### Scenario: Generate on failed instance

Given a Model Instance is failed

When generation is requested

Then Runtime rejects generation.

#### Scenario: Inconsistent lifecycle and readiness reject usage

Given a Model Instance reports readiness Ready but its lifecycle has not reached a state that supports inference use

When a caller attempts to acquire usage or a generation reference

Then Runtime rejects the request based on the lifecycle, not only the readiness value

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

### Requirement: Model Instance May Produce Execution Graphs

A Model Instance or its architecture implementation SHALL be able to produce Execution Graphs
for inference phases.

#### Scenario: Decode graph from instance

Given a Model Instance is ready

When decode begins

Then Runtime may obtain a decode Execution Graph compatible with that instance.

---

### Requirement: Graph Identity Depends On Instance Semantics

Execution Graph identity SHALL reflect Model Instance semantic state where
relevant.

#### Scenario: Adapter merge changes semantics

Given a Model Instance changes due to adapter merge

When a graph is built

Then graph identity reflects the changed semantic state.

### Requirement: Model Instance References Architecture Implementation

A Model Instance SHALL be able to reference the Model Component or Runtime-native
architecture implementation used to create it.

#### Scenario: Instance compatibility

Given a Model Instance was created with Model Component C

When cache compatibility is evaluated

Then C's identity and version may be considered.

---

### Requirement: Model Instance Does Not Grant Component Authority

Referencing a Model Component from a Model Instance SHALL not grant additional
authority to the Component.

#### Scenario: Instance references component

Given Model Instance references Component C

When C requests network access

Then Runtime still denies forbidden authority.

---

### Requirement: Qwen Model Instance References Qwen Component

A Qwen Model Instance SHALL reference the Qwen Component or native architecture
implementation used to validate and execute it.

#### Scenario: Qwen instance metadata

Given Qwen Model Instance is ready

When Runtime reports metadata

Then Qwen Component identity may be included.

---

### Requirement: Qwen Component Metadata Participates In Cache Compatibility

Qwen Component version and config fingerprint SHALL be permitted to participate
in KV Cache and Prefix Cache compatibility.

#### Scenario: Component version changed

Given Prefix Cache entry was produced under Qwen Component version A

When version B changes graph semantics

Then Runtime rejects reuse unless compatibility is proven.

---

### Requirement: Model Instance Operations Are Exposed Safely

Runtime Inference API SHALL expose Model Instance lifecycle operations such as create, inspect, warm, suspend, resume, drain, and unload.

#### Scenario: Inspect instance

Given caller inspects Model Instance

When Runtime returns metadata

Then it does not expose Provider handles, Device handles, Kernel handles, or raw
tensor pointers.

---

### Requirement: Inference API Respects Active Instance Use

Model Instance lifecycle operations through Runtime Inference API SHALL respect active sessions and generations.

#### Scenario: Unload active instance

Given Model Instance has active generation

When unload is requested

Then Runtime drains, rejects, waits, or forces according to policy.

---

### Requirement: CLI Model Instance Operations Use Runtime API

`magnetar-cli` SHALL inspect, warm, suspend, resume, drain, or unload Model
Instances through Runtime Inference API.

#### Scenario: CLI unload

Given user runs model unload

When CLI performs unload

Then Runtime validates active usage and policy before unloading.

---

### Requirement: CLI Does Not Access Instance Internals

CLI SHALL not access raw Model Instance internals such as Provider handles,
Device handles, Kernel handles, Tensor pointers, or raw weights.

#### Scenario: Instance inspect

Given CLI inspects instance

When metadata is displayed

Then only redacted Runtime metadata is shown.

---

### Requirement: E2E Uses Model Instance Lifecycle

E2E conformance SHALL create and use a Runtime-owned Model Instance.

#### Scenario: Instance ready

Given fixture model loads successfully

When Model Instance is published

Then it reaches ready state before session creation.

---

### Requirement: E2E Validates Instance Cleanup

E2E conformance SHALL validate Model Instance and related resources are cleaned
up or retained only according to policy.

#### Scenario: Unload after test

Given E2E test completes

When cleanup runs

Then Model Instance lifecycle follows policy and does not leak active resources.

### Requirement: Model Instance Release Gate

Model Instance lifecycle SHALL have release gate coverage.

#### Scenario: Active unload leak

Given active Model Instance unload leaks resources

When release validation runs

Then stable release is blocked.

---

### Requirement: Model Instance May Have Autotuning Policy

Model Instance MAY be configured with disabled, optional, required, or pinned Kernel Autotuning behavior, and Runtime SHALL enforce exactly one active policy at a time.

#### Scenario: Required warmup tuning

Given deployment requires tuning before readiness

When Model Instance loads

Then it remains warming until required tuning completes or fails according to
policy.

---

### Requirement: Optional Autotuning Does Not Necessarily Block Readiness

Optional tuning MAY occur after Model Instance is usable with known-good Kernels, and optional tuning SHALL NOT block Model Instance readiness.

#### Scenario: Background tuning

Given baseline Kernel is ready

When optional tuning begins

Then Model Instance may remain inference-ready.

---

### Requirement: Reproducible Model Instance May Pin Tuning Record

Model Instance MAY pin a known Autotuning Record, and a pinned Autotuning Record SHALL NOT change without explicit reconfiguration.

#### Scenario: Reproducible deployment

Given pinned record remains compatible

When Model Instance executes

Then Runtime uses its authorized specialization policy without live retuning.

### Requirement: Model Instance Owns Plan Family Context

Model Instance SHALL be able to maintain one or more Prepared Execution Plans for compatible
workloads.

#### Scenario: Prefill and decode

Given Model Instance supports both phases

When prepared

Then it may hold distinct ready Plan families for prefill and decode.

---

### Requirement: Model Revision Participates In Plan Validity

Prepared Execution Plan SHALL not silently survive incompatible Model Instance
revision.

#### Scenario: Adapter set changes

Given adapter configuration modifies graph/resources

When revision changes

Then dependent Plan is stale or invalid according to compatibility.

---

### Requirement: Model Instance Readiness May Depend On Required Plans

Deployment policy SHALL be able to require mandatory Prepared Execution Plans before Model
Instance becomes ready.

#### Scenario: Strict low-latency deployment

Given required decode Plan is not prepared

When readiness is evaluated

Then Model Instance remains warming/not-ready according to policy.

---

### Requirement: Optional Plan Preparation Need Not Block Readiness

Additional workload-specific Plans SHALL be able to be prepared lazily while baseline
known-good Plan remains available.

#### Scenario: Long-sequence Plan absent

Given normal sequence Plan is ready

When Model Instance starts

Then optional 32k sequence Plan may build later.

### Requirement: Model Instance May Use Multiple Devices

One Model Instance SHALL be able to own execution state across multiple local Devices.

#### Scenario: Large model

Given weights cannot fit on one GPU

When Model Instance loads

Then Runtime may create multi-Device placement and memory state.

### Requirement: Model Instance Tracks Placement Generation

Model Instance SHALL identify active MultiDevicePlacementPlan generation.

#### Scenario: Re-placement

Given replacement Plan becomes active

When diagnostics inspect Model Instance

Then new placement generation is distinguishable.

### Requirement: Model Readiness Requires Mandatory Devices

If placement requires multiple Devices, Model Instance SHALL not become READY
until all mandatory placement dependencies are ready.

#### Scenario: GPU1 kernels unprepared

Given GPU0 is ready but GPU1 mandatory stage is not

When readiness is evaluated

Then Model Instance is not fully ready under that Plan.

### Requirement: Model Instance May Own Degraded Plans

Model Instance SHALL be able to retain pre-built fallback placement plans.

#### Scenario: GPU1 failure

Given validated GPU0-only degraded Plan exists

When GPU1 fails

Then Runtime may activate degraded Plan according to policy.

### Requirement: Model Revision Invalidates Placement As Needed

Changes to graph/resource requirements SHALL be able to invalidate current placement.

#### Scenario: Adapter revision increases memory demand

Given new adapter no longer fits current Device allocation

When revision changes

Then placement is rebuilt or rejected.

### Requirement: ModelInstance Owns Executed Resource Bindings
An active ModelInstance SHALL expose the stable resource bindings for weights, constants, and adapters used by its prepared graph execution.

#### Scenario: Two instances are loaded
- **WHEN** two ModelInstances for different artifacts execute
- **THEN** each execution uses only the resource bindings owned by its active instance.

#### Scenario: Instance unloads
- **WHEN** a ModelInstance is unloaded
- **THEN** its resource bindings are released according to Runtime policy and cannot be used for new execution.

