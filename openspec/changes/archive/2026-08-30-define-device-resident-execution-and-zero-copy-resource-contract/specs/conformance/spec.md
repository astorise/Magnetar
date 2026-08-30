## ADDED Requirements
### Requirement: Same Device Residency Conformance

Conformance SHALL prove consecutive compatible Kernels can consume Device
Resource without host round-trip.

#### Scenario: MatMul then RMSNorm

Given both execute on same Device

When pipeline runs

Then intermediate is not required to materialize on host.

### Requirement: Persistent Weight Residency Conformance

Conformance SHALL prove Model weights may remain Device-resident across
inference requests.

#### Scenario: Two requests

Given same Model Instance remains loaded

When second request begins

Then normal execution does not require re-uploading unchanged weights.

### Requirement: Device Resident KV Conformance

Conformance SHALL prove incremental decode can retain KV on Device.

#### Scenario: Multiple decode steps

Given GPU memory capacity remains sufficient

When tokens are generated

Then KV remains Device-resident.

### Requirement: Zero Copy Compatibility Conformance

Conformance SHALL prove zero-copy cannot bypass dtype/layout/alignment
requirements.

#### Scenario: Wrong layout

Given Resource is directly addressable

But Kernel layout requirement differs

When candidate is bound

Then direct access is rejected/materialized explicitly.

### Requirement: Resource View Conformance

Conformance SHALL prove compatible View creation performs no byte-copy.

#### Scenario: Slice View

Given View is created

When allocation accounting is inspected

Then no mandatory new payload allocation exists.

### Requirement: View Bounds Conformance

Conformance SHALL reject invalid or overflowing View ranges.

#### Scenario: Overflow

Given malicious offset/size

When View is created

Then operation fails before native access.

### Requirement: Aliasing Conformance

Conformance SHALL prove overlapping Views preserve dependency and lifetime
safety.

#### Scenario: View write races parent read

Given overlap exists

When operations are submitted asynchronously

Then hazard is detected/ordered.

### Requirement: Mapping Readiness Conformance

Conformance SHALL prove host mapping cannot read incomplete Device writes.

#### Scenario: GPU write pending

Given host requests read mapping

When completion is pending

Then mapping is not made readable prematurely.

### Requirement: Mapping Lifetime Conformance

Conformance SHALL prove mapped Resource cannot be physically evicted/reused.

#### Scenario: Active host mapping

Given pressure requests eviction

When mapping remains active

Then storage remains valid or mapping transition is safely coordinated.

### Requirement: Coherency Conformance

Conformance SHALL prove required visibility transitions occur for non-coherent
mapping.

#### Scenario: Host writes then GPU reads

Given mapping is non-coherent

When mapping ends

Then Provider establishes required Device-visible state before execution.

### Requirement: Native Pointer Isolation Conformance

Conformance SHALL prove native addresses are absent from portable/public
Resource contracts.

#### Scenario: CUDA Tensor

Given Device pointer exists internally

When Runtime/WIT/diagnostics are inspected

Then native pointer is absent.

### Requirement: Explicit Movement Conformance

Conformance SHALL prove residency-changing copies are represented explicitly.

#### Scenario: GPU to CPU

Given no shared mapping exists

When CPU consumes Tensor

Then explicit movement operation exists.

### Requirement: Host Staging Denial Conformance

Conformance SHALL prove hidden host staging cannot bypass policy.

#### Scenario: GPU-to-GPU fallback

Given Provider needs host staging

And policy forbids it

When movement is requested

Then operation fails or alternate path is selected.

### Requirement: Async Transfer Lifetime Conformance

Conformance SHALL prove source/destination storage remains valid around
asynchronous transfer.

#### Scenario: Source released early

Given transfer pending

When owner drops source Resource

Then underlying storage remains until completion.

### Requirement: Peer Capability Conformance

Conformance SHALL prove peer access is not inferred from Device similarity.

#### Scenario: Same vendor GPUs without peer capability

Given direct peer read unsupported

When Runtime plans access

Then zero-copy peer route is denied.

### Requirement: Cross Provider Zero Copy Conformance

Conformance SHALL prove direct shared Resource access across Providers requires
explicit interoperability capability.

#### Scenario: CUDA Provider to CPU Provider

Given no interop capability exists

When direct native memory access is requested

Then Runtime uses explicit movement or rejects it.

### Requirement: Memory Manager Authority Conformance

Conformance SHALL prove Provider cannot silently relocate/spill logical Resource
against Runtime policy where such movement is Runtime-visible.

#### Scenario: Host staging forbidden

Given Provider faces memory pressure

When spill would require host staging

Then it cannot silently violate policy.

### Requirement: Prepared Plan Native Memory Isolation Conformance

Conformance SHALL prove Prepared Plan stores logical Resource bindings rather
than native memory addresses.

#### Scenario: Plan cached

Given Resource native pointer changes after restart

When Plan metadata is restored

Then native pointer was not persisted.

### Requirement: Eviction Safety Conformance

Conformance SHALL prove in-flight Resource cannot be evicted unsafely.

#### Scenario: Pending Kernel

Given Device work references Tensor

When eviction is requested

Then physical storage remains until completion/quiescence.

### Requirement: Observability Redaction Conformance

Conformance SHALL prove residency traces contain no Device pointers, mapped host
addresses, file descriptors, external-memory handles, model data, KV content,
prompts, secrets, or credentials.

#### Scenario: Zero-copy trace

Given shared mapping is used

When trace is exported

Then only logical Resource/memory-domain metadata is present.