# device Specification

## Purpose
This specification defines Device identity, metadata, ownership, status, and Provider association for runtime execution targets.
## Requirements
### Requirement: Device Discovery

Providers SHALL discover available Devices.

#### Scenario: Runtime initialization

Given a Provider

When the Runtime initializes

Then every available Device is registered.

---

### Requirement: Device Identity

Every Device SHALL expose a globally unique identifier.

An identifier SHALL be unique across all Providers registered by a Runtime.

#### Scenario: Enumerate Devices

Given multiple Providers

When the Runtime lists Devices

Then every Device SHALL have a unique identifier.

---

### Requirement: Device Metadata

Every Device SHALL expose descriptive metadata.

Metadata SHALL include:

- name
- device type
- vendor
- architecture
- memory capacity
- compute-unit count
- execution capabilities

#### Scenario: Inspect device metadata

Given a registered Device

When the Runtime retrieves the Device metadata

Then every required metadata field is available.

---

### Requirement: Provider Ownership

Every Device SHALL belong to exactly one Provider.

#### Scenario: Device registration

Given a discovered Device

When registration completes

Then the Device references its owning Provider.

#### Scenario: Mismatched ownership

Given a Device that declares a different owning Provider

When a Provider attempts to register the Device

Then registration fails and the Device is not available from the Runtime.

---

### Requirement: Device Independence

Runtime scheduling SHALL operate on Devices.

Scheduling SHALL NOT depend on Provider implementations.

#### Scenario: Runtime device enumeration

Given registered Providers with discovered Devices

When the Runtime lists Devices

Then it returns the registered Devices without requiring access to a Provider
implementation.

### Requirement: Device Compatibility Constrains Kernels

Kernel execution SHALL validate Device compatibility.

#### Scenario: Kernel requires CUDA capability

Given a Kernel requires a CUDA-capable Device

When Runtime plans execution on CPU Device

Then Runtime rejects the Kernel.

---

### Requirement: Device State Affects Kernel Dispatch

Device readiness, memory pressure, loss, reset, or unavailability SHALL affect
Kernel dispatch eligibility.

#### Scenario: Device unavailable

Given a Device is unavailable

When Runtime considers a Device-bound Kernel

Then the Kernel is not eligible.

---

### Requirement: Device Metadata Supports Kernel Planning

Device metadata SHALL expose features needed for Kernel planning, such as
memory class support, dtype support, layout support, execution limits, and
hardware feature flags.

#### Scenario: Tensor core requirement

Given a Kernel requires tensor-core-like capability

When Runtime validates Device metadata

Then the Device must advertise compatible feature metadata.

### Requirement: Device State Affects Kernel Eligibility

Device readiness, availability, pressure, loss, reset, and memory state SHALL
affect Kernel Registry eligibility and Dispatch revalidation.

#### Scenario: Device lost

Given selected Kernel targets Device A

When Device A is lost before dispatch

Then Runtime rejects, replans, or falls back according to policy.

---

### Requirement: Device Metadata Feeds Kernel Selection

Device metadata SHALL feed Kernel candidate filtering and ranking.

#### Scenario: Required feature missing

Given Kernel requires a Device feature flag

When Device metadata lacks that flag

Then the Kernel is not eligible.

---

### Requirement: Device Pressure Affects Kernel Ranking

Device pressure SHALL be available to Kernel ranking and MAY affect fallback.

#### Scenario: Device pressure high

Given two compatible Devices exist

And Device A has high pressure

When ranking runs

Then Runtime may prefer Device B according to policy.

---

### Requirement: Device Remains Hardware Abstraction

Device SHALL remain a representation of hardware identity, capabilities,
availability, health, pressure, and execution limits.

#### Scenario: Kernel compilation needed

Given Kernel Source Artifact needs compilation

When architecture assigns responsibility

Then Device is not the compiler.

---

### Requirement: Device Does Not Own Kernel Source

Device SHALL not accept arbitrary kernel source through its public contract.

#### Scenario: Triton source

Given Triton source exists

When Provider prepares it

Then Device receives no source-management responsibility.

---

### Requirement: Device Does Not Expose Executable Pointer

Device SHALL not expose native kernel executable pointers.

#### Scenario: GPU device metadata requested

Given diagnostics inspect Device

When metadata is returned

Then native loaded kernel addresses are absent.

---

### Requirement: Device Trait Remains Compilation-Free

Device public contract SHALL NOT gain arbitrary kernel source compilation
methods.

#### Scenario: PM proposes Device::compile

Given Provider requires Triton compilation

When architecture is implemented

Then capability is added to Provider, not Device.

---

### Requirement: Device Binding Is Sufficient Target Reference

Runtime SHALL identify compilation target using portable Device binding and
metadata.

#### Scenario: Native CUDA device exists

Given Provider maps DeviceBinding internally to native CUDA device

When compilation occurs

Then native mapping remains private.

---

### Requirement: Device Pressure May Influence Optimization

Runtime SHALL NOT let device pressure signals override eligibility constraints, though pressure MAY affect ranking among otherwise eligible candidates.

#### Scenario: GPU saturated

Given GPU and CPU candidates are both eligible

And policy allows dynamic pressure-aware selection

When GPU pressure is high

Then CPU may rank higher.

---

### Requirement: Device Unavailability Is Eligibility Constraint

Unavailable Device SHALL not remain eligible merely because benchmark is fast.

#### Scenario: GPU offline

Given GPU Device is unavailable

When ranking occurs

Then GPU candidates are excluded.

---

### Requirement: Device Metadata Does Not Perform Selection

Device SHALL expose state and capabilities but SHALL not select Kernels.

#### Scenario: Device health queried

Given Runtime reads Device health

When Kernel choice is made

Then Runtime policy owns decision.
