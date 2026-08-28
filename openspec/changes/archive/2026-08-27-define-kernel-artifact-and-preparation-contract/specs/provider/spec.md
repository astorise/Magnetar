## ADDED Requirements

### Requirement: Provider Owns Kernel Preparation

Provider SHALL own transformation from Provider-consumable Kernel Artifact to
Prepared Kernel state.

#### Scenario: Load compiled kernel

Given Provider receives compatible compiled artifact

When preparation succeeds

Then Provider returns opaque PreparedKernelId.

---

### Requirement: Provider Keeps Native Handles Private

Provider SHALL keep native executable handles private.

#### Scenario: Metal pipeline

Given Metal Provider creates compute pipeline

When Runtime receives PreparedKernelId

Then pipeline pointer/object is not exposed.

---

### Requirement: Provider Kernel Lifecycle Independent From Provider Lifecycle

Provider SHALL support individual Prepared Kernel destruction without requiring
Provider unload where platform permits.

#### Scenario: Retire kernel

Given prepared kernel is no longer used

When Runtime retires it

Then Provider destroys kernel state while Provider remains active.

---

### Requirement: Provider Compilation Capability Is Separate

Provider compilation capability SHALL be modeled separately from Device.

#### Scenario: CUDA Provider accepts PTX

Given CUDA Provider supports PTX preparation

When capability is advertised

Then Device trait remains unchanged.

---

### Requirement: Provider Rejects Unsupported Artifact

Provider SHALL reject unsupported artifact format or compatibility.

#### Scenario: WGSL sent to CUDA Provider

Given CUDA Provider cannot prepare WGSL

When preparation is attempted

Then structured unsupported format error is returned.