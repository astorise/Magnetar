## ADDED Requirements

### Requirement: Memory Manager Tracks Model Instance Residency

Memory Manager SHALL track all residency associated with Model Instances.

#### Scenario: Instance residency report

Given a Model Instance has host and device residency

When memory usage is queried

Then Memory Manager reports residency by instance metadata.

---

### Requirement: Memory Manager Coordinates Instance Lifecycle

Memory Manager SHALL coordinate allocation, residency update, suspension,
unload, eviction, and release with Model Instance lifecycle.

#### Scenario: Instance unload

Given a Model Instance is unloading

When Runtime releases residency

Then Memory Manager releases or invalidates all associated memory records.

---

### Requirement: Memory Pressure May Affect Instance Readiness

Runtime SHALL define how memory pressure may cause a Model Instance to become suspended, draining,
unloaded, or failed according to Runtime policy.

#### Scenario: Memory pressure

Given Runtime memory pressure is high

When policy permits instance suspension

Then Runtime may mark idle instance suspended and release eligible memory.
