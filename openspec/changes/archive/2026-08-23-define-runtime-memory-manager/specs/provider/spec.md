## ADDED Requirements

### Requirement: Provider Memory Is Coordinated By Runtime Memory Manager

Provider-owned memory resources SHALL be coordinated with Runtime Memory Manager
state.

Providers may allocate native resources, but Runtime memory policy SHALL remain
owned by Memory Manager.

#### Scenario: Provider allocates tensor

Given a Provider creates a tensor in native memory

When Runtime records the result

Then Memory Manager tracks residency and Resource Affinity records Provider
ownership.

---

### Requirement: Provider Does Not Expose Raw Memory Handles

Providers SHALL NOT expose raw native memory handles through portable Runtime
or Component APIs.

#### Scenario: Provider returns native allocation

Given Provider execution creates a native allocation

When Runtime exposes result metadata

Then the result uses opaque Runtime resource identity

And not a raw native pointer or driver handle.

---

### Requirement: Provider Reports Memory Pressure

Providers SHALL expose memory pressure signals where Provider or Device memory pressure is known.

#### Scenario: Device memory high

Given Provider observes high Device memory usage

When status is reported

Then Memory Manager or Runtime policy can consume the memory pressure signal.

---

### Requirement: Provider Allocation Failure Maps To Memory Error

Provider allocation failures SHALL map to stable Runtime memory errors.

#### Scenario: Provider out of memory

Given Provider cannot allocate required Device memory

When execution is attempted

Then Runtime reports a structured memory or out-of-memory error rather than an
opaque Provider failure.
