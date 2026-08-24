## ADDED Requirements
### Requirement: Provider Advertises Adapter Capabilities

Providers SHALL advertise adapter capabilities when Provider-supported adapter execution is exposed.

#### Scenario: Provider supports LoRA

Given a Provider supports LoRA overlay execution

When Runtime reads Provider advertisements

Then supported method, dtype, rank, layout, and execution strategy are visible
to Runtime policy.

---

### Requirement: Provider Adapter Resources Are Opaque

Provider-owned adapter resources SHALL remain opaque to Components and public
portable APIs.

#### Scenario: Native adapter resource

Given Provider materializes adapter state

When Runtime exposes adapter status

Then it exposes Runtime metadata only.

---

### Requirement: Provider Adapter Failure Maps To Runtime Error

Provider failures during adapter materialization, activation, merge, or fused execution SHALL map to stable Runtime adapter errors.
execution SHALL map to stable Runtime adapter errors.

#### Scenario: Fused adapter unsupported

Given Provider cannot execute a requested fused adapter path

When Runtime plans execution

Then it reports Provider-adapter-unsupported or equivalent structured error.