## ADDED Requirements
### Requirement: Runtime Owns Prepared Execution Plans

Runtime SHALL own construction, validation, publication and retirement of
Prepared Execution Plans.

#### Scenario: Model load completes

Given graph and Kernels are available

When Runtime builds Plan

Then cross-Provider execution decisions remain Runtime-owned.

---

### Requirement: Runtime Uses Ready Plan On Hot Path

Where compatible ready Plan exists, Runtime SHALL use it instead of repeating
complete resolution.

#### Scenario: Decode loop

Given decode Plan is ready

When token step executes

Then Runtime performs bounded guard/resource binding and dispatch.

---

### Requirement: Runtime Replans Outside Active Hot Path

Runtime SHALL schedule full Plan rebuild outside current Kernel execution.

#### Scenario: Plan becomes stale

Given current Plan remains safe

When replan requested

Then replacement can be prepared without blocking current active invocation.

---

### Requirement: Runtime Fail-Closes Hard Invalidation

Runtime SHALL not execute hard-invalidated Plan.

#### Scenario: Kernel revoked

Given no fallback Plan exists

When invocation begins

Then Runtime returns structured failure/replan state instead of using revoked
binding.

---

### Requirement: Runtime Supports Atomic Plan Generation Replacement

Runtime SHALL preserve coherent Plan generation across concurrent executions.

#### Scenario: Replacement ready

Given old Plan has active references

When new Plan is published

Then new work uses new generation while old stays alive for existing work.

---

### Requirement: Runtime Does Not Persist Native Plan State As Portable Data

Runtime restart SHALL not trust persisted PreparedKernelId or
ProviderPreparedSegmentId.

#### Scenario: Plan recipe loaded after restart

Given logical Plan metadata exists

When Runtime reconstructs it

Then Provider-native prepared state is re-established.
