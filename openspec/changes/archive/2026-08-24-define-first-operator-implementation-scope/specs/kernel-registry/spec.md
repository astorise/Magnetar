## ADDED Requirements
### Requirement: Registry Supports First Scope Validation

Kernel Registry SHALL support validation that required-now operators have
eligible Kernels.

#### Scenario: Validate first scope

Given first scope requires RMSNorm

When Kernel Registry is checked

Then at least one eligible RMSNorm Kernel must exist or validation fails.

---

### Requirement: Registry Does Not Create Placeholder Candidates

Kernel Registry SHALL not create candidates for placeholder Operators unless a
Provider advertises a concrete Kernel.

#### Scenario: Placeholder lookup

Given no Provider advertises paged-attention

When Registry is queried

Then no candidate is returned.

---

### Requirement: Registry Reports Missing Required Kernels

Kernel Registry SHALL report missing required-now Kernels with structured
errors.

#### Scenario: Missing attention kernel

Given attention is required-now

And no eligible Kernel exists

When first scope validation runs

Then Runtime reports first-scope-kernel-missing.