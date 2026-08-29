## ADDED Requirements
### Requirement: Autotuning Evidence May Be Captured In Plan

Prepared Execution Plan SHALL be able to reference the Autotuning Record used to select a
specialization.

#### Scenario: BLOCK_M 64 wins

Given tuning record selects specialization S64

When Plan is built

Then Plan binding records S64 and tuning evidence identity.

---

### Requirement: Stale Tuning Can Stale Plan

Autotuning evidence becoming performance-stale SHALL be able to mark Plan stale without
automatically making it unsafe.

#### Scenario: Driver behavior changes slightly

Given Kernel remains compatible/qualified

But tuning result is stale

When Runtime evaluates Plan

Then it may request re-tuning/re-plan while allowing temporary execution by
policy.

---

### Requirement: Tuning Does Not Occur During Ready Plan Dispatch

Executing Prepared Execution Plan SHALL not implicitly launch autotuning.

#### Scenario: Tuning cache expired during decode

Given Plan remains safe

When decode executes

Then background re-tuning may be requested but current dispatch does not
benchmark alternatives.
