## ADDED Requirements

### Requirement: E2E Executes Through Reference CPU Provider

The first E2E local inference suite SHALL execute supported inference through
Reference CPU Provider.

#### Scenario: Reference CPU dispatch

Given required operators have Reference CPU Kernels

When E2E generation runs

Then Kernel Dispatch invokes Reference CPU kernels through Runtime.

---

### Requirement: E2E CPU Execution Is Not Hidden Fallback

Reference CPU use in E2E SHALL be explicit and policy-controlled.

#### Scenario: CPU fallback hidden

Given E2E expects explicit CPU policy

When Runtime uses CPU without policy

Then E2E fails with boundary or fallback violation.