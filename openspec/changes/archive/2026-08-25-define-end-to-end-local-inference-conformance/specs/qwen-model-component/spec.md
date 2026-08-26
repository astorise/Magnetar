## ADDED Requirements

### Requirement: E2E Uses Qwen Baseline Component

E2E conformance SHALL use the Qwen-like Model Component baseline or compatible
native architecture implementation for the first fixture path.

#### Scenario: Qwen fixture

Given fixture declares Qwen-like architecture

When Model Loading resolves architecture

Then Qwen baseline component validates it.

---

### Requirement: E2E Validates Qwen Graph Production

E2E conformance SHALL validate Qwen prefill and decode graph production.

#### Scenario: Decode graph produced

Given fixture Model Instance is ready

When decode is requested

Then Qwen baseline component produces a portable decode graph that Runtime
validates.