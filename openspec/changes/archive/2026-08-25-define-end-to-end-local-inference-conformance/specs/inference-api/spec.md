## ADDED Requirements

### Requirement: E2E Uses Runtime Inference API

End-to-End Local Inference Conformance SHALL enter inference through Runtime
Inference API.

#### Scenario: E2E request

Given local E2E test starts inference

When request is submitted

Then it uses Runtime Inference API, not internal Provider or Kernel APIs.

---

### Requirement: E2E Validates One-Shot Inference

E2E suite MAY validate one-shot inference, but one-shot SHALL use normal Runtime
contracts internally.

#### Scenario: One-shot E2E

Given one-shot request runs

When tracing or observations are inspected

Then implicit session, tokenization, generation, and dispatch contracts were
used.

---

### Requirement: E2E Validates API Errors

E2E suite SHALL validate structured Runtime Inference API errors for failure
cases.

#### Scenario: Invalid model reference

Given invalid model reference is submitted

When API validates it

Then structured model resolution or invalid reference error is returned.