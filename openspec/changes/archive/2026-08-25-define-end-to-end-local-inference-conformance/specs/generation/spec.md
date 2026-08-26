## ADDED Requirements

### Requirement: E2E Uses Generation Contract

E2E conformance SHALL execute generation through Generation Contract.

#### Scenario: Generate fixture output

Given tokenized fixture prompt

When generation runs

Then Runtime performs prefill, decode, Sampling, stop handling, and usage
accounting through Generation Contract.

---

### Requirement: E2E Validates Generation Failure

E2E conformance SHALL validate generation failure paths such as cancellation,
timeout, policy denial, and unsupported operator.

#### Scenario: Cancel generation

Given generation is active

When cancellation is requested

Then Runtime reports generation cancelled or cancellation limitation according
to policy.