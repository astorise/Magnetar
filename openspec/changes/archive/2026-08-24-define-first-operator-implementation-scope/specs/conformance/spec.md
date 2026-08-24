## ADDED Requirements
### Requirement: First Scope Conformance Suite

Conformance SHALL include fixtures for each required-now Operator.

#### Scenario: Required operator fixture

Given `softmax` is required-now

When conformance suite runs

Then softmax fixtures are included.

---

### Requirement: First Scope Conformance Is CPU-Compatible

First scope conformance SHALL be runnable without external GPU hardware.

#### Scenario: CPU-only conformance

Given only Reference CPU Provider is available

When first scope conformance runs

Then supported required-now fixtures can execute.

---

### Requirement: First Scope Conformance Reports Placeholders

Placeholder Operators SHALL be reported as pending or unsupported rather than
passing silently.

#### Scenario: Placeholder conformance

Given `paged-attention` is placeholder

When conformance report is generated

Then it is reported as placeholder, pending, or unsupported.