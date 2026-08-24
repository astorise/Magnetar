## ADDED Requirements
### Requirement: Operators Have Implementation Scope Classification

Operators SHALL be classifiable by first implementation scope.

#### Scenario: Classify attention

Given attention Operator metadata is inspected

When first scope is active

Then attention is classified as required-now or as the specific supported
attention baseline.

---

### Requirement: Required Operators Have Conformance Fixtures

Operators in required-now scope SHALL have conformance fixtures.

#### Scenario: Matmul fixture

Given matmul is required-now

When conformance runs

Then matmul fixtures are available.

---

### Requirement: Placeholder Operators Do Not Imply Kernel Availability

A placeholder Operator SHALL not imply that a Kernel exists.

#### Scenario: Paged attention placeholder

Given paged-attention Operator identity exists

When Kernel Registry is queried

Then no Kernel is assumed unless advertised.