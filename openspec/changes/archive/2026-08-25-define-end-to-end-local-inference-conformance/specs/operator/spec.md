## ADDED Requirements

### Requirement: E2E Exercises Required Operators

E2E conformance SHALL exercise required-now Operators for the first decoder
baseline.

#### Scenario: Operator coverage

Given E2E success path completes

When report is generated

Then required operator coverage is recorded.

---

### Requirement: E2E Fails Missing Operator Coverage

E2E conformance SHALL report missing required operator coverage.

#### Scenario: Missing RoPE coverage

Given fixture path does not exercise RoPE

When operator coverage is required

Then E2E reports missing coverage.