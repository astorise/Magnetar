## ADDED Requirements

### Requirement: E2E Observability Is Redacted

E2E conformance SHALL validate observability redaction for inference events.

#### Scenario: Prompt redaction

Given prompt text is submitted

When observability events are emitted

Then raw prompt text is absent by default.

---

### Requirement: E2E Observability Supports Correlation

E2E observations SHALL include correlation IDs that connect request, model
loading, session, generation, graph, kernel, and result events without exposing
raw data.

#### Scenario: Correlated failure

Given generation fails during Kernel dispatch

When observations are inspected

Then failure can be correlated across Runtime subsystems.