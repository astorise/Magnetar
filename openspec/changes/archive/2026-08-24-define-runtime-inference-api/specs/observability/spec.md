## ADDED Requirements

### Requirement: Inference API Observability Is Redacted By Default

Observability produced by Runtime Inference API SHALL be redacted by default.

#### Scenario: Prompt submitted

Given caller submits prompt text

When inference observations are emitted

Then raw prompt text is not logged by default.

---

### Requirement: Inference API Observability Preserves Correlation

Runtime Inference API observations SHALL include stable correlation metadata for requests, sessions, generations, streams, cache events, and errors.

#### Scenario: Generation failure

Given generation fails

When observability emits error metadata

Then events can be correlated without exposing raw prompt or handles.