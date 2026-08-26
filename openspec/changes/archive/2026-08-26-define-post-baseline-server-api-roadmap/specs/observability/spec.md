## ADDED Requirements

### Requirement: Server Observability Is Redacted

Server observations SHALL be redacted by default.

#### Scenario: Request logged

Given generation request contains prompt text

When server emits observation

Then raw prompt text is absent by default.

---

### Requirement: Server Runtime Correlation

Server observations SHALL not expose raw data during correlation.

Server observations SHOULD correlate request, Runtime request, streaming,
cancellation, diagnostics, and errors.

#### Scenario: Stream interrupted

Given stream is interrupted

When observations are emitted

Then server and Runtime events can be correlated.