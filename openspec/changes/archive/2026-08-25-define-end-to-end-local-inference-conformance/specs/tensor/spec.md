## ADDED Requirements

### Requirement: E2E Validates Tensor Resources

E2E conformance SHALL validate Tensor Resource creation, readiness, dtype,
layout, no raw pointer exposure, output metadata, and cleanup.

#### Scenario: Tensor resource ready

Given Reference CPU kernel produces operator output

When dispatch completes

Then Tensor Resource metadata is ready and redacted.

---

### Requirement: E2E Validates No Raw Tensor Exposure

E2E conformance SHALL fail if Runtime exposes raw tensor pointers or raw tensor
values by default.

#### Scenario: Tensor diagnostics

Given diagnostics include tensor metadata

When E2E validates redaction

Then no raw tensor value or pointer is present.