## ADDED Requirements

### Requirement: Runtime Release Security Boundary

Runtime release SHALL preserve inference-only authority and default redaction.

#### Scenario: Release Runtime audit

Given Runtime public API is audited

When authority surface is inspected

Then filesystem, network, secret, shell, process, Git, tool, and agent
authority are absent.

---

### Requirement: Runtime Rejects Security Boundary Violations

Runtime SHALL reject attempts to use inference APIs for non-inference authority.

#### Scenario: Secret access request

Given inference request asks Runtime to read secret

When validation runs

Then Runtime rejects request.