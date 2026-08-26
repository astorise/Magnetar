## ADDED Requirements

### Requirement: E2E Uses Inference Sessions

E2E conformance SHALL create and close Runtime Inference Sessions.

#### Scenario: Session lifecycle

Given fixture model instance is ready

When E2E success path runs

Then a Runtime Inference Session is created, used, and closed.

---

### Requirement: E2E Validates Session Closed Error

E2E conformance SHALL validate closed session behavior.

#### Scenario: Reuse closed session

Given session is closed

When generation is requested

Then Runtime returns structured session closed error.