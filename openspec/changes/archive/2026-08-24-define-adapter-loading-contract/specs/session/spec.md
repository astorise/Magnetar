## ADDED Requirements
### Requirement: Session May Reference Active Adapters

An Inference Session SHALL reference active adapters through session policy when adapters are active.

#### Scenario: Session adapter active

Given session policy allows adapter A

When adapter A is activated for the session

Then future generation in that session uses A according to Runtime plan.

---

### Requirement: Session Controls Adapter Activation

Session policy SHALL control adapter activation, deactivation, allowed adapters,
maximum active adapters, merge permission, and adapter memory budget.

#### Scenario: Activation denied

Given session policy denies adapter activation

When activation is requested

Then Runtime rejects activation with activation-denied.

---

### Requirement: Session Close Applies Adapter Policy

When a session closes, session-scoped adapters SHALL be deactivated, retained,
or unloaded according to policy.

#### Scenario: Close session with adapter

Given session-scoped adapter A is active

When the session closes

Then Runtime applies adapter cleanup policy.