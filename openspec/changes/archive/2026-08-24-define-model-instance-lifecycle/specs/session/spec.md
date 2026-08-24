## ADDED Requirements

### Requirement: Session References Model Instance

An Inference Session SHALL reference a ready Model Instance or use
policy-controlled implicit loading.

#### Scenario: Session with ready instance

Given a ready Model Instance exists

When a session is created for it

Then the session references the Model Instance through Runtime-owned identity.

---

### Requirement: Session Does Not Own Model Instance

Closing a session SHALL not unload a Model Instance unless Runtime policy
requires it.

#### Scenario: Close session

Given multiple sessions reference one Model Instance

When one session closes

Then the Model Instance remains available for other sessions if policy allows.

---

### Requirement: Session Handles Instance Unload

If a referenced Model Instance unloads, fails, or becomes invalid, sessions SHALL
be drained, failed, rebound, or closed according to policy.

#### Scenario: Instance unloaded

Given a session references Model Instance M

When M is unloaded

Then Runtime updates session state according to policy.
