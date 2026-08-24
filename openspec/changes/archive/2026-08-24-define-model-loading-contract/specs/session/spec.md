## ADDED Requirements

### Requirement: Session May Require Loaded Model Context

An Inference Session SHALL require an existing loaded model context when policy disables implicit loading.

If implicit loading is disabled, session creation SHALL fail when the model is
not loaded.

#### Scenario: Session with unloaded model

Given session creation references a valid but unloaded Model Artifact

And implicit loading is disabled

When Runtime creates the session

Then creation fails with model-unavailable.

---

### Requirement: Session May Trigger Policy-Controlled Loading

Runtime SHALL allow session creation to trigger implicit model loading only when policy
explicitly permits it.

#### Scenario: Implicit load allowed

Given session creation references an unloaded but valid Model Artifact

And policy permits implicit loading

When session creation runs

Then Runtime performs Model Loading before the session becomes ready.

---

### Requirement: Session Close Does Not Imply Model Unload

Closing a session SHALL not automatically unload a model unless model residency
policy requires it.

#### Scenario: Close session

Given a session references a loaded model context

When the session closes

Then the model may remain resident according to Runtime cache policy.
