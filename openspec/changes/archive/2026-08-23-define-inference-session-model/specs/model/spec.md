## ADDED Requirements
### Requirement: Session References Model Context

An Inference Session SHALL reference a validated model context or future Model
Instance.

A Model Artifact alone SHALL not be sufficient to run session generation unless
Runtime policy performs loading first.

#### Scenario: Session with artifact only

Given a session creation request references only an unloaded Model Artifact

When Runtime does not allow implicit loading

Then session creation fails with model-unavailable.

---

### Requirement: Model Residency May Outlive Session

Model residency SHALL be able to outlive a session according to Runtime cache policy.

#### Scenario: Session closes

Given a session references resident model memory

When the session closes

Then model residency may remain cached if Runtime policy allows it.