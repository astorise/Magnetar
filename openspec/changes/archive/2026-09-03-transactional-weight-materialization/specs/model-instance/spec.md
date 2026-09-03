## MODIFIED Requirements

### Requirement: Model Instance Creation

Model Instance creation SHALL require successful Model Loading or an explicit policy-controlled loading path, and creation SHALL NOT by itself produce a Ready Model Instance.

A newly created Model Instance SHALL be left in a non-Ready lifecycle state (`Loading`). Transition to Ready SHALL happen only through a separate, explicit step performed after every mandatory readiness condition for that instance -- including, where applicable, weight materialization -- has actually been satisfied.

#### Scenario: Artifact only

Given a valid Model Artifact is not loaded

When a caller requests a Model Instance without implicit loading policy

Then creation fails.

#### Scenario: Creation alone does not imply readiness

Given a Model Instance has just been created from a successfully loaded artifact

When no explicit readiness-completing step has run yet for it

Then the instance reports a non-Ready lifecycle and readiness state, and any check that inspects only that state (not a deeper, resource-specific check) correctly rejects it as not usable
