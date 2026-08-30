## ADDED Requirements
### Requirement: CLI Cutover Occurs After Runtime API

The CLI SHALL integrate only through the stabilized RuntimeInferenceApi first
profile path.

#### Scenario: Runtime model execution unavailable

Given CLI command is implemented

When API cannot execute model

Then CLI SHALL not provide its own fallback logits.

### Requirement: CLI E2E Is Final User Boundary

At least one final conformance test SHALL invoke user-facing CLI or equivalent
CLI command boundary.

#### Scenario: Fixture prompt

Given fixture installed/available

When command runs

Then generated output comes from Runtime native model path.

### Requirement: CLI Placeholder Removal Is Exit Gate

Any placeholder logits used by prior CLI harness SHALL be removed or excluded
from the final profile before stabilization cut.

#### Scenario: Placeholder function remains reachable

Given native `magnetar run` can still use placeholder path

When stabilization is evaluated

Then cut is incomplete.