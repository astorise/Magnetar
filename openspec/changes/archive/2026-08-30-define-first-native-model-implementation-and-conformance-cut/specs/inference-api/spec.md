## ADDED Requirements
### Requirement: RuntimeInferenceApi Cutover Removes Model Callback Authority

The final normal inference API SHALL not require a caller-provided model-forward
or logits callback.

#### Scenario: API type inspected

Given caller wants text generation

When constructing request

Then prompt/model/generation configuration is sufficient.

### Requirement: Migration Helper Is Non-Authoritative

If a callback-based helper remains temporarily, it SHALL be clearly outside the
mandatory production first-profile API.

#### Scenario: Legacy test uses helper

Given helper remains for isolated tests

When native profile conformance runs

Then helper is not exercised.

### Requirement: API E2E Precedes CLI E2E

A RuntimeInferenceApi integration test SHALL succeed before CLI integration is
considered complete.

#### Scenario: CLI not yet implemented

Given Runtime API can execute fixture prompt

When API test runs

Then generated result can be verified independently.