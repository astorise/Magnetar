## ADDED Requirements

### Requirement: CLI Uses Production Runtime Inference API
`magnetar-cli` SHALL submit generation requests through production RuntimeInferenceApi entry points and SHALL NOT depend on `e2e_conformance` for normal `run`, `chat`, `serve`, or agent generation behavior.

#### Scenario: CLI run uses model_ref
- **WHEN** a user runs `magnetar run <model_ref> <prompt>`
- **THEN** CLI passes the resolved model_ref to RuntimeInferenceApi and does not ignore it or replace it with a hard-coded fixture identity.

#### Scenario: CLI has no kernel/provider path
- **WHEN** CLI handles a generation command
- **THEN** CLI does not call Kernel, Provider, Device, Memory Manager, or Reference CPU execution APIs directly.

### Requirement: CLI Reports Runtime Generation Errors Structurally
CLI SHALL map Runtime generation failures to stable user-facing error categories.

#### Scenario: Plan unavailable
- **WHEN** Runtime reports model-not-found, artifact-invalid, trust-rejected, load-failed, component-load-failed, provider-unavailable, plan-unavailable, generation-failed, generation-cancelled, or equivalent structured errors
- **THEN** CLI reports the corresponding category without leaking raw Provider, tensor, KV, prompt, or artifact contents.
