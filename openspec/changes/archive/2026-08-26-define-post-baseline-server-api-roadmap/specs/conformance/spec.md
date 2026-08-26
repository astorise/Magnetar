## ADDED Requirements

### Requirement: Server API Conformance

Conformance SHALL validate Server API boundaries and Runtime API usage.

#### Scenario: Server conformance

Given server API implementation exists

When conformance runs

Then server requests use Runtime Inference API and preserve redaction.

---

### Requirement: Server Boundary Conformance

Conformance SHALL validate server does not read arbitrary files, execute tools,
execute shell/processes, execute Git, or download arbitrary models during
generation.

#### Scenario: Server filesystem violation

Given generation request asks server to read arbitrary file

When conformance runs

Then request is denied.

---

### Requirement: Server Streaming Conformance

Conformance SHALL validate server streaming preserves Runtime event ordering and
redaction.

#### Scenario: Stream order

Given Runtime emits ordered generation events

When server streams them

Then order and redaction are preserved.