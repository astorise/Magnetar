## ADDED Requirements

### Requirement: Server Sessions Are Runtime Inference Sessions

Server session endpoints SHALL create and use Runtime Inference Sessions.

#### Scenario: Server session

Given server creates session

When Runtime records state

Then state remains inference-scoped.

---

### Requirement: Server Connection State Is Separate

Server transport connection state SHALL be separate from Runtime Session state.

#### Scenario: Client disconnects

Given client disconnects from streaming

When server handles disconnect

Then Runtime cancellation or session cleanup follows policy explicitly.