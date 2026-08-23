## ADDED Requirements
### Requirement: Runtime Owns Inference Sessions

Runtime SHALL own creation, lookup, authorization, lifecycle, and cleanup of
Inference Sessions.

#### Scenario: Runtime creates session

Given a valid session creation request

When Runtime validates it

Then Runtime issues an opaque session identity.

---

### Requirement: Runtime Applies Session Policy

Runtime SHALL apply session policy to generation operations executed within the
session.

#### Scenario: Max tokens policy

Given a session policy limits max generated tokens to 100

When a generation request asks for 200

Then Runtime rejects or clamps only if explicit policy allows clamping.

---

### Requirement: Runtime Integrates Sessions With Generation

Runtime SHALL allow generation operations to run inside an Inference Session.

#### Scenario: Session generation

Given a ready session

When a generation request references it

Then Runtime uses the session model binding, tokenizer binding, policy, memory,
and cancellation state.

---

### Requirement: Runtime Supports One-Shot Session Semantics

Runtime SHALL support one-shot inference through implicit short-lived session semantics when one-shot generation is enabled by policy.

#### Scenario: One-shot cleanup

Given one-shot generation completes

When Runtime finishes the request

Then session-scoped temporary resources are released.

---

### Requirement: Runtime Cleans Up Session Resources

Runtime SHALL release session-owned resources when a session closes, expires,
fails, or is cancelled according to policy.

#### Scenario: Session expires

Given a session has temporary token buffers

When the session expires

Then Runtime releases those buffers or transfers eligible resources to managed
cache according to policy.

---

### Requirement: Runtime Does Not Expose Raw Session Internals

Runtime SHALL not expose raw Provider handles, Device handles, memory pointers,
raw KV cache contents, or raw prompt text through session APIs by default.

#### Scenario: Session status

Given session status is requested

When Runtime returns status

Then it includes stable metadata only.

---

### Requirement: Runtime Authorizes Session Access

Runtime SHALL authorize session operations.

A valid session ID alone SHALL not grant access.

#### Scenario: Unauthorized session operation

Given a caller presents a valid session ID

But lacks authorization

When it tries to cancel the session

Then Runtime denies the operation.

---

### Requirement: Runtime Observes Session Lifecycle

Runtime SHALL define observations for session creation, state transitions, operations, cancellation, drain, expiration, cleanup, and policy rejection.

#### Scenario: Session closed

Given a session is closed

When cleanup completes

Then Runtime may emit a session-closed observation.