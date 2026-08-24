## ADDED Requirements

### Requirement: Session Policy Constrains Batching

Session policy SHALL constrain batching behavior for operations in that session.

#### Scenario: Session rejects parallel operation

Given a session allows only one active operation

When batching attempts to schedule a second active operation from that session

Then Runtime rejects or queues it according to session policy.

---

### Requirement: Session Cancellation Affects Batched Operations

Cancelling a session SHALL cancel or drain its batched operations according to
policy.

#### Scenario: Cancel session

Given a session has queued and active batched operations

When session cancellation is requested

Then Runtime applies cancellation to those operations.
