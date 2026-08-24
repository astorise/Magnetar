## ADDED Requirements

### Requirement: Generation Requires Model Instance

Generation SHALL execute against a ready Model Instance or explicit
policy-controlled implicit load path.

#### Scenario: No instance

Given a valid Model Artifact exists

But no ready Model Instance exists

When generation is requested

Then Runtime rejects the request or loads explicitly according to policy.

---

### Requirement: Generation Acquires Instance Usage

Generation SHALL acquire Model Instance usage before prefill or decode.

#### Scenario: Start generation

Given a ready Model Instance

When generation starts

Then Runtime acquires usage before execution begins.

---

### Requirement: Generation Releases Instance Usage

Generation SHALL release Model Instance usage when operation completes, fails,
or is cancelled.

#### Scenario: Generation cancelled

Given generation is active

When cancellation completes

Then Runtime releases Model Instance usage.

---

### Requirement: Generation Handles Instance State Changes

Generation SHALL handle Model Instance draining, suspension, failure,
invalidation, reload, or unload according to Runtime policy.

#### Scenario: Instance draining

Given an instance enters draining

When a new generation request arrives

Then Runtime rejects or routes the request according to policy.
