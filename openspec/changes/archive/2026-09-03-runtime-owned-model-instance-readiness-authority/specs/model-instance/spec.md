## MODIFIED Requirements

### Requirement: Model Instance Readiness

Model Instance lifecycle and readiness SHALL be distinct.

Readiness SHALL consider residency, Provider readiness, Device readiness,
adapter state, memory pressure, Runtime policy, and architecture implementation
readiness.

Readiness-relevant facts the Runtime can itself observe -- including whether
mandatory weight resources are bound, whether a pinned Provider actually
resolves and offers an execution API, and whether a pinned Device is
available -- SHALL be derived from actual Runtime state, not accepted
outright from a caller-supplied claim. A caller MAY assert a stricter
(`false`) value than Runtime state alone would produce; a caller SHALL NOT
be able to assert a Runtime-observable fact as `true` when the Runtime does
not itself observe it as true.

#### Scenario: Provider not ready

Given an instance lifecycle exists

But Provider is not ready

When Runtime checks readiness

Then the instance is not ready for generation.

#### Scenario: Caller cannot forge a Runtime-observable fact

Given a Model Instance has no weight resources bound

When a caller requests warmup asserting weights are materialized

Then the Runtime's own observation of empty resource bindings overrides the caller's claim and the instance does not become Ready.

---

### Requirement: Model Instance Warmup

Model Instance warmup MAY be supported and SHALL be policy-controlled.

Warmup failure SHALL prevent ready state.

Regardless of warmup policy, `readiness` SHALL NOT report `Ready` while
`lifecycle` has not itself reached a state that supports inference use. A
warmup policy that does not perform lifecycle transitions SHALL NOT be able
to publish `Ready` readiness as a side effect.

#### Scenario: Warmup failure

Given Provider warmup fails

When Runtime evaluates instance readiness

Then the instance becomes failed or not-ready according to policy.

#### Scenario: Disabled policy cannot forge readiness

Given a Model Instance is in a lifecycle state that does not support inference use

When warmup is invoked with a policy that does not transition the lifecycle

Then readiness does not report Ready even if the supplied checks would otherwise compute Ready

---

### Requirement: Generation Requires Ready Model Instance

Generation SHALL require a ready Model Instance or a policy-controlled implicit
load path.

Accepting usage (acquiring a generation reference or a usage handle) SHALL
require both that the lifecycle is in a state that supports inference use
and that readiness reports Ready. An internally inconsistent combination
(readiness reporting Ready while lifecycle does not support inference use)
SHALL be rejected, regardless of how that inconsistency arose.

#### Scenario: Generate on failed instance

Given a Model Instance is failed

When generation is requested

Then Runtime rejects generation.

#### Scenario: Inconsistent lifecycle and readiness reject usage

Given a Model Instance reports readiness Ready but its lifecycle has not reached a state that supports inference use

When a caller attempts to acquire usage or a generation reference

Then Runtime rejects the request based on the lifecycle, not only the readiness value
