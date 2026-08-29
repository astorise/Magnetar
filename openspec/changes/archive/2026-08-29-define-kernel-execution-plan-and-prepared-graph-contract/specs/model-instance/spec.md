## ADDED Requirements
### Requirement: Model Instance Owns Plan Family Context

Model Instance SHALL be able to maintain one or more Prepared Execution Plans for compatible
workloads.

#### Scenario: Prefill and decode

Given Model Instance supports both phases

When prepared

Then it may hold distinct ready Plan families for prefill and decode.

---

### Requirement: Model Revision Participates In Plan Validity

Prepared Execution Plan SHALL not silently survive incompatible Model Instance
revision.

#### Scenario: Adapter set changes

Given adapter configuration modifies graph/resources

When revision changes

Then dependent Plan is stale or invalid according to compatibility.

---

### Requirement: Model Instance Readiness May Depend On Required Plans

Deployment policy SHALL be able to require mandatory Prepared Execution Plans before Model
Instance becomes ready.

#### Scenario: Strict low-latency deployment

Given required decode Plan is not prepared

When readiness is evaluated

Then Model Instance remains warming/not-ready according to policy.

---

### Requirement: Optional Plan Preparation Need Not Block Readiness

Additional workload-specific Plans SHALL be able to be prepared lazily while baseline
known-good Plan remains available.

#### Scenario: Long-sequence Plan absent

Given normal sequence Plan is ready

When Model Instance starts

Then optional 32k sequence Plan may build later.
