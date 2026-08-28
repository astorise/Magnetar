## ADDED Requirements

### Requirement: Optimization Plane Separation Conformance

Conformance SHALL validate optimization orchestration remains outside Runtime
inference authority.

#### Scenario: Runtime API audit

Given public Runtime Inference API is inspected

When conformance runs

Then arbitrary generator/optimization execution capability is absent.

---

### Requirement: No Hot-Path Optimization Conformance

Conformance SHALL prove token decode cannot synchronously launch optimization
campaign.

#### Scenario: Kernel unavailable

Given required optimized Kernel is missing

When decode runs

Then structured fallback/failure occurs rather than agent search.

---

### Requirement: Recommendation Does Not Promote Conformance

Conformance SHALL validate external recommendation cannot directly change
active Kernel.

#### Scenario: Recommended candidate

Given recommendation exists

When normal promotion validation has not run

Then active Registry state remains unchanged.

---

### Requirement: Runtime Revalidation Conformance

Conformance SHALL validate candidate is re-evaluated using current production
state.

#### Scenario: Qualification expired

Given campaign evidence was once valid

But qualification is now expired

When promotion is attempted

Then candidate is rejected.

---

### Requirement: Native Handle Boundary Conformance

Conformance SHALL prove Optimization Plane cannot transport Provider native
handles as artifacts.

#### Scenario: Worker-local PreparedKernelId

Given worker prepares candidate

When result is exported

Then production does not reuse worker-local native handle mapping.

---

### Requirement: Offline Inference Conformance

Conformance SHALL prove already prepared baseline inference can operate without
optimization-service connectivity.

#### Scenario: Network disabled

Given compatible Kernel artifacts are local

When inference runs

Then external optimizer is not contacted.

---

### Requirement: Credential Boundary Conformance

Conformance SHALL prove optimization credentials do not enter Runtime
Inference Session.

#### Scenario: Generator token configured

Given CLI/CI owns token

When Runtime session is created

Then token is absent.

---

### Requirement: Workload Privacy Conformance

Conformance SHALL validate default Optimization Workload Profile contains no raw
prompt/user content.

#### Scenario: Profile generated

Given production statistics are summarized

When profile is inspected

Then aggregate shape/sequence metadata may exist while raw prompts are absent.

---

### Requirement: Generator Identity Does Not Grant Trust Conformance

Conformance SHALL prove known generator identity alone cannot trust artifact.

#### Scenario: Trusted-name generator

Given artifact provenance says approved generator

When no authenticated trust mechanism exists

Then provenance alone is insufficient.

---

### Requirement: Campaign Failure Isolation Conformance

Conformance SHALL prove failed campaign does not disturb active known-good
Kernel.

#### Scenario: All candidate builds fail

Given production Kernel generation 8 is active

When optimization campaign fails

Then generation 8 remains active.

---

### Requirement: Tachyon Independence Conformance

Conformance SHALL validate Magnetar Runtime has no required direct Tachyon
dependency for Kernel optimization.

#### Scenario: Standalone deployment

Given Tachyon is absent

When local/CI-produced artifacts are supplied

Then Magnetar can validate and execute them.

---

### Requirement: Tooling Authority Boundary Conformance

Conformance SHALL validate CLI/external tooling authority is not ambiently
delegated to Runtime.

#### Scenario: Optimization CLI has repository access

Given CLI reads source repository

When Kernel Artifact enters Runtime

Then Runtime gains artifact data but not repository authority.

---

### Requirement: Selection Policy Still Authoritative Conformance

Conformance SHALL prove optimization ranking cannot override production Kernel
Selection Policy.

#### Scenario: Campaign says candidate fastest

Given current memory policy makes candidate infeasible

When Runtime selects Kernel

Then candidate remains excluded.

---

### Requirement: Optimization Observability Redaction Conformance

Conformance SHALL validate optimization events redact raw user data, secrets,
native handles and model internals by default.

#### Scenario: Campaign error

Given failure context contains sensitive data

When observation is exported

Then sensitive values are absent.