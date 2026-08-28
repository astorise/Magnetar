## ADDED Requirements

### Requirement: Eligibility Precedes Ranking

Conformance SHALL prove ineligible candidates are removed before performance
ranking.

#### Scenario: Fast untrusted Kernel

Given untrusted Kernel is fastest

When conformance runs

Then it is never selected.

---

### Requirement: Memory Feasibility Precedes Ranking

Conformance SHALL prove Memory Manager rejection cannot be overridden.

#### Scenario: Workspace too large

Given fastest Kernel is infeasible

When selection runs

Then feasible slower candidate wins or selection fails.

---

### Requirement: Affinity Precedes Ranking

Conformance SHALL prove Resource Affinity cannot be bypassed by performance.

#### Scenario: Cross-Provider candidate faster

Given movement is forbidden

When ranking runs

Then faster cross-Provider candidate is excluded.

---

### Requirement: Determinism Policy Conformance

Conformance SHALL prove deterministic profile excludes candidates failing
determinism.

#### Scenario: Nondeterministic fastest candidate

Given deterministic mode

When selection runs

Then candidate is not selected.

---

### Requirement: Stable Tie-Break Conformance

Conformance SHALL prove identical selection input yields identical tie result.

#### Scenario: Equal scores

Given candidates have identical score

When selection runs repeatedly

Then selected Kernel remains stable.

---

### Requirement: Benchmark Context Conformance

Conformance SHALL prove incompatible benchmark evidence is not authoritative.

#### Scenario: Different architecture

Given benchmark from sm90

When candidate targets different incompatible architecture

Then evidence is ignored/rejected.

---

### Requirement: Hysteresis Conformance

Conformance SHALL prove insignificant benefit does not force promotion.

#### Scenario: 0.1 percent improvement

Given threshold is higher

When candidate ranks slightly above active

Then active Kernel remains preferred.

---

### Requirement: Explicit Fallback Conformance

Conformance SHALL prove fallback only occurs according to policy.

#### Scenario: Fallback disabled

Given selected Provider unavailable

When policy says fail

Then Runtime fails instead of silently using CPU.

---

### Requirement: No Hidden Data Movement Conformance

Conformance SHALL prove cross-Provider selection respects explicit movement and
host staging rules.

#### Scenario: Host staging forbidden

Given CPU fallback requires staging

When policy forbids it

Then fallback fails.

---

### Requirement: Model Component Independence Conformance

Conformance SHALL prove Model Component cannot force Kernel implementation.

#### Scenario: Component attempts concrete selection

Given Component requests a specific Provider Kernel

When graph is validated

Then request is rejected/ignored according to portable contract.

---

### Requirement: User Preference Is Non-Authoritative

Conformance SHALL prove user/CLI preferences cannot force an ineligible Kernel.

#### Scenario: CLI requests latency

Given fastest candidate is revoked

When latency mode is requested

Then revoked candidate remains excluded.

---

### Requirement: Exploration Eligibility Conformance

Conformance SHALL prove exploration only includes already eligible candidates.

#### Scenario: Unqualified candidate

Given exploration enabled

When candidate lacks required qualification

Then it is not explored.

---

### Requirement: Provider Global Selection Boundary Conformance

Conformance SHALL prove Provider cannot decide cross-Provider selection.

#### Scenario: Provider advertises high score

Given Runtime policy rejects it

When selection runs

Then Provider cannot override decision.

---

### Requirement: Selection Explainability Conformance

Conformance SHALL validate selection reasoning is available and redacted.

#### Scenario: No eligible candidates

Given every candidate is excluded

When diagnostics are produced

Then structured exclusion reasons are available without native handles.