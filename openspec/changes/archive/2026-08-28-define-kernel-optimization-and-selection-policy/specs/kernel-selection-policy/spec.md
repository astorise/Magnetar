## ADDED Requirements

### Requirement: Filter Before Rank

Kernel selection SHALL apply hard eligibility constraints before optimization
ranking.

#### Scenario: Fast untrusted Kernel

Given candidate A is fastest but trust policy denies it

And candidate B is slower but trusted and qualified

When selection runs

Then candidate A is excluded before ranking and candidate B may be selected.

---

### Requirement: Optimization Cannot Restore Ineligible Candidate

A ranking score SHALL NOT make excluded candidate eligible.

#### Scenario: Ineligible candidate has highest score

Given candidate fails Resource Affinity

When performance scoring gives it highest score

Then it remains excluded.

---

### Requirement: Explicit Optimization Profile

Kernel selection SHALL use explicit optimization policy/profile.

#### Scenario: Latency mode

Given profile is latency

When multiple candidates are eligible

Then latency evidence may prioritize them subject to hard constraints.

---

### Requirement: Deterministic Profile

Deterministic profile SHALL reject candidates that cannot satisfy required
determinism.

#### Scenario: Fast nondeterministic kernel

Given candidate is fastest but nondeterministic

When deterministic profile is active

Then candidate is excluded.

---

### Requirement: Stable Tie-Breaking

Selection SHALL use deterministic tie-breaking.

#### Scenario: Equal candidates

Given two candidates have equal policy scores

When selection repeats with identical inputs

Then the same candidate is selected.

---

### Requirement: Benchmark Evidence Compatibility

Performance evidence SHALL only be used when compatible with current workload
and target context.

#### Scenario: Wrong shape benchmark

Given benchmark covers sequence length 128

When current sequence length is 8192 and evidence is not compatible

Then benchmark is not treated as authoritative.

---

### Requirement: Missing Metrics Are Explicit

Missing optimization metrics SHALL have explicit policy behavior.

#### Scenario: Energy metric absent

Given energy profile is requested

And no reliable energy metric exists

When ranking occurs

Then Runtime follows declared missing-metric policy rather than inventing a
value.

---

### Requirement: Selection Hysteresis

When hysteresis is enabled, promotion SHALL require benefit exceeding the configured threshold; Runtime SHOULD support hysteresis to avoid insignificant Kernel churn.

#### Scenario: Candidate is one percent faster

Given active kernel remains eligible

And replacement benefit is below promotion threshold

When selection is evaluated

Then active kernel may be retained.

---

### Requirement: Selection And Promotion Are Distinct

The highest-ranked candidate SHALL NOT necessarily become active until promotion
policy approves it.

#### Scenario: Candidate ranks first

Given candidate ranks first

But promotion threshold/canary policy has not been satisfied

When selection completes

Then candidate may remain non-active.

---

### Requirement: Fallback Is Explicit

Kernel fallback SHALL follow explicit policy.

#### Scenario: Preferred provider unavailable

Given preferred candidate becomes unavailable

When fallback policy permits Reference CPU

Then explicit fallback may occur.

---

### Requirement: No Hidden Cross-Provider Fallback

Fallback SHALL not silently move resources between Providers.

#### Scenario: CUDA tensor and CPU fallback

Given tensor is CUDA-affine

And host staging is forbidden

When CUDA Kernel unavailable

Then Runtime fails instead of silently copying to CPU.

---

### Requirement: Model Component Does Not Choose Kernel

Model Component SHALL not select concrete Kernel implementation.

#### Scenario: Qwen graph

Given Qwen Component emits Attention Operator

When execution is planned

Then Runtime selection policy chooses implementation.

---

### Requirement: Session Preferences Are Non-Authoritative

Session/user preferences MAY influence optimization but SHALL not override
Runtime eligibility constraints.

#### Scenario: User requests fastest

Given fastest candidate is untrusted

When user requests latency profile

Then untrusted candidate remains excluded.

---

### Requirement: Policy Is Versioned

Kernel selection policy SHALL have explicit version.

#### Scenario: Ranking formula changes

Given policy implementation changes materially

When selection metadata is emitted

Then policy version identifies the decision model used.

---

### Requirement: Selection Is Explainable

Any explanation of candidate eligibility and ranking SHALL exclude native handles and raw tensor data; Runtime SHOULD produce such a redacted explanation.

#### Scenario: Candidate excluded

Given kernel is excluded due to memory

When diagnostics are requested

Then reason reports memory infeasibility without native handles.