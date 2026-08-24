## ADDED Requirements

### Requirement: Runtime Owns Sampling Contract

Runtime SHALL expose Sampling through a stable inference Runtime contract.

#### Scenario: Runtime sampling

Given logits are produced by model execution

When next token selection is needed

Then Runtime uses the Sampling Contract.

---

### Requirement: Runtime Applies Sampling Policy

Runtime SHALL validate Sampling parameters and apply Runtime or session policy
before token selection.

#### Scenario: Disallowed probability metadata

Given a request asks for token probabilities

And policy disallows probability metadata

When Runtime validates Sampling

Then the request is rejected or probability metadata is omitted according to
policy.

---

### Requirement: Runtime Controls Logits Materialization

Runtime SHALL control whether logits may be materialized to host memory.

#### Scenario: Materialization forbidden

Given logits reside on Device memory

And policy forbids host materialization

When Sampling requires host materialization

Then Runtime rejects the request or selects another compatible sampling path.

---

### Requirement: Runtime Preserves Resource Affinity During Sampling

Runtime SHALL preserve Resource Affinity for Provider-owned or Device-resident
logits.

#### Scenario: Device logits

Given logits are bound to Device A

When Sampling is planned

Then Runtime selects a compatible path, explicitly moves data if authorized, or
rejects sampling.

---

### Requirement: Runtime Does Not Expose Raw Logits By Default

Runtime SHALL not expose raw logits to clients or Components by default.

#### Scenario: Logprobs disabled

Given a client requests raw logits

When policy does not allow it

Then Runtime denies the request.

---

### Requirement: Runtime Observes Sampling

Runtime SHALL support Sampling observations without logging raw logits or raw
prompts by default.

#### Scenario: Sampling failed

Given Sampling fails because no eligible token remains

When Runtime emits observability

Then it records a stable no-eligible-token category.
