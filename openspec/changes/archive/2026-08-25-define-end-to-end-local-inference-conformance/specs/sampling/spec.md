## ADDED Requirements

### Requirement: E2E Uses Sampling Contract

E2E conformance SHALL validate that token selection occurs through Sampling
Contract.

#### Scenario: Greedy sampling

Given fixture logits are produced

When next token is selected

Then Sampling Contract returns deterministic selected token.

---

### Requirement: E2E Does Not Require Provider-Assisted Sampling

The first E2E local inference suite SHALL not require Provider-assisted
sampling.

#### Scenario: CPU fixture generation

Given Reference CPU produces logits

When next token is needed

Then Runtime Sampling selects token without Provider-assisted sampling.