## ADDED Requirements

### Requirement: Generation Uses Sampling For Next Token

Generation SHALL use Sampling or equivalent Runtime-owned selection logic to
choose the next token from logits.

#### Scenario: Decode step

Given model forward returns logits

When decode step needs a token

Then Generation invokes Sampling to select the next token.

---

### Requirement: Generation Owns Stop Conditions

Generation SHALL remain responsible for stop condition evaluation.

Sampling may mask or select tokens according to policy, but Generation decides
whether generation stops.

#### Scenario: EOS selected

Given Sampling selects EOS

When Generation receives it

Then Generation applies EOS stop policy.

---

### Requirement: Generation Owns KV Cache Advance

Generation SHALL remain responsible for updating generation state and KV cache
after Sampling returns a token.

#### Scenario: Token selected

Given Sampling selects a token

When Generation accepts it

Then Generation updates state and appends KV cache where applicable.

---

### Requirement: Generation Validates Sampling Parameters

Generation request validation SHALL include Sampling parameter validation or
delegation to the Sampling Contract.

#### Scenario: Invalid top-p

Given generation request includes invalid top-p

When Runtime validates the request

Then validation fails before decode execution.