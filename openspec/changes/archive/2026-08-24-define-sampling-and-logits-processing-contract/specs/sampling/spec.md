## ADDED Requirements

### Requirement: Sampling Contract

Magnetar SHALL define a Sampling Contract for selecting the next token ID from
logits or equivalent token scores.

#### Scenario: Select next token

Given valid logits and sampling parameters

When Sampling runs

Then it returns a selected token ID or a structured sampling error.

---

### Requirement: Sampling Is Token-Based

Sampling SHALL operate on token IDs and token scores.

Sampling SHALL NOT operate on raw text.

#### Scenario: Text stop sequence

Given a stop condition is textual

When it affects token selection

Then Runtime uses Tokenizer-derived token metadata before Sampling applies token
constraints.

---

### Requirement: Sampling Does Not Decode Text

Sampling SHALL NOT produce decoded text.

#### Scenario: Selected token

Given Sampling selects token ID 42

When output text is needed

Then Runtime uses Tokenizer decode or streaming decode.

---

### Requirement: Sampling Request

Sampling SHALL accept a structured SamplingRequest including logits reference,
vocabulary size, step index, token history where needed, tokenizer metadata,
parameters, processor configuration, RNG metadata, token constraints, policy,
and observability correlation.

#### Scenario: Missing logits

Given a SamplingRequest has no logits or score reference

When validation runs

Then Sampling fails with logits-unavailable.

---

### Requirement: Sampling Result

Sampling SHALL return a structured SamplingResult.

The result SHOULD include selected token ID, selection mode, token rank where
available, probability metadata where policy permits, diagnostics, and updated
RNG state where applicable.

#### Scenario: Probability disabled

Given policy disables probability metadata

When Sampling returns a result

Then selected token ID is returned without probability details.

---

### Requirement: Logits Representation

Raw logits SHALL be represented without exposing raw Provider handles and MAY
use Runtime tensor reference, Provider-owned
tensor reference, Device-resident tensor reference, host score buffer, or test
fixture vector.

Raw Provider handles SHALL NOT be exposed.

#### Scenario: Provider-owned logits

Given logits are Provider-owned

When Sampling consumes them

Then Runtime uses opaque Runtime references and preserves Resource Affinity.

---

### Requirement: Ordered Processor Chain

Sampling SHALL apply logits processors in a deterministic, documented order.

#### Scenario: Processor order

Given banned tokens and top-k are both configured

When Sampling runs

Then processors are applied in the defined order.

---

### Requirement: Processor Authority Is Inference-Scoped

Logits processors SHALL NOT receive filesystem, network, Git, secrets,
workspace, or process authority.

#### Scenario: Component processor

Given a processor is implemented as a Component

When Runtime links it

Then only inference-scoped processor authority is available.

---

### Requirement: Greedy Selection

Greedy selection SHALL choose the highest valid score after logits processing.

#### Scenario: Greedy next token

Given processed logits have token 7 as highest score

When greedy sampling runs

Then token 7 is selected.

---

### Requirement: Temperature Validation

Temperature SHALL be explicit and validated.

Temperature zero behavior SHALL be defined by policy.

#### Scenario: Invalid temperature

Given temperature is invalid

When SamplingRequest is validated

Then Sampling fails with temperature-invalid.

---

### Requirement: Top-K Filtering

Top-k filtering SHALL keep only the highest-k eligible tokens.

#### Scenario: Top-k three

Given top-k is 3

When Sampling filters logits

Then only the three highest eligible tokens remain.

---

### Requirement: Top-P Filtering

Top-p filtering SHALL keep the smallest eligible set whose cumulative
probability reaches the configured threshold.

#### Scenario: Invalid top-p

Given top-p is outside valid range

When validation runs

Then Sampling fails with top-p-invalid.

---

### Requirement: Unsupported Reserved Sampling Modes

Reserved modes such as min-p or typical-p SHALL return structured unsupported
errors when requested but unavailable.

#### Scenario: Typical-p unsupported

Given typical-p is requested

And implementation does not support it

When validation runs

Then Sampling fails with typical-p-unsupported.

---

### Requirement: Penalty Processing

Sampling SHALL support repetition, frequency, and presence penalty metadata
where implemented.

Penalties SHALL use token history.

#### Scenario: Repetition penalty

Given token 10 appears in history

When repetition penalty is applied

Then token 10 score is adjusted according to configured penalty.

---

### Requirement: Banned Tokens

Banned token IDs SHALL be removed from eligibility.

#### Scenario: Banned highest token

Given token 4 has the highest score

And token 4 is banned

When Sampling runs

Then token 4 is not selected.

---

### Requirement: Allowed Tokens

Allowed token IDs, when provided, SHALL restrict eligibility to that set.

#### Scenario: Allowed set

Given allowed tokens are 5 and 6

When Sampling runs

Then only tokens 5 or 6 may be selected.

---

### Requirement: No Eligible Token

If processors remove all eligible tokens, Sampling SHALL fail with a structured
no-eligible-token error.

#### Scenario: All tokens banned

Given every token is banned

When Sampling runs

Then Sampling fails with no-eligible-token.

---

### Requirement: Special Token Policy

Sampling SHALL use Tokenizer special token metadata to apply special token
policy.

#### Scenario: PAD suppressed

Given PAD token is configured as suppressed

When Sampling runs

Then PAD token is not eligible.

---

### Requirement: Minimum Length

Sampling SHALL define minimum length behavior and MAY support it by masking EOS
or stop tokens until minimum generated token count is reached.

#### Scenario: EOS before minimum length

Given minimum generated length is 5

And only 2 tokens have been generated

When Sampling applies policy

Then EOS is masked if policy requires.

---

### Requirement: Deterministic Sampling

Sampling SHALL define deterministic stochastic sampling support and MAY
implement deterministic stochastic sampling.

If deterministic mode is requested but unsupported, Sampling SHALL return a
structured unsupported error.

#### Scenario: Determinism unsupported

Given deterministic stochastic sampling is requested

And implementation cannot guarantee it

When validation runs

Then Sampling fails with deterministic-mode-unsupported.

---

### Requirement: RNG State

Sampling SHALL keep RNG state Runtime-owned when used.

RNG state SHALL be opaque unless policy permits inspection.

#### Scenario: Session RNG state

Given a session carries RNG state

When Sampling runs in that session

Then Sampling may update Runtime-owned RNG state.

---

### Requirement: Probability Metadata Policy

Probability and log probability metadata SHALL be policy-controlled.

#### Scenario: Logprob requested

Given log probabilities are requested

And policy denies them

When Sampling runs

Then Sampling returns probability-metadata-unsupported or policy-denied.

---

### Requirement: Logits Materialization Policy

Runtime SHALL control logits materialization.

Host materialization SHALL respect Memory Manager policy and HostStagingPolicy.

#### Scenario: Host materialization denied

Given logits are Device-resident

And host materialization is denied

When Sampling requires host logits

Then Sampling fails with logits-materialization-denied.

---

### Requirement: Sampling Memory Management

Sampling SHALL use Memory Manager for temporary buffers and SHALL not allocate
unbounded memory outside Runtime policy.

#### Scenario: Top-p workspace

Given top-p filtering needs workspace

When Sampling requests buffers

Then Memory Manager admits or rejects the allocation.

---

### Requirement: Provider-Assisted Sampling

Sampling SHALL keep Provider-assisted sampling under Runtime Resolution when it
is used.

Provider-assisted sampling SHALL be selected by Runtime Resolution and SHALL
preserve Resource Affinity.

#### Scenario: Device-resident logits

Given logits reside on Device A

When Provider-assisted sampling is available on Device A

Then Runtime may execute sampling without host materialization.

---

### Requirement: Sampling Does Not Select Provider Or Device Directly

Sampling requests SHALL NOT contain authoritative Provider or Device selection.

#### Scenario: Request pins Provider

Given a SamplingRequest attempts to select Provider `cuda`

When Runtime validates it

Then the request is rejected or the field is ignored as non-authoritative policy
metadata.

---

### Requirement: Sampling Does Not Advance KV Cache

Sampling SHALL not mutate or advance KV cache state.

#### Scenario: Token selected

Given Sampling selects token ID 8

When the token is returned

Then Generation remains responsible for state update and KV cache append.

---

### Requirement: Sampling Error Categories

Sampling failures SHALL use structured error categories.

#### Scenario: Vocabulary mismatch

Given logits length does not match tokenizer vocabulary size

When Sampling validates input

Then Sampling fails with vocabulary-mismatch.

---

### Requirement: Sampling Observability

Runtime SHALL support Sampling observations.

Observability SHALL not log raw logits or raw prompt text by default.

#### Scenario: Token selected observation

Given Sampling selects a token

When observability records the event

Then it may include redacted metadata such as selection mode and step index.
