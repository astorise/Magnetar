## ADDED Requirements

### Requirement: D07 Operational compatible prefix cache

Prefix reuse SHALL use versioned tokenized fingerprints including model, tokenizer, template, adapter, position, attention configuration, dtype/layout and sharing scope; Radix/Trie choices SHALL be benchmarked.

#### Scenario: D07 Qualified execution

- **GIVEN** an explicitly selected profile with its required dependencies and authorized resources
- **WHEN** compatible authorized sessions reuse a sealed prefix
- **THEN** prefill work is skipped only for the validated prefix and retained KV pages remain owned

#### Scenario: D07 Ineligible or failed execution

- **WHEN** tokens match but adapter, tenant scope or model identity differs
- **THEN** the lookup is a safe miss and no cross-session state is exposed

### Requirement: E07 Measured prefix savings in batches

Batch scheduling SHALL integrate compatible prefix reuse and report actual avoided prefill tokens/time separately from cache lookups and prefetch hits.

#### Scenario: E07 Qualified execution

- **GIVEN** an explicitly selected profile with its required dependencies and authorized resources
- **WHEN** a batch contains requests with authorized common prefixes
- **THEN** results match uncached generation and useful saved-prefill work is reported

#### Scenario: E07 Ineligible or failed execution

- **WHEN** prefix sharing increases memory pressure or crosses a privacy scope
- **THEN** admission uses a safe miss or eviction without counting lookup hits as performance savings

### Requirement: production-prefix-reuse promotion gate

This profile SHALL remain proposed, contract-only or experimental until all covered action gates pass on an appropriate real checkpoint and real target hardware. Runtime SHALL preserve trust, authoritative ResourceAffinity and MemoryManager admission before optimization ranking. Components SHALL NOT select native implementations or receive native handles. New paths SHALL have a disable control and a qualified fallback or explicit Unsupported outcome, and SHALL NOT grant ambient external authority.

#### Scenario: Evidence is incomplete

- **WHEN** only a contract, fixture, simulated Device, discovery result or hardware skip is available
- **THEN** the support matrix does not label this profile production-validated or claim a measured end-to-end speedup

#### Scenario: Performance conflicts with eligibility

- **WHEN** a proposed faster path violates trust, affinity, memory admission or Component authority
- **THEN** it is rejected before dispatch regardless of measured speed
