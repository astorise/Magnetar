## ADDED Requirements

### Requirement: Runtime Owns Prefix Cache

Runtime SHALL own Prefix Cache lookup, insertion, validation, sharing,
invalidation, eviction, and cleanup.

#### Scenario: Runtime lookup

Given generation requests prefix reuse

When Runtime performs lookup

Then Runtime returns a structured Prefix Cache result.

---

### Requirement: Runtime Applies Prefix Cache Policy

Runtime SHALL apply sharing, privacy, session, model, tokenizer, Resource
Affinity, memory, and lifecycle policy before reuse.

#### Scenario: Policy denies reuse

Given a matching prefix entry exists

But policy denies sharing

When Runtime validates reuse

Then Runtime does not reuse the entry.

---

### Requirement: Runtime Protects Prefix Privacy

Runtime SHALL not expose raw prompt text, raw token sequences, or raw backing KV
cache contents through Prefix Cache APIs by default.

#### Scenario: Prefix diagnostics

Given diagnostics are requested for a prefix entry

When Runtime responds

Then only redacted metadata is returned.

---

### Requirement: Runtime Observes Prefix Cache

Runtime SHALL define observations for Prefix Cache lookup, hit, miss, invalidation,
eviction, and policy denial.

#### Scenario: Prefix miss

Given Prefix Cache lookup misses

When observability records the event

Then Runtime emits a prefix-cache-miss category.
