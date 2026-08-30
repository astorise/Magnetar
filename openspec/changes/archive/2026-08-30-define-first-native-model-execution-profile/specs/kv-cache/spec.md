## ADDED Requirements
### Requirement: First Profile Uses Real KV Cache

Qwen incremental generation SHALL store and reuse K/V state through Runtime KV
Cache contract.

#### Scenario: Decode second generated token

Given prefill and first decode step produced KV

When second decode step executes

Then Attention can consume previous cached K/V.

### Requirement: KV State Is Runtime Owned

KV state SHALL not live in CLI or caller-provided logits callback.

#### Scenario: CLI generates tokens

Given Session is active

When CLI receives output

Then KV remains internal Runtime inference state.

### Requirement: Simple Contiguous KV Layout Is Sufficient

First profile SHALL allow contiguous per-layer K/V storage to satisfy mandatory
KV semantics.

#### Scenario: Paged KV unimplemented

Given contiguous KV correctly supports fixture context

When profile runs

Then lack of paged KV does not fail conformance.

### Requirement: Decode Position Is Incremental

KV append position SHALL progress with actual sequence length.

#### Scenario: Prompt length four

Given first generated token becomes sequence position four

When K/V are appended

Then subsequent decode reads the correct prior range.

### Requirement: KV Bounds Are Enforced

Fixture maximum context SHALL not be exceeded without structured handling.

#### Scenario: Sequence reaches context limit

Given another append is requested

When capacity is insufficient

Then Runtime returns defined limit/error rather than corrupting cache.
