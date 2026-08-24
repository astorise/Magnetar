## ADDED Requirements

### Requirement: Qwen KV Cache Layout Metadata

KV cache layout metadata for a Qwen Model Instance SHALL be derivable from Qwen
Model Component metadata, including layer count, KV head count, attention head
count, head dimension, cache dtype, sequence dimension, batch dimension,
position encoding behavior, append behavior, and layout preference.

#### Scenario: Qwen cache metadata requested

Given a ready Qwen Model Instance

When Runtime prepares KV cache for prefill

Then layer count, KV head count, head dimension, cache dtype, and layout
preference are derived from Qwen Model Component metadata.

---

### Requirement: Qwen Baseline Supports Non-Paged KV Cache First

Paged KV cache support status for the Qwen baseline SHALL be explicit and MAY
remain a placeholder; the baseline MAY support only non-paged layout in the
first implementation.

#### Scenario: Paged cache unavailable

Given Qwen baseline does not implement paged KV cache

When Runtime queries paged cache support status

Then Runtime reports paged cache as unsupported rather than silently
substituting a different layout.

---

### Requirement: Qwen KV Cache Compatibility Includes Component Metadata

KV cache compatibility validation for a Qwen Model Instance SHALL include Qwen
Model Component version and architecture metadata in addition to base KV Cache
Compatibility requirements.

#### Scenario: Qwen component version mismatch

Given a KV cache was created under Qwen Model Component version 1

When generation resumes under an incompatible Qwen Model Component version 2

Then Runtime rejects reuse with cache-model-mismatch or a Qwen-specific
incompatibility error.
