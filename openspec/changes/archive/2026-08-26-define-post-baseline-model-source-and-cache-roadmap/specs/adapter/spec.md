## ADDED Requirements

### Requirement: Adapter Artifacts May Use Source Cache

Adapter Artifacts MAY be resolved through source/cache workflow, and Runtime SHALL validate the adapter cache entry before use.

#### Scenario: Cached LoRA adapter

Given LoRA adapter is cached

When adapter activation is requested

Then Runtime validates adapter cache entry, base model compatibility, and policy.

---

### Requirement: Adapter Cache Does Not Activate Adapter

Cached adapter presence SHALL not activate adapter.

#### Scenario: Adapter cache hit

Given adapter is cached

When Model Instance is loaded

Then adapter activation remains explicit.