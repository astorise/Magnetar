## ADDED Requirements

### Requirement: Qwen Adapter Validation Uses Target Modules

Adapter Loading SHALL use Qwen target module metadata for Qwen-compatible Model
Instances.

#### Scenario: LoRA target unavailable

Given LoRA adapter targets unavailable module

When Runtime validates against Qwen metadata

Then adapter validation fails.

---

### Requirement: Qwen Baseline May Reject Adapter Execution

Qwen baseline adapter activation support status SHALL be explicit; the baseline
MAY reject adapter activation if adapter overlay or merge graph is not
implemented.

#### Scenario: Adapter unsupported

Given adapter activation is requested

And Qwen baseline lacks adapter graph support

Then Runtime rejects activation explicitly.