## ADDED Requirements
### Requirement: Reference CPU Provider Satisfies First Profile

Reference CPU Provider SHALL execute every mandatory first-profile Kernel.

#### Scenario: Complete fixture generation

Given no other Provider is installed

When Qwen fixture executes

Then Reference CPU Provider can complete model forward/generation.

### Requirement: Reference CPU Uses Logical CPU Device

Provider SHALL expose a conformant logical CPU Device through normal Provider
contract.

#### Scenario: Provider discovery

Given first profile initializes

When Runtime resolves Device

Then CPU Device participates through normal Device metadata/readiness.

### Requirement: Synchronous Execution Is Conformant

First profile SHALL allow Reference CPU Provider to execute first-profile Kernel
synchronously.

#### Scenario: Kernel submitted

Given execution completes inline

When submission returns

Then logical CompletionToken is completed.

### Requirement: Provider Does Not Expose Direct Qwen API

Reference CPU Provider SHALL expose Kernel execution capabilities, not a special
Qwen-model forward API.

#### Scenario: Provider surface inspected

Given Qwen fixture exists

Then Provider remains architecture-neutral at model-family level.
