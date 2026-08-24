## ADDED Requirements
### Requirement: Adapter Validation Uses Model Component Metadata

Adapter Loading SHALL use Model Component target module metadata where available.

#### Scenario: LoRA target validation

Given adapter targets `q_proj`

When Runtime validates the adapter

Then it checks target module metadata from the compatible Model Component.

---

### Requirement: Adapter Graph Changes May Be Produced By Model Component

Model Component SHALL define adapter overlay or merge graph production metadata where supported.

#### Scenario: LoRA overlay graph

Given LoRA adapter is active

When Runtime requests graph production

Then Model Component may emit explicit adapter overlay graph metadata.