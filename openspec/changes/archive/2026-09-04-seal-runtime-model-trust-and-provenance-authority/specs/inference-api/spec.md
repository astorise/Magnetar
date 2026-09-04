## MODIFIED Requirements

### Requirement: Model Loading API

Runtime Inference API SHALL expose explicit model loading or policy-controlled implicit loading, and the trust decision used to authorize that load SHALL be sourced from the Runtime instance performing it rather than accepted as a loading-call parameter.

#### Scenario: Explicit load

Given a valid model reference

When loading request is submitted

Then Runtime validates artifact, component, memory, provider, device, and policy
contracts before creating a ready Model Instance.

---

#### Scenario: Loading API does not accept a caller-supplied trust decision

Given a loading request for a model artifact

When the Model Loading API call is made

Then the trust decision applied is the one the performing Runtime instance was configured with, and the API surface provides no parameter through which a caller can substitute a different trust decision for that call

---
