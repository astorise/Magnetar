## ADDED Requirements

### Requirement: Inference API Uses Model Components Through Runtime

Runtime Inference API SHALL resolve and use Model Components through Runtime validation.

#### Scenario: Qwen request

Given model reference resolves to Qwen-compatible artifact

When inference begins

Then Runtime resolves compatible Model Component before graph production.

---

### Requirement: Inference API Does Not Grant Component Extra Authority

Calling inference through Runtime API SHALL not grant Model Components additional authority.

#### Scenario: Component asks filesystem

Given Model Component attempts filesystem access during API request

When Runtime authorizes it

Then access is denied.