## MODIFIED Requirements

### Requirement: Model Artifact Trust

A Model Artifact SHALL be validated and trusted before loading, and that trust evaluation SHALL be sourced from a trust policy configured once for the Runtime instance performing the load, not from a policy the loading caller supplies for that call.

A Model Artifact SHALL NOT declare itself trusted.

A caller SHALL NOT be able to substitute a different trust policy than the one the Runtime instance was configured with in order to obtain a favorable trust decision for a specific load.

#### Scenario: Manifest claims trusted

Given a Model Artifact manifest claims the model is trusted

When Runtime evaluates trust

Then Runtime uses trust policy instead.

---

#### Scenario: Caller-supplied trust policy is not honored

Given a Runtime instance was configured with a trust policy that does not trust a given artifact's digest

When a loading caller attempts to load that artifact by supplying their own, more permissive trust policy for that call

Then Runtime still evaluates trust using its own configured policy, not the caller-supplied one, and the load is rejected

---
