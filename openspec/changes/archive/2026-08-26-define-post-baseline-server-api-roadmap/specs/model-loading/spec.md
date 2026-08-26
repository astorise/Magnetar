## ADDED Requirements

### Requirement: Server Model Loading Uses Model Loading Contract

Server model load operations SHALL use Model Loading Contract.

#### Scenario: Server load model

Given server receives model load request

When Runtime processes it

Then artifact trust, integrity, component compatibility, memory, provider, and
policy validation run.

---

### Requirement: Server Does Not Load From Arbitrary Paths

Server model load operations SHALL not load arbitrary filesystem paths unless
wrapped in authorized source contracts.

#### Scenario: Arbitrary path

Given request includes raw filesystem path

When server validates it

Then request is rejected or converted only through authorized source contract.