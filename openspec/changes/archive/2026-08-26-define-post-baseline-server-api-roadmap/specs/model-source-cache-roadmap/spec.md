## ADDED Requirements

### Requirement: Server Uses Authorized Source Cache Contracts

Server API SHALL use authorized source/cache contracts for model references.

#### Scenario: Cached model

Given generation references cached model

When server resolves it

Then cache validation and Model Loading still run.

---

### Requirement: Server Does Not Download During Generation

Server generation SHALL not perform arbitrary model downloads.

#### Scenario: Remote URL

Given generation request includes remote model URL

When server validates it

Then request is rejected or routed through authorized source policy outside
generation path.