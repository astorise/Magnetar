## ADDED Requirements

### Requirement: Runtime Remains Inference-Only Under Server

Runtime SHALL remain inference-only when called from Server API.

#### Scenario: Server asks Runtime to read file

Given server request attempts Runtime file access

When Runtime validates it

Then request is rejected.

---

### Requirement: Runtime Policy Still Applies Under Server

Server authorization SHALL not bypass Runtime admission, memory, model loading,
session, generation, provider, or policy constraints.

#### Scenario: Server authorized request

Given server authorizes user

When Runtime rejects due to memory pressure

Then request fails.