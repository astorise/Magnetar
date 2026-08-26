## ADDED Requirements

### Requirement: Inference API Uses Source Resolution Safely

Runtime Inference API MAY resolve ModelRefs through authorized source/cache contracts, and resolution SHALL validate the resulting artifact before it is used.

#### Scenario: Cached model reference

Given inference request references cached model

When Runtime resolves it

Then cache entry is validated before loading.

---

### Requirement: Inference API Does Not Gain Download Authority

Runtime Inference API SHALL not perform arbitrary model downloads during
inference.

#### Scenario: Download requested in inference

Given inference request asks Runtime to download model from arbitrary URL

When validation runs

Then Runtime rejects it or delegates only through authorized source contract.