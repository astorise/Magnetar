## ADDED Requirements

### Requirement: Prefix Cache Policy Is Exposed Through Inference API

Runtime Inference API SHALL expose Prefix Cache policy inputs without exposing raw prompt text or raw KV cache contents.

#### Scenario: Prefix cache enabled

Given Prefix Cache policy is enabled

When generation prepares prefill

Then Runtime may use Prefix Cache internally and report redacted hit/miss
metadata.

---

### Requirement: Inference API Does Not Expose Prefix Fingerprint Inputs By Default

Raw prompt text and raw token sequences used for Prefix Cache fingerprinting SHALL not be exposed by default.

#### Scenario: Diagnostics requested

Given Prefix Cache miss occurs

When diagnostics are returned

Then raw prompt fingerprint inputs are redacted by default.