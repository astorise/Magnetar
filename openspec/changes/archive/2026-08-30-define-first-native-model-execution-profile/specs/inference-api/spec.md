## ADDED Requirements
### Requirement: Inference API Accepts Model And Generation Intent

RuntimeInferenceApi SHALL expose model/session/generation operations without
requiring model execution implementation from caller.

#### Scenario: Text generation

Given caller selects loaded Qwen fixture and supplies prompt

When generate is invoked

Then Runtime performs tokenization/model execution/sampling.

### Requirement: Caller Cannot Inject Ordinary Forward Function

RuntimeInferenceApi SHALL not expose caller-supplied forward/logits callback as
normal inference authority.

#### Scenario: Caller tries to supply logits closure

Given normal generation API

When request is constructed

Then no such callback is required or authoritative.

### Requirement: Cancellation Is Runtime Owned

RuntimeInferenceApi SHALL allow caller-requested cancellation without exposing
or delegating Provider execution details to the caller.

#### Scenario: Generation cancelled

Given request is active

When caller cancels

Then Runtime propagates cancellation through Session/execution contracts.

### Requirement: Native Handles Are Not Exposed

RuntimeInferenceApi SHALL not expose Kernel, Provider, Tensor native pointer, or
ExecutionStream native handle.

#### Scenario: Client receives token

Given internal execution uses prepared Kernels

When output is returned

Then only high-level inference data/status is exposed.
