## ADDED Requirements

### Requirement: Production Inference API Rejects Logits Injection
RuntimeInferenceApi SHALL NOT expose a production API that lets callers provide logits, substitute model execution, or install a per-request forward callback.

#### Scenario: Caller cannot inject logits
- **WHEN** production callers build a generation request
- **THEN** the request contains prompt/session/model parameters but no logits array or forward callback.

#### Scenario: Synthetic support is test-only
- **WHEN** tests require synthetic logits
- **THEN** the support is gated by `#[cfg(test)]` or an explicitly non-production conformance feature.

### Requirement: Runtime Owns Model Execution Chain
RuntimeInferenceApi SHALL own the chain from model reference or loaded ModelInstance to graph planning, PreparedExecutionPlan execution, logits production, Sampling, streaming, and cleanup.

#### Scenario: Generation fails without model instance
- **WHEN** generation cannot resolve or access a valid ready ModelInstance
- **THEN** RuntimeInferenceApi rejects generation with a structured model error.
