## ADDED Requirements

### Requirement: Runtime Owns Generation Contract

Runtime SHALL expose generation through a stable token-based Generation
Contract.

#### Scenario: Generate from tokens

Given input tokens are validated

When a caller requests inference output

Then Runtime uses Generation Contract to produce output tokens.

---

### Requirement: Runtime Separates Generation From Tokenizer

Runtime SHALL keep tokenization and generation as separate stages.

#### Scenario: Decode output

Given generation completes with output token IDs

When text is requested

Then Runtime uses Tokenizer decode.

---

### Requirement: Runtime Validates Generation Before Execution

Runtime SHALL validate model availability, tokenizer compatibility, input
tokens, context limits, parameters, stop conditions, memory admission, and policy
before execution.

#### Scenario: Invalid request

Given generation parameters are invalid

When Runtime receives the request

Then execution does not begin.

---

### Requirement: Runtime Uses Memory Manager For Generation Admission

Runtime SHALL request Memory Manager admission before memory-dependent
generation execution.

#### Scenario: KV cache placeholder memory unavailable

Given generation requires future KV cache memory

And Memory Manager rejects admission

When generation is requested

Then Runtime rejects, queues, or retries according to policy.

---

### Requirement: Runtime Resolves Providers For Generation Internally

Runtime SHALL resolve Providers and Devices internally for generation execution.

Generation request inputs SHALL not directly select Providers or Devices.

#### Scenario: Provider unavailable

Given no compatible Provider is available

When generation execution is planned

Then Runtime reports provider-resolution-failed.

---

### Requirement: Runtime Supports Streaming Generation

Runtime SHALL support streaming token events from Generation.

#### Scenario: Streaming response

Given streaming mode is enabled

When tokens are generated

Then Runtime emits ordered token events and may integrate tokenizer streaming
decode for text chunks.

---

### Requirement: Runtime Supports Generation Cancellation

Runtime SHALL support cancellation of generation requests according to policy and
Provider capabilities.

#### Scenario: Cancel request

Given a generation request is active

When cancellation is requested

Then Runtime stops generation or reports cancellation unsupported according to
the execution path.

---

### Requirement: Runtime Does Not Log Raw Prompts By Default

Runtime observability SHALL not log raw prompts during generation by default.

#### Scenario: Generation observed

Given a generation request is observed

When telemetry is emitted

Then prompt text is omitted unless explicit policy enables prompt logging.