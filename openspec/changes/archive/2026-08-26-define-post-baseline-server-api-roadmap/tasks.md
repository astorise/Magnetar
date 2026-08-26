# Tasks

## 1. Roadmap Scope

- [x] Define Server API roadmap.
- [x] Document serve mode as Runtime Inference API facade.
- [x] Document server versus Runtime boundary.
- [x] Document server versus CLI boundary.
- [x] Document non-goals.

## 2. Serve Mode Ownership

- [x] Allow serve mode in CLI, companion binary, or server crate.
- [x] Require Runtime Inference API usage.
- [x] Prevent direct Provider calls.
- [x] Prevent direct Kernel calls.
- [x] Prevent direct Model Component calls.
- [x] Add serve ownership tests.

## 3. Initial Endpoint Scope

- [x] Define health endpoint placeholder.
- [x] Define readiness endpoint placeholder.
- [x] Define model list endpoint placeholder.
- [x] Define model inspect endpoint placeholder.
- [x] Define session create endpoint placeholder.
- [x] Define session close endpoint placeholder.
- [x] Define generation endpoint placeholder.
- [x] Define streaming generation endpoint placeholder.
- [x] Define cancellation endpoint placeholder.
- [x] Define diagnostics endpoint placeholder.
- [x] Document endpoint names as illustrative.

## 4. Health And Readiness

- [x] Define health semantics.
- [x] Ensure health does not imply Runtime readiness.
- [x] Ensure health does not imply model availability.
- [x] Define readiness semantics.
- [x] Include Runtime readiness summary.
- [x] Include Provider readiness summary where safe.
- [x] Include memory/admission summary where safe.
- [x] Redact responses by default.
- [x] Add health/readiness tests.

## 5. Model Endpoints

- [x] Define model list behavior.
- [x] Define model inspect behavior.
- [x] Define loaded instance inspect behavior.
- [x] Define model load request placeholder.
- [x] Define model unload request placeholder.
- [x] Preserve Model Source validation.
- [x] Preserve Model Artifact validation.
- [x] Preserve Model Loading validation.
- [x] Preserve trust/integrity checks.
- [x] Add model endpoint tests.

## 6. Session Endpoints

- [x] Define session create behavior.
- [x] Define session inspect behavior.
- [x] Define session close behavior.
- [x] Keep Runtime session inference-scoped.
- [x] Keep server connection state separate.
- [x] Prevent workspace/Git/tool/secret state in Runtime sessions.
- [x] Add session endpoint tests.

## 7. Generation Endpoint

- [x] Define generation request placeholder.
- [x] Accept model or session reference.
- [x] Accept prompt text.
- [x] Accept chat messages.
- [x] Accept tokenized input where policy allows.
- [x] Accept generation parameters.
- [x] Accept sampling parameters.
- [x] Accept stop conditions.
- [x] Accept streaming flag.
- [x] Accept timeout.
- [x] Call Runtime Inference API.
- [x] Prevent tool execution from model output.
- [x] Add generation endpoint tests.

## 8. Streaming Endpoint

- [x] Define streaming transport options.
- [x] Support SSE placeholder.
- [x] Support chunked HTTP placeholder.
- [x] Support WebSocket placeholder.
- [x] Preserve Runtime event ordering.
- [x] Prevent raw logits exposure by default.
- [x] Prevent raw tensor exposure by default.
- [x] Prevent raw KV cache exposure by default.
- [x] Add streaming endpoint tests.

## 9. Cancellation Endpoint

- [x] Define cancellation request.
- [x] Call Runtime cancellation.
- [x] Close server transport stream where applicable.
- [x] Preserve Runtime cancellation limitations.
- [x] Add cancellation endpoint tests.

## 10. Diagnostics Endpoint

- [x] Define server diagnostics.
- [x] Include server status.
- [x] Include Runtime readiness.
- [x] Include Provider readiness summary.
- [x] Include memory pressure summary.
- [x] Include queue/admission status.
- [x] Include loaded model summary.
- [x] Include session summary.
- [x] Include cache summary.
- [x] Include recent structured errors.
- [x] Redact by default.
- [x] Add diagnostics endpoint tests.

## 11. OpenAI-Compatible Facade Placeholder

- [x] Define facade as optional post-baseline layer.
- [x] Map compatible requests to Runtime Inference API.
- [x] Reject unsupported fields explicitly where needed.
- [x] Prevent tool-call fields from executing tools.
- [x] Preserve Runtime semantics.
- [x] Add facade placeholder tests.

## 12. Authentication Boundary

- [x] Define authentication as server concern.
- [x] Prevent Runtime from receiving ambient credentials.
- [x] Prevent secrets in Runtime diagnostics.
- [x] Prevent credentials in observability.
- [x] Add auth boundary tests.

## 13. Authorization Boundary

- [x] Define model authorization.
- [x] Define source authorization.
- [x] Define session authorization.
- [x] Define generation authorization.
- [x] Define streaming authorization.
- [x] Define diagnostics authorization.
- [x] Define cache inspection authorization.
- [x] Define model loading/unloading authorization.
- [x] Ensure authorization does not bypass Runtime policy.
- [x] Add authorization tests.

## 14. Admission And Rate Policy

- [x] Define concurrent request limit placeholder.
- [x] Define queue limit placeholder.
- [x] Define max token limit placeholder.
- [x] Define max session limit placeholder.
- [x] Define max loaded model limit placeholder.
- [x] Define memory budget placeholder.
- [x] Define streaming connection limit placeholder.
- [x] Define request body size limit.
- [x] Define prompt size limit.
- [x] Preserve Runtime admission.
- [x] Add admission/rate tests.

## 15. Source And Cache Boundary

- [x] Allow model references from authorized cache.
- [x] Allow authorized source references.
- [x] Prevent arbitrary downloads during generation.
- [x] Keep model pull/import as future work.
- [x] Preserve source/cache contracts.
- [x] Add source/cache server tests.

## 16. Filesystem Boundary

- [x] Prevent arbitrary filesystem access.
- [x] Prevent generation endpoints from reading server files.
- [x] Require authorized source contracts for future import.
- [x] Add filesystem boundary tests.

## 17. Tool/Shell/Git Boundary

- [x] Prevent tool execution.
- [x] Prevent shell execution.
- [x] Prevent process execution.
- [x] Prevent Git execution.
- [x] Document agent server as separate future scope.
- [x] Add boundary tests.

## 18. Error Model

- [x] Define server-api-unavailable error.
- [x] Define server-request-invalid error.
- [x] Define server-request-too-large error.
- [x] Define server-authentication-required error.
- [x] Define server-authentication-failed error.
- [x] Define server-authorization-denied error.
- [x] Define server-rate-limited error.
- [x] Define server-admission-rejected error.
- [x] Define server-stream-unavailable error.
- [x] Define server-stream-interrupted error.
- [x] Define server-cancellation-failed error.
- [x] Define server-model-not-found error.
- [x] Define server-model-load-failed error.
- [x] Define server-session-not-found error.
- [x] Define server-generation-failed error.
- [x] Define server-diagnostics-redacted status.
- [x] Define server-source-policy-denied error.
- [x] Define server-cache-policy-denied error.
- [x] Define server-boundary-violation error.
- [x] Define internal-server-api error.
- [x] Preserve Runtime structured error cause.

## 19. Observability

- [x] Emit server started observation.
- [x] Emit server stopped observation.
- [x] Emit request received observation.
- [x] Emit request rejected observation.
- [x] Emit request authorized observation.
- [x] Emit Runtime request submitted observation.
- [x] Emit stream opened observation.
- [x] Emit stream closed observation.
- [x] Emit stream interrupted observation.
- [x] Emit generation completed observation.
- [x] Emit generation failed observation.
- [x] Emit cancellation requested observation.
- [x] Emit diagnostics requested observation.
- [x] Emit model endpoint used observation.
- [x] Emit session endpoint used observation.
- [x] Emit rate limit hit observation.
- [x] Emit admission rejected observation.
- [x] Emit boundary violation detected observation.
- [x] Verify default redaction.

## 20. Conformance

- [x] Validate server uses Runtime Inference API.
- [x] Validate health does not imply readiness.
- [x] Validate readiness redaction.
- [x] Validate generation endpoint uses Runtime Generation.
- [x] Validate streaming order.
- [x] Validate cancellation calls Runtime cancellation.
- [x] Validate diagnostics redaction.
- [x] Validate model endpoints do not bypass loading.
- [x] Validate server does not execute tools.
- [x] Validate server does not execute shell.
- [x] Validate server does not execute Git.
- [x] Validate server does not read arbitrary files.
- [x] Validate server does not download during generation.
- [x] Validate Runtime errors are preserved.

## 21. Documentation

- [x] Document Server API roadmap.
- [x] Document serve mode ownership.
- [x] Document endpoint scope.
- [x] Document health/readiness semantics.
- [x] Document model endpoints.
- [x] Document session endpoints.
- [x] Document generation endpoint.
- [x] Document streaming endpoint.
- [x] Document cancellation endpoint.
- [x] Document diagnostics endpoint.
- [x] Document OpenAI-compatible facade placeholder.
- [x] Document auth boundary.
- [x] Document admission/rate policy.
- [x] Document source/cache boundary.
- [x] Document filesystem/tool/shell/Git boundary.
- [x] Document non-goals.

## 22. Final Validation

- [x] Run OpenSpec validation.
- [x] Verify server roadmap does not bypass Runtime Inference API.
- [x] Verify server roadmap does not add agent/tool runtime.
- [x] Verify server roadmap preserves redaction.
- [x] Verify server roadmap preserves source/cache boundaries.