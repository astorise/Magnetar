# Tasks

## 1. Session Scope

- [x] Define Inference Session as Runtime-owned inference context.
- [x] Document session versus Model Artifact.
- [x] Document session versus Model Instance.
- [x] Document session versus KV cache.
- [x] Document session versus client conversation.
- [x] Document session versus agent memory.
- [x] Document session versus workspace/tool state.

## 2. Session Module

- [x] Create first-class `session` module or equivalent.
- [x] Export canonical session types from crate root.
- [x] Keep session platform-neutral.
- [x] Keep session independent from direct Provider selection.
- [x] Add module-level documentation.

## 3. Session Identity

- [x] Define InferenceSessionId.
- [x] Ensure session ID is Runtime-issued.
- [x] Ensure session ID is opaque.
- [x] Ensure session ID does not encode Provider handles.
- [x] Ensure session ID does not encode Device handles.
- [x] Ensure session ID does not encode memory pointers.
- [x] Ensure session ID alone does not grant authority.
- [x] Add session identity tests.

## 4. Session Lifecycle

- [x] Define creating state.
- [x] Define ready state.
- [x] Define active state.
- [x] Define idle state.
- [x] Define draining state.
- [x] Define cancelled state.
- [x] Define failed state.
- [x] Define closed state.
- [x] Define expired state.
- [x] Define allowed transitions.
- [x] Add lifecycle transition tests.

## 5. Session Creation

- [x] Define session creation request.
- [x] Validate model availability.
- [x] Validate tokenizer compatibility.
- [x] Validate generation defaults.
- [x] Validate Runtime policy.
- [x] Validate memory limits.
- [x] Validate allowed Capabilities.
- [x] Validate session TTL.
- [x] Validate concurrency policy.
- [x] Validate streaming policy.
- [x] Validate cancellation policy.
- [x] Add session creation tests.

## 6. Session Binding

- [x] Bind model reference.
- [x] Bind tokenizer reference.
- [x] Bind generation defaults.
- [x] Bind policy reference.
- [x] Bind memory budget.
- [x] Bind observability correlation.
- [x] Validate model/tokenizer compatibility.
- [x] Add binding tests.

## 7. Session Policy

- [x] Define maximum prompt tokens.
- [x] Define maximum generated tokens.
- [x] Define maximum total tokens.
- [x] Define allowed generation parameters.
- [x] Define allowed sampling modes placeholder.
- [x] Define streaming allowed flag.
- [x] Define cancellation allowed flag.
- [x] Define concurrency policy.
- [x] Define memory budget.
- [x] Define KV cache budget placeholder.
- [x] Define prefix cache allowed placeholder.
- [x] Define observability redaction policy.
- [x] Define raw prompt logging policy.
- [x] Define idle TTL.
- [x] Define total TTL.
- [x] Add policy tests.

## 8. Session Resources

- [x] Define session resource registry.
- [x] Track tokenizer state.
- [x] Track streaming decode state.
- [x] Track output token buffers.
- [x] Track temporary generation buffers.
- [x] Track memory reservations.
- [x] Track future KV cache placeholder resources.
- [x] Track future prefix cache placeholder resources.
- [x] Track model residency references.
- [x] Release resources on close.
- [x] Add resource lifecycle tests.

## 9. Session Memory

- [x] Integrate session with Memory Manager.
- [x] Define session memory budget.
- [x] Track input token buffers.
- [x] Track output token buffers.
- [x] Track logits buffers.
- [x] Track sampling workspace.
- [x] Track tokenizer streaming state.
- [x] Track temporary workspace.
- [x] Track future KV cache memory.
- [x] Track future prefix cache memory.
- [x] Reject or queue when budget exceeded.
- [x] Add session memory tests.

## 10. Session Concurrency

- [x] Define `single-active-operation` policy.
- [x] Define `allow-parallel-operations` policy.
- [x] Define `queue-operations` policy.
- [x] Define `reject-while-active` policy.
- [x] Default stateful session policy conservatively.
- [x] Prevent parallel mutation of future KV cache unless supported.
- [x] Add concurrency tests.

## 11. Session Operations

- [x] Define generate operation.
- [x] Define stream-generate operation.
- [x] Define cancel operation.
- [x] Define drain operation.
- [x] Define close operation.
- [x] Define inspect-status operation.
- [x] Define reset transient state operation where useful.
- [x] Reserve future prefix reuse operation.
- [x] Reserve future fork operation.
- [x] Reserve future snapshot operation.

## 12. One-Shot Inference

- [x] Define one-shot generation behavior.
- [x] Model one-shot request as implicit short-lived session.
- [x] Apply session validation.
- [x] Apply session memory policy.
- [x] Apply session cleanup.
- [x] Add one-shot tests.

## 13. Session And Generation

- [x] Allow GenerationRequest to reference session.
- [x] Use session model binding.
- [x] Use session tokenizer binding.
- [x] Use session policy.
- [x] Use session memory budget.
- [x] Use session cancellation state.
- [x] Use session observability correlation.
- [x] Add session-generation integration tests.

## 14. Session And Streaming

- [x] Track generated token order.
- [x] Track tokenizer streaming decode state.
- [x] Track consumer backpressure state.
- [x] Track cancellation state.
- [x] Track finish reason.
- [x] Track partial decode state.
- [x] Clean streaming state after operation end.
- [x] Add streaming session tests.

## 15. Session Cancellation

- [x] Cancel current operation.
- [x] Cancel queued operations.
- [x] Cancel entire session.
- [x] Coordinate cancellation with Generation.
- [x] Coordinate cancellation with Scheduler.
- [x] Coordinate cancellation with Provider execution.
- [x] Coordinate cancellation with Memory Manager.
- [x] Coordinate cancellation with Tokenizer streaming decode.
- [x] Add cancellation tests.

## 16. Session Drain

- [x] Define session drain behavior.
- [x] Reject new operations while draining.
- [x] Allow current operation to finish where policy permits.
- [x] Drain queued operations according to policy.
- [x] Drain on Runtime shutdown.
- [x] Drain on memory pressure where policy permits.
- [x] Add drain tests.

## 17. Session Expiration

- [x] Define idle TTL.
- [x] Define total TTL.
- [x] Expire idle sessions.
- [x] Expire sessions on policy timeout.
- [x] Expire sessions on model unload.
- [x] Expire sessions on Runtime shutdown.
- [x] Release or cache eligible resources.
- [x] Add expiration tests.

## 18. Session Status

- [x] Define session status structure.
- [x] Include session ID.
- [x] Include lifecycle state.
- [x] Include model reference.
- [x] Include tokenizer reference.
- [x] Include active operation count.
- [x] Include queued operation count.
- [x] Include memory usage summary.
- [x] Include future KV cache usage placeholder.
- [x] Include streaming state summary.
- [x] Include cancellation state.
- [x] Include last error.
- [x] Include created timestamp.
- [x] Include last activity timestamp.
- [x] Include expiration metadata.
- [x] Avoid raw prompt exposure by default.
- [x] Avoid raw handle exposure.
- [x] Add status tests.

## 19. Session Authority

- [x] Define session access policy.
- [x] Prevent session ID from granting authority by itself.
- [x] Prevent raw prompt access by default.
- [x] Prevent KV cache content access by default.
- [x] Prevent memory handle access.
- [x] Prevent Provider handle access.
- [x] Prevent Device handle access.
- [x] Preserve inference-scoped authority.
- [x] Add authority tests.

## 20. Resource Affinity

- [x] Allow session to carry Runtime-derived Resource Affinity.
- [x] Prevent client-forged session affinity.
- [x] Preserve model residency affinity.
- [x] Preserve future KV cache affinity.
- [x] Preserve Provider/Device binding where derived from resources.
- [x] Add affinity tests.

## 21. Model Residency Relationship

- [x] Reference model residency from session.
- [x] Do not expose raw model memory handles.
- [x] Allow model residency to outlive session if cache policy permits.
- [x] Release session-specific references on close.
- [x] Add model residency tests.

## 22. Browser Compatibility

- [x] Keep session model platform-neutral.
- [x] Avoid Wasmtime dependency.
- [x] Avoid native Provider loading requirement.
- [x] Report unsupported browser features explicitly.
- [x] Respect browser memory constraints.
- [x] Add wasm32 check where feasible.

## 23. Error Model

- [x] Define session-creation-failed error.
- [x] Define session-not-found error.
- [x] Define session-not-ready error.
- [x] Define session-active error.
- [x] Define session-closed error.
- [x] Define session-expired error.
- [x] Define session-cancelled error.
- [x] Define session-draining error.
- [x] Define session-policy-denied error.
- [x] Define model-unavailable error.
- [x] Define tokenizer-incompatible error.
- [x] Define memory-admission-failed error.
- [x] Define memory-budget-exceeded error.
- [x] Define generation-failed error.
- [x] Define streaming-failed error.
- [x] Define cancellation-failed error.
- [x] Define operation-queued status.
- [x] Define operation-rejected error.
- [x] Define concurrency-violation error.
- [x] Define resource-cleanup-failed error.
- [x] Define runtime-shutdown error.
- [x] Define internal-session error.

## 24. Observability

- [x] Emit session create requested observation.
- [x] Emit session created observation.
- [x] Emit session creation failed observation.
- [x] Emit session ready observation.
- [x] Emit session active observation.
- [x] Emit session idle observation.
- [x] Emit session draining observation.
- [x] Emit session cancelled observation.
- [x] Emit session closed observation.
- [x] Emit session expired observation.
- [x] Emit session operation started observation.
- [x] Emit session operation completed observation.
- [x] Emit session operation failed observation.
- [x] Emit session memory pressure observation.
- [x] Emit session cleanup observation.
- [x] Emit session policy rejection observation.
- [x] Avoid raw prompt logging by default.

## 25. Tests

- [x] Test session creation success.
- [x] Test session creation with invalid model.
- [x] Test session creation with incompatible tokenizer.
- [x] Test session lifecycle transitions.
- [x] Test one-shot implicit session cleanup.
- [x] Test generate operation through session.
- [x] Test stream-generate operation through session.
- [x] Test cancellation current operation.
- [x] Test cancellation queued operation.
- [x] Test close releases resources.
- [x] Test idle expiration.
- [x] Test total TTL expiration.
- [x] Test reject operation on closed session.
- [x] Test reject operation while draining.
- [x] Test concurrency violation.
- [x] Test queued operation policy.
- [x] Test memory budget exceeded.
- [x] Test session ID does not grant authority.
- [x] Test Resource Affinity cannot be forged.
- [x] Test raw prompt not present in status by default.
- [x] Test raw handles not exposed.

## 26. Documentation

- [x] Document Inference Session model.
- [x] Document lifecycle.
- [x] Document session creation.
- [x] Document session policy.
- [x] Document session resources.
- [x] Document one-shot behavior.
- [x] Document generation relationship.
- [x] Document streaming relationship.
- [x] Document cancellation.
- [x] Document drain.
- [x] Document expiration.
- [x] Document memory relationship.
- [x] Document Resource Affinity relationship.
- [x] Document browser compatibility.
- [x] Document non-goals.

## 27. Final Validation

- [x] Run formatting.
- [x] Run compilation checks.
- [x] Run wasm32 check where feasible.
- [x] Run Clippy.
- [x] Run complete tests.
- [x] Run Session tests.
- [x] Run Generation tests.
- [x] Run Tokenizer tests.
- [x] Run Memory Manager tests where impacted.
- [x] Run Provider conformance tests where impacted.
- [x] Run OpenSpec validation.
- [x] Run coverage validation.
- [x] Verify session is Runtime-owned.
- [x] Verify session does not contain client workspace state.
- [x] Verify session does not expose raw handles.
- [x] Verify raw prompts are not exposed by default.