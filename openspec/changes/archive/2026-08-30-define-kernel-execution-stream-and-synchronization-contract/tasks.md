# Tasks

## 1. Execution Stream Domain

- [x] Define ExecutionStreamId.
- [x] Define ExecutionStream.
- [x] Define ExecutionStreamClass.
- [x] Define ExecutionStreamState.
- [x] Define optional priority hint.
- [x] Add lifecycle tests.

## 2. Stream Classes

- [x] Define compute class.
- [x] Define transfer class.
- [x] Define control class.
- [x] Keep class identity extensible.
- [x] Prevent native queue-type leakage.

## 3. Stream Ordering

- [x] Define same-stream ordering.
- [x] Define cross-stream non-ordering by default.
- [x] Add ordered submission tests.
- [x] Add cross-stream race tests.
- [x] Prevent host submission order from implying cross-stream order.

## 4. Completion Token

- [x] Define CompletionTokenId.
- [x] Define CompletionToken state.
- [x] Add pending.
- [x] Add completed.
- [x] Add failed.
- [x] Add cancelled.
- [x] Add lost.
- [x] Enforce terminal-state-once semantics.

## 5. Completion Scope

- [x] Support Kernel completion.
- [x] Support transfer completion.
- [x] Support prepared-segment completion.
- [x] Support grouped submission completion.
- [x] Add scope tests.

## 6. Execution Dependency

- [x] Define ExecutionDependency.
- [x] Reference predecessor tokens.
- [x] Validate stream/provider relationships.
- [x] Detect invalid dependencies.
- [x] Detect dependency cycles where applicable.
- [x] Add dependency tests.

## 7. Dependency Satisfaction

- [x] Block dependent execution until predecessors ready.
- [x] Preserve Device-side dependency opportunity.
- [x] Avoid unnecessary host waits.
- [x] Add cross-stream dependency tests.

## 8. Execution Submission

- [x] Define ExecutionSubmission.
- [x] Bind logical stream.
- [x] Bind PreparedKernelId or prepared segment.
- [x] Bind resources.
- [x] Bind dependencies.
- [x] Bind deadline.
- [x] Bind cancellation scope.
- [x] Add validation tests.

## 9. Non-Blocking Completion

- [x] Add completion poll.
- [x] Add terminal-state query.
- [x] Keep operation non-blocking.
- [x] Add pending/completed tests.

## 10. Host Wait

- [x] Add explicit wait semantics.
- [x] Add timeout.
- [x] Add Provider failure propagation.
- [x] Add Device-loss propagation.
- [x] Add wait tests.

## 11. Resource Readiness

- [x] Define ResourceReadiness.
- [x] Associate writer CompletionToken.
- [x] Track host readiness.
- [x] Track Device-consumer readiness.
- [x] Add readiness tests.

## 12. Last Writer

- [x] Track last pending writer conservatively.
- [x] Block incompatible readers.
- [x] Allow ordered same-stream access.
- [x] Add RAW hazard tests.

## 13. Read/Write Hazards

- [x] Handle RAW.
- [x] Handle WAR.
- [x] Handle WAW.
- [x] Support conservative whole-resource tracking.
- [x] Add hazard tests.

## 14. Aliasing

- [x] Integrate Tensor alias metadata.
- [x] Detect overlapping lifetime requirements.
- [x] Prevent false independence.
- [x] Add alias synchronization tests.

## 15. Memory Reuse

- [x] Bind allocation reuse to completion.
- [x] Track outstanding users.
- [x] Delay reuse until quiescent.
- [x] Add use-after-free prevention tests.

## 16. Workspace Reuse

- [x] Track workspace completion.
- [x] Allow reuse after completion.
- [x] Prevent overlapping incompatible reuse.
- [x] Add workspace tests.

## 17. Resource Retirement

- [x] Separate logical handle destruction from physical reuse.
- [x] Retain allocation while in-flight.
- [x] Add asynchronous destruction tests.

## 18. Transfer Stream

- [x] Support asynchronous data movement.
- [x] Produce CompletionToken.
- [x] Support dependency on compute.
- [x] Support compute dependency on transfer.
- [x] Add overlap tests.

## 19. Resource Affinity

- [x] Validate Provider affinity.
- [x] Validate Device affinity.
- [x] Reject implicit cross-Provider access.
- [x] Require explicit movement.
- [x] Add affinity tests.

## 20. Cross-Device Dependencies

- [x] Define logical cross-Device dependency.
- [x] Permit Provider-native optimization where supported.
- [x] Provide Runtime-mediated fallback.
- [x] Add cross-Device tests.

## 21. Cross-Provider Dependencies

- [x] Prevent raw native event exchange.
- [x] Runtime-mediate completion.
- [x] Add unsupported-interop errors.
- [x] Add cross-Provider tests.

## 22. KV Cache Ordering

- [x] Bind KV writes to completion.
- [x] Bind next decode read to prior KV readiness.
- [x] Preserve sequence ordering.
- [x] Add incremental decode tests.

## 23. Paged KV Cache

- [x] Track page in-flight usage.
- [x] Prevent early page reuse.
- [x] Synchronize page mutation.
- [x] Add page-retirement tests.

## 24. Prefix Cache

- [x] Ensure constructed prefix resource ready before shared reads.
- [x] Support concurrent read-only consumers.
- [x] Add prefix synchronization tests.

## 25. Continuous Batching

- [x] Associate batch step with completion.
- [x] Track sequence/resource dependencies.
- [x] Prevent early slot reuse.
- [x] Allow independent sequence concurrency.
- [x] Add batching tests.

## 26. Scheduler Boundary

- [x] Scheduler supplies logical work decisions.
- [x] Scheduler never owns native stream.
- [x] Runtime maps work to ExecutionStream.
- [x] Add architecture boundary tests.

## 27. Stream Priority

- [x] Define portable priority hint.
- [x] Keep advisory.
- [x] Prevent correctness override.
- [x] Prevent starvation where policy requires fairness.
- [x] Add priority tests.

## 28. Cancellation

- [x] Stop future dependent submissions.
- [x] Request queued/native cancellation where supported.
- [x] Preserve in-flight resource lifetime.
- [x] Prevent cancelled outputs from normal publication.
- [x] Add cancellation tests.

## 29. Provider Cancellation Capability

- [x] Advertise not-supported.
- [x] Advertise before-submit-only.
- [x] Advertise queued-work.
- [x] Advertise cooperative.
- [x] Advertise interruptible.
- [x] Allow Provider-specific capability.
- [x] Add capability tests.

## 30. Cancellation Completion Separation

- [x] Distinguish request cancellation from physical completion.
- [x] Keep CompletionToken alive.
- [x] Prevent early resource reuse.
- [x] Add late-completion tests.

## 31. Deadlines

- [x] Add submission deadline.
- [x] Stop future work after expiry.
- [x] Request cancellation where possible.
- [x] Keep resources until quiescent.
- [x] Add deadline tests.

## 32. Dependency Failure

- [x] Propagate predecessor failure.
- [x] Prevent dependent submission/execution.
- [x] Preserve structured reason.
- [x] Add failure-chain tests.

## 33. Provider Failure

- [x] Transition affected tokens.
- [x] Mark completion lost where appropriate.
- [x] Invalidate streams.
- [x] Add Provider-crash tests.

## 34. Device Loss

- [x] Invalidate affected streams.
- [x] Invalidate dependent Plans.
- [x] Do not mark unfinished output ready.
- [x] Add Device-loss tests.

## 35. Stream Drain

- [x] Stop new work.
- [x] Allow outstanding completion.
- [x] Close after quiescence.
- [x] Add drain tests.

## 36. Stream Destruction

- [x] Prevent destruction while required native state is in use.
- [x] Release Provider state safely.
- [x] Add concurrent destruction tests.

## 37. CompletionToken Lifetime

- [x] Retain token until dependencies consumed.
- [x] Release Provider-native event state afterward.
- [x] Prevent premature release.
- [x] Add lifecycle tests.

## 38. ABA Protection

- [x] Add generation or equivalent identity.
- [x] Prevent reused numeric token confusion.
- [x] Add token-reuse tests.

## 39. Provider Capability Discovery

- [x] Advertise async submission.
- [x] Advertise ordered streams.
- [x] Advertise cross-stream dependencies.
- [x] Advertise Device-side dependency.
- [x] Advertise host wait/poll.
- [x] Advertise transfer overlap.
- [x] Advertise priorities.
- [x] Advertise cancellation.
- [x] Advertise deadlines.
- [x] Advertise multi-Device capability.

## 40. Synchronous Provider Fallback

- [x] Allow immediate completion semantics.
- [x] Produce completed CompletionToken.
- [x] Preserve same logical API.
- [x] Add Reference CPU baseline test.

## 41. Prepared Plan Integration

- [x] Add stream assignment to Plan binding.
- [x] Add static dependencies.
- [x] Add dynamic dependency slots.
- [x] Add resource readiness edges.
- [x] Add segment completion dependencies.
- [x] Add Plan tests.

## 42. Prepared Segment Synchronization

- [x] Define segment completion.
- [x] Keep internal Provider sync opaque.
- [x] Bind segment outputs to CompletionToken.
- [x] Add segment tests.

## 43. Provider ABI Extension

- [x] Define optional synchronization capability version.
- [x] Keep C-compatible ABI.
- [x] Avoid Rust trait objects.
- [x] Use opaque identifiers.
- [x] Define ownership.
- [x] Prevent unwind across ABI.
- [x] Add ABI conformance tests.

## 44. ABI Operations

- [x] Reserve create stream.
- [x] Reserve release stream.
- [x] Reserve submit Kernel.
- [x] Reserve submit segment.
- [x] Reserve poll completion.
- [x] Reserve wait completion.
- [x] Reserve release completion.
- [x] Reserve cancellation request.

## 45. Device Boundary

- [x] Keep Device metadata/status only.
- [x] Prevent Device native stream API.
- [x] Prevent Device native event API.
- [x] Add boundary tests.

## 46. WIT Boundary

- [x] Prevent native synchronization exposure to Components.
- [x] Prevent CompletionToken hardware control in Component API.
- [x] Preserve portable graph/data-movement semantics.
- [x] Add WIT boundary tests.

## 47. Runtime Inference API Boundary

- [x] Expose only high-level cancellation/completion where needed.
- [x] Never expose native stream/event handle.
- [x] Add API boundary tests.

## 48. Hot Path

- [x] Reuse prepared dependency plan.
- [x] Keep dependency binding bounded.
- [x] Prevent compile.
- [x] Prevent qualification.
- [x] Prevent Registry-wide search.
- [x] Prevent autotuning.
- [x] Add hot-path conformance tests.

## 49. Error Model

- [x] Add stream errors.
- [x] Add submission errors.
- [x] Add dependency errors.
- [x] Add completion errors.
- [x] Add readiness errors.
- [x] Add cancellation errors.
- [x] Add deadline errors.
- [x] Add Provider synchronization errors.
- [x] Add Device-loss errors.

## 50. Observability

- [x] Observe stream lifecycle.
- [x] Observe submissions.
- [x] Observe dependencies.
- [x] Observe completion.
- [x] Observe readiness.
- [x] Observe cancellation.
- [x] Observe deadlines.
- [x] Observe resource reuse delays.
- [x] Redact native handles/resources.

## 51. Conformance

- [x] Prove logical/native separation.
- [x] Prove same-stream ordering.
- [x] Prove cross-stream explicit dependency.
- [x] Prove unfinished writes unavailable.
- [x] Prove host-read synchronization.
- [x] Prove memory reuse fencing.
- [x] Prove alias safety.
- [x] Prove transfer overlap.
- [x] Prove cross-Provider mediation.
- [x] Prove KV ordering.
- [x] Prove continuous-batch safety.
- [x] Prove cancellation != completion.
- [x] Prove deadline resource safety.
- [x] Prove failure propagation.
- [x] Prove Device-loss fail-close.
- [x] Prove stream retirement safety.
- [x] Prove synchronous Provider support.
- [x] Prove Plan native-handle isolation.
- [x] Prove observability redaction.

## 52. Documentation

- [x] Document ExecutionStream.
- [x] Document CompletionToken.
- [x] Document dependency semantics.
- [x] Document resource readiness.
- [x] Document same/cross-stream ordering.
- [x] Document memory reuse fences.
- [x] Document KV-cache ordering.
- [x] Document cancellation.
- [x] Document Provider synchronization ownership.
- [x] Document synchronous fallback.

## 53. Final Validation

- [x] Run OpenSpec validation.
- [x] Verify Runtime owns logical dependencies.
- [x] Verify Provider owns native synchronization.
- [x] Verify Device remains metadata/status-only.
- [x] Verify no native synchronization leaks through public APIs.
- [x] Verify asynchronous memory reuse is safe.
- [x] Verify token hot path needs no global Device synchronization.