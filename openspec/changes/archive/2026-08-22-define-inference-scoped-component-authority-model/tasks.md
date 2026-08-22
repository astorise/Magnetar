# Tasks

## 1. Scope Recadrage

- [x] Update architecture documentation to state that Magnetar is an inference
      Runtime.
- [x] Document that `magnetar-cli` owns workspace and agent orchestration.
- [x] Document that Magnetar does not own general-purpose tool execution.
- [x] Document that Magnetar does not own filesystem authority.
- [x] Document that Magnetar does not own network authority.
- [x] Document that Magnetar does not own secret authority.
- [x] Document that Magnetar does not own Git authority.
- [x] Document that Magnetar does not own process execution authority.

## 2. Authority Taxonomy Replacement

- [x] Remove broad authority categories from Magnetar Component validation.
- [x] Remove `filesystem` as a Magnetar Runtime authority.
- [x] Remove `network` as a Magnetar Runtime authority.
- [x] Remove `environment` as a Magnetar Runtime authority.
- [x] Remove `process` as a Magnetar Runtime authority.
- [x] Remove `shell` as a Magnetar Runtime authority.
- [x] Remove `secrets` as a Magnetar Runtime authority.
- [x] Remove `workspace` as a Magnetar Runtime authority.
- [x] Remove `git` as a Magnetar Runtime authority.
- [x] Remove `source-control` as a Magnetar Runtime authority.
- [x] Remove `tool-execution` as a Magnetar Runtime authority.
- [x] Remove `external-service` as a Magnetar Runtime authority.

## 3. Inference Authority Taxonomy

- [x] Add `model-artifact-read`.
- [x] Add `tokenizer-artifact-read`.
- [x] Add `prompt-template-read`.
- [x] Add `adapter-artifact-read`.
- [x] Add `quantization-artifact-read`.
- [x] Add `inference-session-state`.
- [x] Add `generation-session-state`.
- [x] Add `kv-cache-access`.
- [x] Add `prefix-cache-access`.
- [x] Add `compute-capability`.
- [x] Add `generation-capability`.
- [x] Add `sampling-capability`.
- [x] Add `observability-emit`.
- [x] Add `runtime-diagnostics`.
- [x] Document each authority.

## 4. Manifest Schema Update

- [x] Update Component Artifact manifest schema examples.
- [x] Update manifest schema validation.
- [x] Reject broad authority kinds by default.
- [x] Accept inference-scoped authority kinds.
- [x] Preserve manifest digest validation.
- [x] Preserve WIT import/export validation.
- [x] Preserve Runtime compatibility validation.
- [x] Preserve Capability compatibility validation.
- [x] Preserve trust policy validation.

## 5. Fail-Closed Behavior

- [x] Ensure unknown authority kinds fail closed.
- [x] Ensure broad tool-like authority kinds fail closed.
- [x] Ensure development mode does not silently allow broad authorities.
- [x] Ensure trusted digest does not override forbidden authority.
- [x] Ensure trusted publisher does not override forbidden authority.
- [x] Ensure Tachyon source metadata does not override forbidden authority.
- [x] Ensure local source metadata does not override forbidden authority.

## 6. Artifact Validation Pipeline

- [x] Insert authority taxonomy validation before trust finalization.
- [x] Reject invalid authority before ComponentEngine preparation.
- [x] Reject forbidden broad authority before ComponentEngine preparation.
- [x] Record rejected authority reason in diagnostics.
- [x] Emit observability event for authority rejection.
- [x] Ensure artifact is not prepared after authority rejection.

## 7. Runtime Linking

- [x] Link only inference-scoped Capabilities.
- [x] Map `compute-capability` to authorized Compute Runtime endpoint.
- [x] Map `generation-capability` to authorized Generation endpoint when it
      exists.
- [x] Map `sampling-capability` to authorized sampling/logits endpoint when it
      exists.
- [x] Map artifact-read authorities to Runtime-managed inference artifact
      registries.
- [x] Map cache authorities to Runtime-managed inference cache services.
- [x] Map observability authority to Runtime observability endpoint.
- [x] Do not link filesystem, network, process, Git, secrets, or workspace
      interfaces.

## 8. Model Artifact Access Boundary

- [x] Ensure `model-artifact-read` refers to Runtime-registered model artifacts.
- [x] Ensure it does not expose arbitrary local file paths.
- [x] Ensure model artifact identity remains separate from Component Artifact
      identity.
- [x] Ensure model artifact access can be scoped to one inference session.
- [x] Add tests for authorized model artifact reference.
- [x] Add tests rejecting arbitrary path access.

## 9. Tokenizer and Template Boundary

- [x] Ensure tokenizer access refers to Runtime-registered tokenizer artifacts.
- [x] Ensure prompt-template access refers to Runtime-registered templates.
- [x] Reject arbitrary filesystem reads for tokenizer/template access.
- [x] Add tests for valid tokenizer/template references.
- [x] Add tests for unauthorized tokenizer/template references.

## 10. Adapter and Quantization Boundary

- [x] Ensure adapter artifact access is Runtime-registered.
- [x] Ensure quantization artifact access is Runtime-registered.
- [x] Reject arbitrary adapter file reads.
- [x] Reject arbitrary quantization file reads.
- [x] Preserve distinction between executable Component and data artifacts.

## 11. Cache Authority

- [x] Define cache authority as inference-session scoped.
- [x] Scope KV cache access to authorized sessions.
- [x] Scope prefix cache access to authorized sessions.
- [x] Prevent cross-session cache inspection.
- [x] Prevent unrelated model cache access.
- [x] Add cache authority tests where cache model exists or add pending tests
      marked for future cache implementation.

## 12. Observability Authority

- [x] Ensure `observability-emit` allows Runtime-mediated observation emission.
- [x] Ensure it does not allow direct network export.
- [x] Ensure exporter destination remains Runtime policy.
- [x] Redact prompt, token, diagnostic, and artifact metadata according to policy.
- [x] Add tests that observability authority does not imply network access.

## 13. Diagnostics Authority

- [x] Define `runtime-diagnostics` as inference-diagnostic authority only.
- [x] Redact secrets and external client data.
- [x] Avoid exposing native Provider handles.
- [x] Avoid exposing Device handles.
- [x] Avoid exposing filesystem paths unless policy permits.
- [x] Add diagnostics redaction tests.

## 14. Component Type Guidance

- [x] Document valid Magnetar Component examples.
- [x] Include model architecture Components.
- [x] Include tokenizer Components.
- [x] Include prompt-template Components.
- [x] Include sampling/logits processor Components.
- [x] Include observability Components where inference-scoped.
- [x] Document invalid Magnetar Component examples.
- [x] Include filesystem tool Components as out of scope.
- [x] Include Git tool Components as out of scope.
- [x] Include shell tool Components as out of scope.
- [x] Include network fetcher Components as out of scope.
- [x] Include secret reader Components as out of scope.

## 15. magnetar-cli Boundary Documentation

- [x] Document that `magnetar-cli` may implement agent orchestration.
- [x] Document that `magnetar-cli` may own workspace access.
- [x] Document that `magnetar-cli` may own Git access.
- [x] Document that `magnetar-cli` may own filesystem access.
- [x] Document that `magnetar-cli` may own process execution.
- [x] Document that `magnetar-cli` may own network access.
- [x] Document that `magnetar-cli` may own secret access.
- [x] Document that `magnetar-cli` calls Magnetar for inference.
- [x] Do not implement CLI authority in this change.

## 16. Tachyon Boundary Documentation

- [x] Document that Tachyon may distribute inference Components.
- [x] Document that Tachyon does not grant Magnetar broad tool authority.
- [x] Document that Magnetar validates Tachyon-provided Components locally.
- [x] Document that Tachyon distribution does not bypass authority validation.
- [x] Preserve vendor-neutral source model.

## 17. Tests

- [x] Test manifest with `model-artifact-read` authority succeeds when trusted.
- [x] Test manifest with `compute-capability` authority succeeds when trusted.
- [x] Test manifest with `observability-emit` authority succeeds when trusted.
- [x] Test manifest with `filesystem` authority fails.
- [x] Test manifest with `network` authority fails.
- [x] Test manifest with `secrets` authority fails.
- [x] Test manifest with `git` authority fails.
- [x] Test manifest with `workspace` authority fails.
- [x] Test manifest with `process` authority fails.
- [x] Test trusted digest does not override forbidden authority.
- [x] Test development mode does not override forbidden broad authority.
- [x] Test Tachyon source metadata does not override forbidden authority.
- [x] Test local source metadata does not override forbidden authority.

## 18. Documentation Updates

- [x] Update Component Artifact documentation.
- [x] Update manifest examples.
- [x] Update architecture overview.
- [x] Update README if it mentions tool authority.
- [x] Update OpenSpec project context if required.
- [x] Remove broad authority examples from current documentation.
- [x] Add inference-scoped authority examples.

## 19. OpenSpec Consistency

- [x] Review active specs for filesystem authority assigned to Magnetar.
- [x] Review active specs for network authority assigned to Magnetar.
- [x] Review active specs for secret authority assigned to Magnetar.
- [x] Review active specs for Git authority assigned to Magnetar.
- [x] Review active specs for workspace authority assigned to Magnetar.
- [x] Review active specs for tool execution assigned to Magnetar.
- [x] Replace those with client-owned language where needed.
- [x] Preserve archived OpenSpec history unchanged.

## 20. Final Validation

- [x] Run formatting.
- [x] Run compilation checks.
- [x] Run Clippy.
- [x] Run complete tests.
- [x] Run Component artifact tests.
- [x] Run WIT validation.
- [x] Run OpenSpec validation.
- [x] Run coverage validation.
- [x] Verify Magnetar grants only inference-scoped Component authority.
- [x] Verify broad tool authorities are rejected.
- [x] Verify `magnetar-cli` remains the documented owner of workspace/tool
      authority.
