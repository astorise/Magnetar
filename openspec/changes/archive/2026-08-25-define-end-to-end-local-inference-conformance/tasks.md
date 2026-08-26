# Tasks

## 1. E2E Scope

- [x] Define End-to-End Local Inference Conformance suite.
- [x] Document local-only scope.
- [x] Document no-GPU requirement.
- [x] Document no-network requirement.
- [x] Document no-Tachyon requirement.
- [x] Document no-shortcut rule.
- [x] Document non-goals.

## 2. E2E Module

- [x] Create `e2e_conformance` module or equivalent.
- [x] Add suite version.
- [x] Add fixture version.
- [x] Add report type.
- [x] Add test case registry.
- [x] Add redaction policy for reports.
- [x] Add module-level documentation.

## 3. Fixture Model

- [x] Define minimal Qwen-like decoder-only fixture.
- [x] Define minimal vocabulary.
- [x] Define layer count.
- [x] Define hidden size.
- [x] Define attention head count.
- [x] Define KV head count.
- [x] Define intermediate size.
- [x] Define context length.
- [x] Define deterministic weights.
- [x] Define deterministic expected output.
- [x] Add fixture validation tests.

## 4. Fixture Artifact

- [x] Create fixture Model Artifact manifest.
- [x] Include config metadata.
- [x] Include tensor inventory.
- [x] Include tokenizer fixture metadata.
- [x] Include generation defaults.
- [x] Include test trust state.
- [x] Include deterministic weights.
- [x] Include architecture family metadata.
- [x] Include integrity metadata where applicable.
- [x] Ensure fixture passes Model Artifact validation.
- [x] Add artifact tests.

## 5. Fixture Tokenizer

- [x] Define tokenizer fixture vocabulary.
- [x] Define deterministic encode.
- [x] Define deterministic decode.
- [x] Define streaming decode behavior.
- [x] Define EOS token metadata.
- [x] Define BOS token policy where needed.
- [x] Define special token metadata.
- [x] Validate through Tokenizer Contract.
- [x] Add tokenizer fixture tests.

## 6. Fixture Prompt

- [x] Define plain text prompt fixture.
- [x] Define already-tokenized prompt fixture.
- [x] Define optional chat-message prompt fixture.
- [x] Verify text path uses Tokenizer Contract.
- [x] Verify token path bypasses text tokenization only explicitly.
- [x] Verify raw prompt logging disabled by default.
- [x] Add prompt tests.

## 7. Required Success Path

- [x] Resolve model.
- [x] Load model.
- [x] Create Model Instance.
- [x] Create session.
- [x] Tokenize prompt.
- [x] Run generation.
- [x] Run prefill.
- [x] Run decode.
- [x] Run Sampling.
- [x] Stream events.
- [x] Return result.
- [x] Close session.
- [x] Cleanup resources.
- [x] Add success path test.

## 8. One-Shot Path

- [x] Add one-shot inference E2E case where policy allows.
- [x] Verify implicit session creation.
- [x] Verify normal Model Instance path.
- [x] Verify normal Tokenizer path.
- [x] Verify normal Generation path.
- [x] Verify normal Sampling path.
- [x] Verify normal Provider/Kernel path.
- [x] Add one-shot tests.

## 9. No Shortcut Validation

- [x] Detect direct Provider invocation.
- [x] Detect direct Kernel invocation.
- [x] Detect raw tensor pointer access.
- [x] Detect Model Loading bypass.
- [x] Detect Model Component bypass.
- [x] Detect Kernel Registry bypass.
- [x] Detect Memory Manager bypass.
- [ ] Detect Tokenizer bypass for text input.
- [x] Detect Runtime tool execution.
- [x] Detect Runtime workspace file reads.
- [ ] Detect silent CPU fallback.
- [x] Detect silent dtype conversion.
- [x] Detect silent layout conversion.
- [x] Add no-shortcut tests.

## 10. Reference CPU Path

- [x] Require Reference CPU Provider availability.
- [x] Select Reference CPU through Kernel Registry.
- [x] Dispatch Reference CPU through Runtime dispatch.
- [x] Verify CPU use is policy-explicit.
- [x] Verify CPU use is not hidden fallback.
- [x] Add Reference CPU E2E tests.

## 11. Operator Coverage

- [x] Exercise embedding.
- [x] Exercise RMSNorm.
- [x] Exercise matmul.
- [x] Exercise RoPE.
- [x] Exercise attention.
- [x] Exercise softmax.
- [x] Exercise SiLU.
- [x] Exercise add.
- [x] Exercise mul.
- [x] Exercise residual-add.
- [x] Add dtype conversion E2E case where implemented.
- [x] Add layout conversion E2E case where implemented.
- [x] Add operator coverage tests.

## 12. Graph Validation

- [x] Validate prefill graph.
- [x] Validate decode graph.
- [x] Add invalid graph fixture.
- [x] Add unsupported operator fixture.
- [x] Add invalid tensor shape fixture.
- [x] Add missing kernel fixture.
- [x] Add graph validation tests.

## 13. Generation Validation

- [x] Validate max new tokens.
- [x] Validate max total tokens.
- [x] Validate stop condition.
- [x] Validate EOS behavior.
- [x] Validate greedy sampling path.
- [x] Validate finish reason.
- [x] Validate usage accounting.
- [x] Validate cancellation behavior.
- [x] Validate streaming event sequence.
- [x] Add generation tests.

## 14. Sampling Validation

- [x] Use greedy deterministic sampling in success path.
- [x] Validate Sampling Contract invocation.
- [x] Validate Sampling Result usage.
- [x] Add seed test if stochastic sampling is tested.
- [x] Verify Provider-assisted sampling is not required.
- [x] Add sampling tests.

## 15. Streaming Validation

- [x] Validate generation-started event.
- [x] Validate prefill-started event.
- [x] Validate prefill-completed event.
- [x] Validate decode-started event.
- [x] Validate decode-token event.
- [ ] Validate decoded-text event.
- [ ] Validate usage-updated event.
- [x] Validate stop-reached or generation-completed event.
- [x] Validate stream-closed event.
- [x] Validate ordering.
- [x] Add streaming tests.

## 16. Session Validation

- [x] Validate session creation.
- [x] Validate session use.
- [x] Validate session close.
- [x] Validate session not usable after close.
- [x] Validate cleanup releases inference resources.
- [x] Validate session excludes workspace state.
- [x] Validate session excludes Git state.
- [x] Validate session excludes tool state.
- [x] Validate session excludes secret state.
- [x] Add session tests.

## 17. KV Cache Validation

- [x] Add KV cache E2E case if implemented.
- [x] Validate cache allocation.
- [x] Validate cache append during prefill.
- [x] Validate cache consumption during decode.
- [x] Validate cache cleanup.
- [x] Validate raw cache not exposed.
- [x] Add KV cache tests.

## 18. Prefix Cache Validation

- [x] Keep Prefix Cache optional in first E2E suite.
- [x] Add Prefix Cache case if enabled.
- [x] Validate hit metadata redacted.
- [x] Validate miss metadata redacted.
- [x] Validate raw prompt not exposed.
- [x] Validate raw KV cache not exposed.
- [x] Add Prefix Cache tests where applicable.

## 19. Tensor Validation

- [x] Validate Tensor Descriptors created.
- [x] Validate Tensor Resources allocated.
- [x] Validate host contiguous layout for Reference CPU.
- [x] Validate dtype explicitness.
- [x] Validate layout explicitness.
- [x] Validate no raw pointer exposure.
- [x] Validate output readiness update.
- [x] Validate tensor cleanup.
- [x] Add tensor tests.

## 20. Memory Validation

- [x] Validate model tensor accounting.
- [x] Validate operator output accounting.
- [x] Validate workspace accounting where needed.
- [x] Validate memory pressure failure path.
- [x] Validate cleanup release.
- [x] Validate no untracked Runtime-visible allocation.
- [x] Add memory tests.

## 21. CLI Boundary Validation

- [x] Add CLI harness or boundary test.
- [x] Verify CLI sends explicit prompt/context.
- [x] Verify Runtime does not read workspace files.
- [x] Verify Runtime does not execute Git.
- [x] Verify Runtime does not execute tools.
- [x] Verify Runtime does not execute shell/process.
- [x] Verify Runtime receives no ambient CLI authority.
- [x] Verify CLI preserves Runtime structured errors.
- [x] Add CLI boundary tests.

## 22. Diagnostics And Redaction

- [x] Validate diagnostics redaction.
- [x] Validate observability redaction.
- [x] Prevent raw prompt exposure.
- [x] Prevent raw model weight exposure.
- [x] Prevent raw tensor value exposure.
- [x] Prevent raw KV cache exposure.
- [x] Prevent secret exposure.
- [x] Prevent filesystem authority exposure.
- [x] Prevent Provider handle exposure.
- [x] Prevent Device handle exposure.
- [x] Prevent Kernel handle exposure.
- [x] Prevent memory pointer exposure.
- [x] Add redaction tests.

## 23. Failure Cases

- [x] Add invalid model reference test.
- [x] Add untrusted artifact test.
- [x] Add incompatible tokenizer test.
- [x] Add unsupported operator test.
- [x] Add missing required kernel test.
- [x] Add invalid tensor shape test.
- [x] Add memory admission failure test.
- [x] Add session closed test.
- [x] Add generation cancelled test.
- [x] Add generation timeout test.
- [x] Add policy denied test.
- [x] Add raw handle access denied test.
- [x] Add Runtime file access denied test.
- [x] Add Runtime tool execution denied test.

## 24. Determinism

- [x] Use deterministic fixture weights.
- [x] Use greedy sampling.
- [x] Use fixed input tokens.
- [x] Use fixed generation limit.
- [x] Use Reference CPU execution.
- [x] Use explicit dtype policy.
- [x] Use explicit layout policy.
- [x] Avoid stochastic sampling in first success path.
- [x] Define expected output tokens or text.
- [x] Add determinism tests.

## 25. Report Format

- [x] Define machine-readable E2E report.
- [x] Include suite version.
- [x] Include fixture version.
- [x] Include Runtime version.
- [x] Include Provider summary.
- [x] Include Device summary.
- [x] Include Model Component summary.
- [x] Include Operator coverage summary.
- [x] Include Kernel coverage summary.
- [x] Include test case results.
- [x] Include pass/fail/skipped status.
- [x] Include structured failure reasons.
- [x] Include redaction status.
- [x] Include duration metadata.
- [x] Prevent sensitive raw values in report.
- [x] Add report tests.

## 26. CI Integration

- [x] Add E2E suite to CI.
- [x] Ensure CPU-only environment support.
- [x] Keep first suite lightweight.
- [x] Gate large or slow tests separately.
- [x] Run with OpenSpec validation.
- [x] Run with coverage checks.
- [x] Add CI documentation.

## 27. Browser And Tachyon Boundary

- [x] Document browser E2E out of scope.
- [x] Return structured browser unsupported paths.
- [x] Document Tachyon not required.
- [x] Prevent Tachyon dependency in local E2E.
- [x] Add boundary checks.

## 28. Error Model

- [x] Define e2e-suite-unavailable error.
- [x] Define e2e-fixture-invalid error.
- [x] Define e2e-model-resolution-failed error.
- [x] Define e2e-model-loading-failed error.
- [x] Define e2e-model-component-failed error.
- [x] Define e2e-tokenizer-failed error.
- [x] Define e2e-session-failed error.
- [x] Define e2e-generation-failed error.
- [x] Define e2e-sampling-failed error.
- [x] Define e2e-streaming-failed error.
- [x] Define e2e-graph-validation-failed error.
- [x] Define e2e-operator-coverage-missing error.
- [x] Define e2e-kernel-coverage-missing error.
- [x] Define e2e-memory-validation-failed error.
- [x] Define e2e-redaction-failed error.
- [x] Define e2e-boundary-violation error.
- [x] Define e2e-determinism-failed error.
- [x] Define internal-e2e-conformance error.

## 29. Observability

- [x] Emit E2E suite started observation.
- [x] Emit E2E fixture loaded observation.
- [x] Emit E2E success path started observation.
- [x] Emit E2E success path completed observation.
- [x] Emit E2E failure case started observation.
- [x] Emit E2E failure case completed observation.
- [x] Emit E2E redaction failure observation.
- [x] Emit E2E boundary violation observation.
- [x] Emit E2E report generated observation.
- [x] Avoid raw prompt/weight/tensor/cache/handle/path/secret logging.

## 30. Documentation

- [x] Document End-to-End Local Inference Conformance.
- [x] Document fixture model.
- [x] Document fixture tokenizer.
- [x] Document fixture artifact.
- [x] Document success path.
- [x] Document no-shortcut rule.
- [x] Document Reference CPU requirement.
- [x] Document operator coverage.
- [x] Document graph validation.
- [x] Document generation validation.
- [x] Document streaming validation.
- [x] Document session validation.
- [x] Document cache validation.
- [x] Document tensor/memory validation.
- [x] Document CLI boundary validation.
- [x] Document diagnostics/redaction validation.
- [x] Document failure cases.
- [x] Document determinism.
- [x] Document report format.
- [x] Document CI integration.
- [x] Document non-goals.

## 31. Final Validation

- [x] Run formatting.
- [x] Run compilation checks.
- [x] Run wasm32 check where feasible.
- [x] Run Clippy.
- [x] Run unit tests.
- [x] Run contract tests.
- [x] Run Provider conformance tests.
- [x] Run first operator scope conformance.
- [x] Run Qwen baseline conformance.
- [x] Run Runtime Inference API tests.
- [x] Run CLI boundary tests.
- [x] Run E2E local inference conformance.
- [x] Run OpenSpec validation.
- [x] Run coverage validation.
- [x] Verify E2E does not require GPU.
- [x] Verify E2E does not require network.
- [x] Verify E2E does not bypass Runtime contracts.
- [x] Verify redaction defaults pass.
