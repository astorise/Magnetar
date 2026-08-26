# Tasks

## 1. Boundary Scope

- [x] Define `magnetar-cli` as first-party client/runtime around inference.
- [x] Define `magnetar-runtime` as inference runtime.
- [x] Document CLI versus Runtime responsibilities.
- [x] Document Runtime Inference API usage.
- [x] Document forbidden Runtime responsibilities.
- [x] Document non-goals.

## 2. CLI Boundary Module

- [x] Add CLI boundary documentation or spec module.
- [x] Define CLI request-to-runtime flow.
- [x] Define CLI-owned authorities.
- [x] Define Runtime-owned authorities.
- [x] Add boundary tests where applicable.

## 3. Command Boundary

- [x] Document `magnetar run` conceptual flow.
- [x] Document `magnetar chat` conceptual flow.
- [x] Document `magnetar model list` conceptual flow.
- [x] Document `magnetar model inspect` conceptual flow.
- [x] Document `magnetar model load` conceptual flow.
- [x] Document `magnetar model unload` conceptual flow.
- [x] Document `magnetar providers` conceptual flow.
- [x] Document `magnetar devices` conceptual flow.
- [x] Document `magnetar sessions` conceptual flow.
- [x] Document `magnetar serve` conceptual boundary.

## 4. `magnetar run`

- [x] Parse model argument in CLI.
- [x] Parse prompt argument in CLI.
- [x] Resolve prompt input in CLI.
- [x] Read requested files in CLI only.
- [x] Build prompt payload in CLI.
- [x] Call Runtime Inference API.
- [x] Render generation output in CLI.
- [x] Ensure Runtime does not read workspace files.
- [x] Add `run` boundary tests.

## 5. `magnetar chat`

- [x] Manage interactive loop in CLI.
- [x] Store CLI transcript where policy allows.
- [x] Store command history where policy allows.
- [x] Manage terminal UI in CLI.
- [x] Call Runtime session APIs.
- [x] Call Runtime generation APIs.
- [x] Render streaming events.
- [x] Ensure Runtime does not own terminal UX.
- [x] Add `chat` boundary tests.

## 6. Model Commands

- [x] Map friendly model aliases in CLI.
- [x] Call Runtime model resolution.
- [x] Call Runtime model loading.
- [x] Call Runtime model unload.
- [x] Display redacted model metadata.
- [x] Prevent bypassing trust checks.
- [x] Prevent bypassing Model Loading checks.
- [x] Add model command boundary tests.

## 7. Provider And Device Commands

- [x] Call Runtime diagnostics for Providers.
- [x] Call Runtime diagnostics for Devices.
- [x] Display redacted Provider metadata.
- [x] Display redacted Device metadata.
- [x] Prevent raw Provider handle display.
- [x] Prevent raw Device handle display.
- [x] Prevent raw Kernel handle display.
- [x] Add diagnostics command tests.

## 8. Workspace And File Access

- [x] Keep workspace access in CLI.
- [x] Keep arbitrary file reads in CLI.
- [x] Require explicit user request or policy.
- [x] Pass selected content to Runtime as prompt/context.
- [x] Prevent Runtime workspace scanning.
- [x] Prevent Runtime arbitrary file reading.
- [x] Add file boundary tests.

## 9. Git Access

- [x] Keep Git access in CLI.
- [x] Allow Git context collection where policy allows.
- [x] Pass Git-derived context explicitly to Runtime if needed.
- [x] Prevent Runtime direct Git calls.
- [x] Add Git boundary tests.

## 10. Network Access

- [x] Keep arbitrary network access in CLI.
- [x] Apply CLI network policy.
- [x] Pass retrieved context explicitly to Runtime if needed.
- [x] Prevent Runtime arbitrary network operations.
- [x] Preserve model distribution contract separately.
- [x] Add network boundary tests.

## 11. Secret Access

- [x] Keep user secret access in CLI.
- [x] Apply CLI secret policy.
- [x] Avoid sending secrets to Runtime by default.
- [x] Redact secrets from CLI observability.
- [x] Redact secrets from Runtime observability.
- [x] Add secret boundary tests.

## 12. Tool Execution

- [x] Keep tool execution in CLI.
- [x] Interpret tool-call-like model output in CLI only.
- [x] Prevent Runtime tool execution.
- [x] Prevent automatic tool execution from Runtime output.
- [x] Add tool boundary tests.

## 13. Shell And Process Execution

- [x] Keep shell/process execution in CLI.
- [x] Apply CLI process policy.
- [x] Prevent Runtime shell execution.
- [x] Prevent Runtime process execution.
- [x] Add process boundary tests.

## 14. Agent Orchestration

- [x] Keep agent planning in CLI.
- [x] Keep agent tool loops in CLI.
- [x] Keep workspace mutation in CLI.
- [x] Allow repeated Runtime API calls.
- [x] Prevent Runtime agent orchestration.
- [x] Add agent boundary tests.

## 15. Prompt Assembly

- [x] Assemble file context in CLI.
- [x] Assemble workspace context in CLI.
- [x] Assemble Git context in CLI.
- [x] Assemble tool output context in CLI.
- [x] Assemble network retrieval context in CLI.
- [x] Assemble user interaction context in CLI.
- [x] Pass explicit prompt/context to Runtime.
- [x] Add prompt assembly tests.

## 16. Chat Template Boundary

- [x] Allow CLI to send plain text.
- [x] Allow CLI to send chat messages.
- [x] Allow Runtime to apply authorized chat template.
- [x] Allow CLI to pre-render prompt text.
- [x] Make boundary explicit.
- [x] Prevent Runtime fetching templates from arbitrary files/network.
- [x] Add chat template boundary tests.

## 17. Runtime Sessions From CLI

- [x] Allow CLI to create Runtime sessions.
- [x] Allow CLI to close Runtime sessions.
- [x] Keep CLI metadata separate from Runtime session state.
- [x] Prevent Runtime sessions from storing workspace state.
- [x] Prevent Runtime sessions from storing Git/tool/secret state.
- [x] Add session boundary tests.

## 18. Streaming

- [x] Consume Runtime streaming events in CLI.
- [x] Render token output in CLI.
- [x] Render decoded text in CLI.
- [x] Render progress where desired.
- [x] Store transcript in CLI where policy allows.
- [x] Preserve Runtime event order.
- [x] Add streaming tests.

## 19. Cancellation

- [x] Expose user cancellation in CLI.
- [x] Call Runtime cancellation for inference work.
- [x] Cancel CLI-owned file/Git/network/tool work separately.
- [x] Report Runtime cancellation limitations.
- [x] Add cancellation boundary tests.

## 20. Diagnostics

- [x] Display Runtime diagnostics in CLI.
- [x] Keep Runtime diagnostics redacted.
- [x] Optionally enrich with CLI command metadata.
- [x] Keep workspace path context CLI-side.
- [x] Prevent Runtime owning CLI context.
- [x] Add diagnostics boundary tests.

## 21. Configuration

- [x] Define CLI config ownership.
- [x] Define Runtime policy ownership.
- [x] Keep default model alias in CLI.
- [x] Keep default generation parameter profiles in CLI or explicit Runtime policy.
- [x] Keep workspace behavior in CLI.
- [x] Keep tool policy in CLI.
- [x] Keep network policy in CLI.
- [x] Keep secret providers in CLI.
- [x] Keep output formatting in CLI.
- [x] Add config boundary tests.

## 22. Model Aliases

- [x] Resolve friendly aliases in CLI.
- [x] Pass resolved ModelRef to Runtime.
- [x] Prevent aliases from bypassing trust.
- [x] Prevent aliases from bypassing loading.
- [x] Add alias tests.

## 23. Local Model Files

- [x] Resolve local paths in CLI.
- [x] Convert local model input to authorized artifact source reference.
- [x] Prevent Runtime directory scanning.
- [x] Preserve artifact validation.
- [x] Preserve trust validation.
- [x] Add local model file tests.

## 24. Serve Mode Boundary

- [x] Define serve mode as CLI or companion boundary.
- [x] Ensure serve mode calls Runtime Inference API.
- [x] Prevent server from bypassing Runtime validation.
- [x] Keep HTTP API details out of this change.
- [x] Add serve boundary tests.

## 25. Error Model

- [x] Define cli-command-invalid error.
- [x] Define cli-prompt-input-invalid error.
- [x] Define cli-file-read-failed error.
- [x] Define cli-workspace-access-denied error.
- [x] Define cli-git-failed error.
- [x] Define cli-network-denied error.
- [x] Define cli-secret-unavailable error.
- [x] Define cli-tool-failed error.
- [x] Define cli-shell-denied error.
- [x] Define cli-model-alias-not-found error.
- [x] Define cli-model-reference-invalid error.
- [x] Define cli-runtime-unavailable error.
- [x] Define cli-runtime-request-failed error.
- [x] Define cli-stream-interrupted error.
- [x] Define cli-cancellation-requested status.
- [x] Define cli-diagnostics-redacted status.
- [x] Define cli-boundary-violation error.
- [x] Define internal-cli error.
- [x] Preserve Runtime structured errors.

## 26. Observability

- [x] Emit CLI command received observation.
- [x] Emit CLI command parsed observation.
- [x] Emit file context collected observation.
- [x] Emit Git context collected observation.
- [x] Emit tool executed observation.
- [x] Emit Runtime request submitted observation.
- [x] Emit stream rendered observation.
- [x] Emit CLI command completed observation.
- [x] Emit CLI command failed observation.
- [x] Redact raw prompts by default.
- [x] Redact secrets by default.
- [x] Redact file contents by default.
- [x] Redact tokens by default.
- [x] Redact Runtime handles by default.

## 27. Security Boundary

- [x] Prevent CLI authority from becoming Runtime ambient authority.
- [x] Ensure all CLI-to-Runtime data is explicit.
- [x] Prevent Runtime receiving filesystem authority.
- [x] Prevent Runtime receiving network authority.
- [x] Prevent Runtime receiving secret authority.
- [x] Prevent Runtime receiving tool authority.
- [x] Add security boundary tests.

## 28. Browser Boundary

- [x] Document CLI is primarily native.
- [x] Keep Runtime Inference API platform-neutral.
- [x] Do not define browser CLI.
- [x] Add notes for future browser client.
- [x] Add wasm32 Runtime boundary check where feasible.

## 29. Tests

- [x] Test `run` sends prompt payload, not filesystem authority.
- [x] Test `chat` uses Runtime sessions without storing workspace state in Runtime.
- [x] Test CLI file read stays in CLI.
- [x] Test Git stays in CLI.
- [x] Test network stays in CLI.
- [x] Test secrets stay in CLI.
- [x] Test tools stay in CLI.
- [x] Test shell stays in CLI.
- [x] Test generated tool-call-like text is not executed by Runtime.
- [x] Test model alias does not bypass Runtime validation.
- [x] Test Provider diagnostics are redacted.
- [x] Test Device diagnostics are redacted.
- [x] Test Runtime structured error is preserved by CLI.
- [x] Test CLI cancellation calls Runtime cancellation.
- [x] Test CLI observability redaction.

## 30. Documentation

- [x] Document `magnetar-cli` inference boundary.
- [x] Document CLI-owned responsibilities.
- [x] Document Runtime-owned responsibilities.
- [x] Document command conceptual flows.
- [x] Document prompt assembly boundary.
- [x] Document workspace/file boundary.
- [x] Document Git boundary.
- [x] Document network boundary.
- [x] Document secret boundary.
- [x] Document tool/shell boundary.
- [x] Document agent boundary.
- [x] Document session boundary.
- [x] Document streaming boundary.
- [x] Document diagnostics boundary.
- [x] Document configuration boundary.
- [x] Document serve boundary.
- [x] Document security boundary.
- [x] Document non-goals.

## 31. Final Validation

- [x] Run formatting.
- [x] Run compilation checks.
- [x] Run Clippy.
- [x] Run complete tests.
- [x] Run CLI boundary tests.
- [x] Run Runtime Inference API tests.
- [x] Run Session tests.
- [x] Run Generation tests.
- [x] Run Tokenizer tests.
- [x] Run Observability tests.
- [x] Run OpenSpec validation.
- [x] Run coverage validation.
- [x] Verify Runtime is inference-only.
- [x] Verify CLI owns workspace/tools/agent concerns.
- [x] Verify CLI authority is not delegated to Runtime.
