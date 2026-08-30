# Phase 0 Migration Inventory

This inventory records known migration paths that can make the first native
vertical slice look healthier than it is. These paths are allowed only as
explicit migration or focused-test utilities until the final cut removes or
isolates them from conformance.

## Freeze

Architecture Freeze #1 covers the first native local single-Device
model-execution path.

The freeze may be reopened only for:

- correctness blocker
- security blocker
- impossible implementation contract
- unavoidable ABI break
- contradiction between accepted specifications

Feature expansion and performance optimization do not block the first baseline.

## Bypasses

| Class | Location | Symbol | Disposition | Final cut |
| --- | --- | --- | --- | --- |
| CLI placeholder logits | `magnetar-cli/src/pipeline.rs` | removed; normal path calls `run_first_native_fixture_generation` | deprecated/removed from normal path | complete |
| caller forward callback | `magnetar-runtime/src/inference_api.rs` | `RuntimeGenerationExecutor` | non-conformant migration path | remove from normal API before PR 20 |
| caller-provided logits | `magnetar-runtime/src/inference_api.rs` | `RuntimeGenerationStep::new(logits, evidence)` | non-conformant migration path | replace with Runtime-owned model execution before PR 20 |
| direct Reference CPU execution | `magnetar-runtime/src/e2e_conformance.rs` | `e2e_forward_hidden_states` / `dispatch_matmul` | isolated test-only oracle path | not final E2E proof by itself |
| full-sequence decode shortcut | `magnetar-runtime/src/e2e_conformance.rs` | `E2eRuntimeGenerationExecutor::execute_generation_step` | tracked for removal | replace with incremental KV decode before PR 20 |
| Candle model execution | workspace manifests | no `candle` dependency present | absent/deprecated | keep absent from native profile |

The executable inventory lives in
`magnetar-runtime/src/first_native_implementation_cut.rs` and is covered by
unit tests.
