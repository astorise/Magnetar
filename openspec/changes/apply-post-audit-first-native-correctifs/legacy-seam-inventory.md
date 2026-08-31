# Legacy Seam Inventory

This inventory was captured while applying `apply-post-audit-first-native-correctifs`.

## RuntimeGenerationExecutor

- `magnetar-runtime/src/inference_api.rs`: crate-internal legacy trait and shared wrapper still exist.
- `magnetar-runtime/src/runtime.rs`: `RuntimeBuilder::generation_executor` and `Runtime::generation_executor` still wire the crate-internal executor.
- `magnetar-runtime/src/first_native_runtime.rs`: first-native runtime implementation still uses `E2eRuntimeGenerationExecutor` while the final non-recompute prepared-plan model executor is being completed.
- `magnetar-runtime/src/tests.rs`: synthetic executors remain as unit-test support.
- `magnetar-runtime/src/first_native_implementation_cut.rs`: static inventory and regression guards intentionally reference the seam.

## RuntimeGenerationStep::new(logits, ...)

- `magnetar-runtime/src/first_native_runtime.rs`: first-native runtime still returns logits through the legacy generation step.
- `magnetar-runtime/src/tests.rs`: unit tests still build synthetic logits.
- `magnetar-runtime/src/first_native_implementation_cut.rs`: static inventory intentionally references the constructor.

## Full-History Decode

- `magnetar-runtime/src/first_native_runtime.rs`: `E2eRuntimeGenerationExecutor::execute_generation_step` still materializes `request.input_token_ids + generated_tokens` before logits production. KV lifecycle is Runtime-owned, but the Reference CPU attention/operator path still needs a KV-aware ABI before this seam can be removed.

## Not Present In CLI

No `magnetar-cli/src/*.rs` file contains:

- `RuntimeGenerationExecutor`
- `RuntimeGenerationStep::new`
- `.generation_executor(`
- `run_first_native_fixture_generation`
- `e2e_conformance`

The static regression test in `first_native_implementation_cut` enforces this boundary.
