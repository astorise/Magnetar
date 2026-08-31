# Legacy Seam Inventory

This inventory was captured while applying `apply-post-audit-first-native-correctifs`.

## RuntimeGenerationExecutor

- `magnetar-runtime/src/inference_api.rs`: public legacy trait and shared wrapper still exist.
- `magnetar-runtime/src/runtime.rs`: `RuntimeBuilder::generation_executor` and `Runtime::generation_executor` still wire the legacy executor.
- `magnetar-runtime/src/first_native_runtime.rs`: first-native runtime implementation still uses `E2eRuntimeGenerationExecutor` while the real prepared-plan model executor is being completed.
- `magnetar-runtime/src/tests.rs`: synthetic executors remain as unit-test support.
- `magnetar-runtime/src/first_native_implementation_cut.rs`: static inventory and regression guards intentionally reference the seam.

## RuntimeGenerationStep::new(logits, ...)

- `magnetar-runtime/src/first_native_runtime.rs`: first-native runtime still returns logits through the legacy generation step.
- `magnetar-runtime/src/tests.rs`: unit tests still build synthetic logits.
- `magnetar-runtime/src/first_native_implementation_cut.rs`: static inventory intentionally references the constructor.

## Not Present In CLI

No `magnetar-cli/src/*.rs` file contains:

- `RuntimeGenerationExecutor`
- `RuntimeGenerationStep::new`
- `.generation_executor(`
- `run_first_native_fixture_generation`
- `e2e_conformance`

The static regression test in `first_native_implementation_cut` enforces this boundary.
