# Legacy Seam Inventory

This inventory was captured while applying `apply-post-audit-first-native-correctifs`.

## Runtime Model Execution Engine

- `magnetar-runtime/src/inference_api.rs`: normal generation uses a crate-internal Runtime-owned model execution engine; production callers cannot provide per-request logits or callbacks.
- `magnetar-runtime/src/runtime.rs`: `RuntimeBuilder::model_execution_engine` and `Runtime::model_execution_engine` are crate-internal wiring only.
- `magnetar-runtime/src/first_native_runtime.rs`: first-native runtime installs `E2eRuntimeModelExecutionEngine` for the fixture-backed production path.
- `magnetar-runtime/src/tests.rs`: synthetic executors remain unit-test support only.
- `magnetar-runtime/src/first_native_implementation_cut.rs`: static inventory and regression guards verify there is no removal-required P0 bypass.

## RuntimeModelExecutionStep::new(logits, ...)

- `magnetar-runtime/src/first_native_runtime.rs`: first-native runtime returns logits produced by the evidence-bearing model execution engine.
- `magnetar-runtime/src/tests.rs`: unit tests still build synthetic logits under test-only support.
- `magnetar-runtime/src/first_native_implementation_cut.rs`: static inventory verifies this is not exposed through public or CLI surfaces.

## Full-History Decode

- `magnetar-runtime/src/first_native_runtime.rs`: cleared from the normal first-native path. Decode now dispatches only the newly admitted token, applies RoPE with the absolute cache position, attends over historical plus new K/V state, appends Runtime-owned KV state, and is covered by a static no-full-history regression plus an oracle comparison.

## Not Present In CLI

No `magnetar-cli/src/*.rs` file contains:

- `RuntimeModelExecutionEngine`
- `RuntimeModelExecutionStep::new`
- `.model_execution_engine(`
- `run_first_native_fixture_generation`
- `e2e_conformance`

The static regression test in `first_native_implementation_cut` enforces this boundary.
