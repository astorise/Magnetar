## Why

MAG-07 from the Tachyon integration audit (`docs/audits/audit-magnetar-integration-tachyon-2026-09-13.md`) flagged that no test in the repository registers two distinct, real Components implementing the same `model-component-graph-producer` contract side by side and proves they behave independently -- `wire-generic-inference-component-runtime`'s own registry tests all exercise a single artifact (`QWEN_REAL_COMPONENT_BYTES`) reused under different digests/trust decisions, which proves the registry accepts a caller-supplied digest but never proves it actually serves *two different* Components at once with no hidden model-family branching left anywhere in the call path.

## What Changes

- A second, real, structurally distinct WASM Component fixture is added: `magnetar-runtime/fixtures/components/synthetic-minimal.component.wasm`, implementing the identical `magnetar:model-component-graph@1.2.0` `model-component-graph-producer` world as the real Qwen Component, but with a deliberately degenerate graph (`embedding -> rmsnorm -> matmul`, every decoder layer omitted) so its node count and operator-sequence hash can never coincide with Qwen's. Its manifest (`synthetic-minimal.component.wasm.magnetar-component.yaml`) mirrors the real Qwen fixture's manifest format exactly, labeled as a test fixture, never production.
- `register_inference_component_artifact_enforces_trust_is_idempotent_and_matches_the_singleton_path` (the existing consolidated registry test -- consolidated for the documented reason that splitting digest-sharing assertions across separate `#[test]` functions races under parallel test execution) gains a final section: it also registers the synthetic Component under its own real digest, builds its graphs, and asserts they differ from the real Qwen Component's (node counts and operator-sequence hashes), and that building them did not disturb the real Qwen Component's own continued correctness.
- **BREAKING**: none. Test-only fixture and test-only assertions; no production code path changes.

## Capabilities

### New Capabilities
(none)

### Modified Capabilities
- `model-component-graph-contract`: "Multiple Registered Components May Coexist Under Caller-Supplied Trust" gains a scenario proving two structurally distinct, simultaneously registered Components produce genuinely different graphs and do not disturb each other -- closing MAG-07.

## Impact

- `magnetar-runtime/fixtures/components/`: two new checked-in files (`synthetic-minimal.component.wasm`, `synthetic-minimal.component.wasm.magnetar-component.yaml`).
- `magnetar-runtime/src/first_native_runtime.rs`: two new `#[cfg(test)]`-gated `include_bytes!` constants, mirroring the existing `QWEN_REAL_COMPONENT_BYTES`/`_MANIFEST_BYTES` pattern.
- `magnetar-runtime/src/first_native_runtime/tests.rs`: the existing consolidated registry test gains new assertions; no new `#[test]` function (avoids re-introducing the documented digest-sharing race).
- README/`docs/audits/audit-magnetar-integration-tachyon-2026-09-13.md`: MAG-07 moves from open to closed.
