## Why

The post-audit review found that the first-native vertical slice still has implementation bypasses around model execution, Qwen graph authority, incremental KV decode, CLI/runtime separation, and evidence production. This change aligns implementation and OpenSpec contracts so the shipped path demonstrates the architecture rather than a collection of fixtures.

## What Changes

- **BREAKING** Remove the normal-production logits injection seam from `RuntimeInferenceApi`; synthetic logits remain allowed only under explicit test/conformance support.
- Execute first-native generation through a Runtime-owned `ModelInstance` and phase-specific `PreparedExecutionPlan`.
- Route every executable Qwen operator through Kernel Registry resolution, prepared kernel identity, Provider dispatch, and completion evidence.
- Implement true prefill plus incremental decode over Runtime-owned KV cache state; decode receives only newly admitted token input for the first baseline.
- Make the Qwen WASM Component artifact authoritative for first-native E2E graph semantics.
- Cut `magnetar-cli` over to production Runtime APIs with real `model_ref` resolution and no dependency on `e2e_conformance`.
- Replace declarative E2E evidence flags with observations emitted by the Runtime, Kernel Registry, Provider, plan executor, KV subsystem, and Sampling.
- Keep `e2e_conformance` as a test harness only; production modules must not depend on it for generation behavior.
- Restore and keep docs quality gates green, and keep `SECURITY.md` synchronized with implemented Component Runtime controls.

## Capabilities

### New Capabilities

- None.

### Modified Capabilities

- `generation`: Require first-native logits to originate from actual model graph execution through prepared plans, and require incremental KV decode for the first profile.
- `kernel-execution-plan`: Require first-native generation steps to bind to compatible phase-specific `PreparedExecutionPlan` generations and reject invalidated plans.
- `kernel-registry`: Require first-native Qwen E2E operators to resolve and dispatch through Kernel Registry and Provider without direct Reference CPU bypasses.
- `qwen-model-component`: Require the Qwen WASM Component artifact to be the authoritative source of first-native graph semantics.
- `inference-api`: Remove production caller-supplied logits/forward callbacks from Runtime generation APIs.
- `cli-boundary`: Require `magnetar-cli` to use `RuntimeInferenceApi` and model reference resolution without `e2e_conformance` or direct Provider/Kernel logic.
- `e2e-conformance`: Require authoritative E2E evidence to be observational and produced by real runtime layers.
- `first-native-implementation-cut`: Update bypass inventory and final cut criteria for F01-F06.
- `quality`: Require docs CI to cover default and Wasmtime-relevant documentation builds.
- `release-security`: Require security documentation to reflect implemented Wasmtime fuel, deadline, resource-limit, no-ambient-WASI, and trust controls.

## Impact

- Affected crates: `magnetar-runtime`, `magnetar-cli`.
- Affected runtime areas: inference API, generation loop, model loading/instance lifecycle, Qwen component integration, execution graph planning, kernel registry/dispatch, Reference CPU Provider, KV cache, observability/evidence, E2E conformance.
- Affected docs/specs: OpenSpec deltas, `SECURITY.md`, quality documentation.
- Compatibility: production APIs that let callers inject model logits or substitute model execution will be removed or gated as test-only; tests and fixtures must migrate to explicit conformance/test support.
