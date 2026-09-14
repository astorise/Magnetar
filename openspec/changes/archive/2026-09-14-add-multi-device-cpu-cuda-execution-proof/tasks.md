## 1. Research

- [x] 1.1 Explored `multi_device_placement.rs`, `execution_graph.rs`, `kernel_dispatch.rs`, `model_instance.rs`, both provider crates' `devices()`, existing CPU+CUDA-together test precedent, and `kernel_registry.rs`/`kernel.rs` selection logic to find the real dispatch seam for a genuine multi-Provider proof, rather than guessing at one.
- [x] 1.2 Confirmed via real, fetched source reading (not assumption) that `KernelDispatchPlan` already carries a per-invocation `provider`/`device` binding, that `Runtime::register_provider` has no arity limit, and that `KernelRegistry::select`'s candidate filtering has no per-Provider exclusion tied to `ResourceAffinity`.

## 2. Implementation

- [x] 2.1 `integration-tests/multi-device-cpu-cuda`: new crate, depending on `magnetar-runtime`, `magnetar-provider-cpu`, `magnetar-provider-cuda`.
- [x] 2.2 `dispatch_add`: a Provider-agnostic helper driving the full generic Kernel dispatch contract for one `add` invocation, parameterized by a `Stage` (executor, affinity, Provider/Device binding, output placement, real Kernel memory class).
- [x] 2.3 The real, two-stage test: `a + b` on Reference CPU, explicit host round trip, `(a + b) + c` on real CUDA hardware, verified against a hand-computed expected value.
- [x] 2.4 Fixed, found by running the first version against real hardware: explicit per-Provider candidate selection from `selection.candidates` (not `selection.selected`, which does not respect the request's intended Provider); a per-stage `KernelMemoryClass` (CPU advertises `Host`, CUDA advertises `Device` -- a shared constant was wrong); an explicit `dtype_requirements` on the request (CUDA's `add`/`add-half` share one `OperatorId`, and an empty requirement left both nominally compatible).
- [x] 2.5 An independent design review (before compiling) and this repository's real compiler/test feedback (after) both found real issues, both fixed: an overstated doc-comment claim about `MultiDevicePlacementPlan` "driving" execution (it does not -- corrected), and a redundant `add_binding` call duplicating `PipelineStage`'s own binding (removed).
- [x] 2.6 `magnetar-runtime`'s existing `multi_device_placement` types (`DeviceSet`, `MultiDevicePlacementPlan`, `PipelineStage`, `StageMovementEdge`) built from this run's own real Device metadata/results after both stages succeeded, progressed through their real state machine to `Ready`.

## 3. Tests

- [x] 3.1 `cargo test` passed, re-run 3x consecutively for stability on this development machine's real NVIDIA GPU.
- [x] 3.2 `cargo fmt`/`cargo clippy --all-targets -- -D warnings` clean.
- [x] 3.3 Verified on GitHub Actions' `ubuntu-latest` runner (no GPU): the test gracefully skips (0.00s), confirming the tolerance path works, not just the hardware path.
- [x] 3.4 Verified on the `arc-gpu-magnetar` self-hosted real-GPU runner via `gpu-runner-smoke.yml` (`workflow_dispatch`, run `34877112076`): the test genuinely executed (0.46s, not skipped) and passed.

## 4. CI wiring

- [x] 4.1 `.github/workflows/quality.yml`: `submodule-integration`'s sweep includes the new crate.
- [x] 4.2 `.github/workflows/gpu-runner-smoke.yml`: a new step runs it against real hardware, right after the existing CUDA Provider step (both already have `providers/cpu`/`providers/cuda` checked out).
- [x] 4.3 Found and fixed a real, separate, pre-existing gap while verifying 4.2: the workflow never checked out `loaders/gguf`/`formats/gguf`, both real dependencies of the already-existing `integration-tests/production-loading` step immediately after -- confirmed by the first real dispatch (`34877112076`) failing there with "failed to read formats/gguf/Cargo.toml", unrelated to this change's own content. Fixed and re-verified with a second real dispatch (`34882463403`).

## 5. Documentation

- [x] 5.1 `openspec validate add-multi-device-cpu-cuda-execution-proof --strict` passes.
- [x] 5.2 README.md's top-level scope-charter status updated once archived: multi-device execution's real, scoped status (a real CPU+CUDA foundational proof, not production ModelInstance-level placement).
