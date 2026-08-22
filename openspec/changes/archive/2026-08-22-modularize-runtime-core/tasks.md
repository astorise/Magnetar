# Tasks

## 1. OpenSpec Correction

- [x] Clarify that this change creates the Runtime Core foundation rather than only improving readability.
- [x] Document that future AI domains build above Runtime Core.
- [x] Document that `model`, `generation`, and `agent` are intentionally not created here.
- [x] Record `stabilize-component-runtime-boundary` as the follow-up change for the real WASM engine boundary.

## 2. Runtime Source Modularization

- [x] Reduce `magnetar-runtime/src/lib.rs` to a crate facade.
- [x] Declare explicit primary modules from the crate root.
- [x] Preserve explicit crate-root public re-exports for the existing stable API surface.
- [x] Move Component contracts and lifecycle management to `component`.
- [x] Move Capability identity, versioning, compatibility, and parsing to `capability`.
- [x] Move Device identity and metadata to `device`.
- [x] Move Resource Affinity, bindings, health state, and affinity resolution to `affinity`.
- [x] Move resolution policy and candidate selection types to `resolution`.
- [x] Move portable Compute descriptors, schemas, validation, WIT-facing constants, and Compute errors to `compute`.
- [x] Move memory and execution planning to `planning`.
- [x] Move scheduling, operation lifecycle, provider execution handles, cancellation, and scheduler errors to `scheduler`.
- [x] Move Provider metadata, registry, loading, native execution API trait, and Provider errors to `provider`.
- [x] Move Runtime configuration, builder, orchestration, graph validation, planning entry points, and lifecycle to `runtime`.
- [x] Integrate the existing observability exporter implementation under `observability/exporter.rs`.
- [x] Move Runtime events, metrics, traces, diagnostics, and observability WIT surface to `observability`.

## 3. Boundary Rules

- [x] Keep `magnetar-runtime` as one workspace crate.
- [x] Do not create empty `model`, `generation`, `inference`, `agent`, or `tool` modules.
- [x] Do not implement Wasmtime or a concrete Component engine in this change.
- [x] Do not reintroduce Backend or Plugin architecture.
- [x] Do not introduce Tachyon dependencies.
- [x] Keep Provider as the native extension boundary and Component as the portable extension lifecycle boundary.
- [x] Use `pub(crate)` for internal cross-module seams required by the refactor instead of making implementation helpers public API.

## 4. Tests And Validation

- [x] Move existing unit tests out of the crate root implementation and keep them under the facade-owned test module.
- [x] Preserve the existing observability exporter tests with the exporter module.
- [x] Run `cargo fmt`.
- [x] Run `cargo check`.
- [x] Run `cargo test`.
- [x] Run `cargo clippy --all-targets -- -D warnings`.
- [x] Run strict OpenSpec validation.
