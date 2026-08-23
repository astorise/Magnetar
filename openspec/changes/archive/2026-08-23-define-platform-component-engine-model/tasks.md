# Tasks

## 1. Current Engine Inventory

- [x] Inventory current ComponentEngine abstraction.
- [x] Inventory current Wasmtime-specific implementation.
- [x] Inventory public exports of Wasmtime-specific types.
- [x] Inventory Cargo features related to Component execution.
- [x] Inventory cfg attributes around Component engine code.
- [x] Identify any Wasmtime dependency compiled for wasm32 targets.
- [x] Identify assumptions that native engine behavior is universal.
- [x] Identify tests that assume only Wasmtime exists.

## 2. Platform Engine Model

- [x] Define platform-aware Component Engine model.
- [x] Define native Component Engine profile.
- [x] Define web Component Engine profile.
- [x] Define test Component Engine profile.
- [x] Define engine feature declarations.
- [x] Define engine profile compatibility.
- [x] Define unsupported engine behavior.
- [x] Document that Wasmtime is native implementation, not universal model.

## 3. Source Layout

- [x] Decide between flat module layout and nested component engine layout.
- [x] Keep `component.rs` as platform-neutral Component Runtime logic.
- [x] Keep Wasmtime-specific code in `component_wasmtime.rs` or equivalent.
- [x] Add web-specific code in `component_web.rs` or equivalent.
- [x] Avoid platform-specific code leaking into platform-neutral module.
- [x] Update crate root exports.
- [x] Add platform module documentation.

## 4. Target Gating

- [x] Gate Wasmtime implementation with native target cfg.
- [x] Gate web implementation with `target_arch = "wasm32"` cfg.
- [x] Ensure `wasmtime-component-engine` cannot force Wasmtime into browser
      build.
- [x] Ensure `web-component-engine` does not affect native builds unless
      explicitly designed.
- [x] Add compile-fail or cfg validation where feasible.
- [x] Add CI target check for wasm32 where feasible.

## 5. Cargo Feature Policy

- [x] Define `wasmtime-component-engine` feature behavior.
- [x] Define `web-component-engine` feature behavior.
- [x] Define default feature behavior.
- [x] Define impossible feature/target combinations.
- [x] Ensure feature names do not imply universal engine support.
- [x] Document feature matrix.

## 6. Native Engine Profile

- [x] Define native engine supported features.
- [x] Include Component preparation.
- [x] Include instantiation.
- [x] Include Link Plan translation.
- [x] Include async host calls where supported.
- [x] Include trap normalization.
- [x] Include interruption where supported.
- [x] Include resource limits where supported.
- [x] Include no ambient authority.
- [x] Include no broad WASI by default.
- [x] Add native profile tests.

## 7. Web Engine Profile

- [x] Define web engine supported features.
- [x] Include browser WebAssembly execution.
- [x] Include JavaScript-mediated host bindings.
- [x] Include browser-compatible Link Plan translation.
- [x] Include browser-compatible diagnostics.
- [x] Include browser-compatible observability.
- [x] Exclude Wasmtime.
- [x] Exclude native dynamic Provider loading.
- [x] Exclude native filesystem/process/secrets authority.
- [x] Exclude native pinned memory assumptions.
- [x] Add web profile tests where feasible.

## 8. Test Engine Profile

- [x] Define test engine capabilities.
- [x] Simulate preparation success.
- [x] Simulate preparation failure.
- [x] Simulate instantiation success.
- [x] Simulate instantiation failure.
- [x] Simulate unauthorized import.
- [x] Simulate trap.
- [x] Simulate interruption.
- [x] Simulate resource-limit failure.
- [x] Simulate destruction.
- [x] Use test engine in contract tests.

## 9. Engine Capability Model

- [x] Define ComponentEngineCapabilities by profile.
- [x] Include Component Model support indicator.
- [x] Include async host call support indicator.
- [x] Include interruption support indicator.
- [x] Include resource limit support indicator.
- [x] Include browser compatibility indicator.
- [x] Include native Provider endpoint support indicator.
- [x] Include controlled WASI support indicator.
- [x] Add capability compatibility tests.

## 10. Engine Selection

- [x] Define engine selection inputs.
- [x] Use target architecture.
- [x] Use enabled features.
- [x] Use Runtime configuration.
- [x] Use Component Artifact requirements.
- [x] Use engine capabilities.
- [x] Fail closed when no compatible engine exists.
- [x] Add native selection tests.
- [x] Add web selection tests where feasible.
- [x] Add no-compatible-engine tests.

## 11. Artifact Compatibility

- [x] Allow Component Artifacts to declare required engine profile.
- [x] Allow Component Artifacts to declare required engine features.
- [x] Reject native-only Component on web target.
- [x] Reject browser-only Component on native target where unsupported.
- [x] Reject Component requiring unavailable interruption support.
- [x] Reject Component requiring unavailable resource limits.
- [x] Add artifact-engine compatibility tests.

## 12. Link Plan Translation

- [x] Keep Link Plan Runtime-owned.
- [x] Define native Link Plan translation.
- [x] Define web Link Plan translation.
- [x] Ensure only authorized imports are linked.
- [x] Ensure web JS bindings do not bypass authority.
- [x] Ensure Capability linking still does not pin Provider directly.
- [x] Add Link Plan translation tests.

## 13. WASI And Ambient Authority

- [x] Preserve no ambient WASI by default.
- [x] Ensure native Wasmtime does not link broad WASI unless explicitly allowed.
- [x] Ensure web engine does not expose broad browser APIs by default.
- [x] Reject filesystem authority.
- [x] Reject network authority.
- [x] Reject secrets authority.
- [x] Reject process authority.
- [x] Reject Git/workspace authority.
- [x] Add no-ambient-authority tests.

## 14. Native Provider Boundary

- [x] Document native Provider loading as native-only.
- [x] Ensure dynamic Provider loading is not required on web.
- [x] Ensure web build does not expose native Provider ABI loading.
- [x] Ensure ComponentEngine does not imply Provider ABI.
- [x] Add cfg tests where feasible.

## 15. Web Provider Future Placeholder

- [x] Document future WebGPU Provider possibility.
- [x] Document future pure-WASM Provider possibility.
- [x] Document future JS-mediated Provider possibility.
- [x] Keep these as placeholders.
- [x] Do not implement WebGPU Provider in this change.
- [x] Do not implement JS Provider ABI in this change.

## 16. Memory Manager Integration

- [x] Ensure web engine reports browser-compatible memory constraints.
- [x] Ensure native engine does not assume browser memory behavior.
- [x] Ensure browser engine does not assume native pinned memory.
- [x] Ensure engine capabilities can inform Memory Manager.
- [x] Add integration tests where feasible.

## 17. Diagnostics

- [x] Define no-compatible-engine diagnostic.
- [x] Define engine-profile-mismatch diagnostic.
- [x] Define engine-feature-unavailable diagnostic.
- [x] Define Wasmtime-unavailable diagnostic.
- [x] Define browser-engine-unavailable diagnostic.
- [x] Define host-binding-failed diagnostic.
- [x] Define platform-unsupported diagnostic.
- [x] Redact platform-private objects.
- [x] Add diagnostic tests.

## 18. Observability

- [x] Emit engine selected observation.
- [x] Emit engine rejected observation.
- [x] Emit platform unsupported observation.
- [x] Emit preparation start observation.
- [x] Emit preparation success observation.
- [x] Emit preparation failure observation.
- [x] Emit instantiation observation.
- [x] Emit Link Plan translation observation.
- [x] Emit host binding failure observation.
- [x] Emit trap observation.
- [x] Ensure observability failure does not alter engine behavior.

## 19. CI

- [x] Add native engine tests.
- [x] Add feature-gated Wasmtime tests.
- [x] Add wasm32 check target where feasible.
- [x] Ensure wasm32 check does not compile Wasmtime.
- [x] Ensure platform-neutral tests run without Wasmtime.
- [x] Ensure test engine works without native Wasmtime.
- [x] Document skipped browser runtime tests if full browser execution is not yet
      available.

## 20. Documentation

- [x] Document platform Component Engine model.
- [x] Document native profile.
- [x] Document web profile.
- [x] Document test profile.
- [x] Document Cargo feature matrix.
- [x] Document target cfg behavior.
- [x] Document no ambient authority on every engine.
- [x] Document Wasmtime as native implementation.
- [x] Document browser engine as separate implementation.
- [x] Document unsupported feature errors.

## 21. Final Validation

- [x] Run formatting.
- [x] Run native compilation checks.
- [x] Run wasm32 compilation check where feasible.
- [x] Run Clippy.
- [x] Run complete tests.
- [x] Run Component Runtime tests.
- [x] Run Wasmtime feature tests.
- [x] Run platform engine selection tests.
- [x] Run OpenSpec validation.
- [x] Run coverage validation.
- [x] Verify Wasmtime is not required on wasm32.
- [x] Verify web engine path is defined.
- [x] Verify public Component Runtime API remains platform-neutral.
