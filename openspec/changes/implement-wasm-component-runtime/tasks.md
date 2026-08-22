# Tasks

## 1. Dependency and Feature Setup

- [x] Add the selected WebAssembly Component Model engine dependency.
- [x] Prefer Wasmtime as the first implementation unless evidence requires a
      different engine.
- [x] Add required Component Model features.
- [x] Add async support if required by host calls.
- [ ] Add test fixture build dependencies.
- [x] Define a Cargo feature for the concrete engine if appropriate.
- [x] Decide whether the Wasmtime engine feature is enabled by default.
- [x] Document the feature policy.

## 2. ComponentEngine Adapter Module

- [x] Create a concrete engine adapter module.
- [x] Implement the engine-neutral `ComponentEngine` abstraction.
- [x] Keep concrete Wasmtime types inside the adapter module.
- [x] Avoid exposing Wasmtime types through public Magnetar APIs.
- [ ] Add compile-time checks where practical to prevent leakage.
- [x] Keep engine-specific configuration private or wrapped in Magnetar-owned
      config types.

## 3. Engine Configuration

- [x] Configure WebAssembly Component Model support.
- [x] Configure async support where needed.
- [ ] Configure resource-limit support where feasible.
- [x] Configure interruption support where feasible.
- [x] Avoid broad default WASI configuration.
- [x] Avoid ambient host interface registration.
- [x] Provide safe defaults.

## 4. Component Byte Loading

- [x] Load Component bytes from a local path for initial implementation.
- [x] Validate file existence.
- [x] Validate readable bytes.
- [x] Reject non-Component artifacts.
- [x] Normalize load errors into Magnetar Component errors.
- [x] Keep artifact trust and digest validation out of this change.
- [ ] Prepare for future artifact model integration.

## 5. Component Preparation

- [x] Implement engine validation of Component bytes.
- [x] Implement Component preparation/compilation.
- [x] Store prepared Component state opaquely.
- [x] Associate prepared state with Component definition identity.
- [x] Allow multiple instances from one prepared Component.
- [x] Normalize preparation failures.
- [x] Add preparation tests.

## 6. WIT Import/Export Inspection

- [ ] Inspect Component imports.
- [ ] Inspect Component exports.
- [ ] Map imported interfaces to Magnetar Component import requirements.
- [ ] Map exported interfaces to Magnetar Component export descriptions.
- [ ] Preserve package/interface/version identity.
- [ ] Reject unsupported mandatory imports.
- [ ] Do not auto-link exports globally.

## 7. Contract Validation

- [ ] Validate required imports before instantiation.
- [ ] Validate import compatibility.
- [ ] Validate export compatibility where required.
- [ ] Distinguish validation errors from engine preparation errors.
- [ ] Add tests for malformed WIT.
- [ ] Add tests for missing required import.
- [ ] Add tests for unsupported import version.

## 8. Link Plan Translation

- [ ] Use Runtime-owned Component Link Plan as the source of truth.
- [ ] Translate approved Link Plan entries into engine-native linker entries.
- [ ] Reject imports absent from the Link Plan.
- [ ] Reject unauthorized imports.
- [ ] Keep engine-native Linker private.
- [ ] Ensure Link Plan is immutable for one instantiation.
- [ ] Add Link Plan tests.

## 9. Capability Host Adapter Infrastructure

- [ ] Implement host adapter infrastructure for linked Capabilities.
- [ ] Map WIT imports to Runtime endpoints.
- [ ] Ensure Capability linking does not select a Provider.
- [ ] Ensure host adapter does not expose Provider handles.
- [ ] Ensure host adapter does not expose Device handles.
- [ ] Support async host operations where needed.
- [ ] Normalize host adapter errors.

## 10. Initial Test Capability

- [ ] Define a minimal test Capability WIT fixture.
- [ ] Implement a Runtime host adapter for the test Capability.
- [ ] Build a test Component importing that Capability.
- [ ] Invoke the Component export end-to-end.
- [ ] Verify host call result crosses the WASM boundary correctly.
- [ ] Verify the test path uses the same Link Plan mechanism as real
      Capabilities.

## 11. Compute Capability Fixture

- [ ] Add a minimal fixture importing `magnetar:compute/run@2.0.0` where
      practical.
- [ ] Link the fixture to a Runtime Compute endpoint.
- [ ] Verify linking Compute does not select a Provider.
- [ ] Verify Provider resolution happens only when Compute work is submitted.
- [ ] Keep full Compute execution coverage limited to what current Runtime
      fixtures can support.
- [ ] Defer complete model inference fixtures to later changes.

## 12. Instance Creation

- [ ] Instantiate prepared Components.
- [ ] Allocate Runtime-owned ComponentInstanceId.
- [ ] Associate engine Store with ComponentInstanceId.
- [ ] Preserve definition identity.
- [ ] Support multiple instances per definition.
- [ ] Reject invocation before instance readiness.
- [ ] Add multiple-instance tests.

## 13. Store Isolation

- [ ] Ensure each Component Instance receives isolated mutable Store state.
- [ ] Prevent accidental shared state between instances.
- [ ] Validate instance-local state with fixture Components.
- [ ] Ensure Runtime invocation context is instance-specific.
- [ ] Ensure cancellation state is instance-specific where required.
- [ ] Ensure resource tables are not shared accidentally.

## 14. Invocation

- [ ] Implement invocation through typed or contract-specific adapters.
- [ ] Avoid making a generic string/dynamic invocation ABI the canonical API.
- [ ] Support invocation of test Component exports.
- [ ] Support returning primitive values in fixtures.
- [ ] Support host-call round trips.
- [ ] Normalize invocation errors.
- [ ] Add invocation success tests.

## 15. Async Host Calls

- [x] Enable async host calls in the concrete engine where required.
- [ ] Verify async host call completion.
- [ ] Verify long-running host operations do not block an engine thread
      unnecessarily.
- [ ] Avoid exposing a concrete async runtime in public API.
- [ ] Add async host-call test fixture if practical.

## 16. WASI Fail-Closed Behavior

- [x] Do not install broad default WASI.
- [ ] Reject or fail Components requiring unauthorized WASI imports.
- [ ] Add fixture requesting filesystem without authorization.
- [ ] Add fixture requesting environment without authorization where practical.
- [ ] Verify such imports fail to link or instantiate.
- [ ] Document how authorized WASI will be handled by a later scoping change.

## 17. Resource Limits

- [ ] Map Magnetar resource-limit config to engine mechanisms where possible.
- [ ] Support memory ceilings if feasible.
- [ ] Support deadline or execution budget if feasible.
- [ ] Support maximum concurrent invocation policy at Runtime level.
- [ ] Fail closed when required limits cannot be enforced.
- [ ] Add tests for unsupported required limit.
- [ ] Add tests for enforced limit where practical.

## 18. Interruption

- [ ] Implement Runtime-requested interruption using engine-supported
      mechanisms.
- [ ] Support cancellation-triggered interruption.
- [ ] Support shutdown-triggered interruption.
- [ ] Support deadline-triggered interruption where feasible.
- [ ] Normalize interruption result.
- [ ] Keep fuel/epoch details private.
- [ ] Add interruption tests.

## 19. Trap Normalization

- [ ] Map engine traps to stable Component trap errors.
- [ ] Distinguish trap from interruption.
- [ ] Distinguish trap from link failure.
- [ ] Distinguish trap from host Capability failure.
- [ ] Redact diagnostic messages.
- [ ] Add trapping Component fixture.
- [ ] Verify public error does not expose engine-native trap type.

## 20. Engine Error Normalization

- [ ] Normalize validation failures.
- [ ] Normalize preparation failures.
- [ ] Normalize linker failures.
- [ ] Normalize instantiation failures.
- [ ] Normalize invocation failures.
- [ ] Normalize resource-limit failures.
- [ ] Normalize internal engine failures.
- [ ] Preserve source chaining internally where safe.

## 21. Resource Table Handling

- [ ] Keep engine resource table entries private.
- [ ] Validate WIT resource ownership on host calls.
- [ ] Prevent cross-instance resource forgery.
- [ ] Release instance-owned resources on destruction.
- [ ] Preserve Runtime-owned resources according to Runtime lifecycle.
- [ ] Add resource ownership tests where fixtures allow.

## 22. Concurrency

- [ ] Define per-instance invocation synchronization.
- [ ] Prevent unsafe concurrent Store mutation.
- [ ] Allow multiple independent instances to run concurrently when supported.
- [ ] Add tests for serialized same-instance calls where applicable.
- [ ] Add tests for independent instance execution where practical.
- [ ] Document current concurrency limits.

## 23. Component Destruction

- [ ] Implement instance destruction.
- [ ] Prevent invocation after destruction.
- [ ] Release engine-owned Store state.
- [ ] Release instance-local WIT resources.
- [ ] Preserve independently owned Runtime resources.
- [ ] Add destruction tests.
- [ ] Add Runtime shutdown integration test.

## 24. Runtime Shutdown Integration

- [ ] Prevent new Component invocations during shutdown.
- [ ] Drain or interrupt active Component invocations according to policy.
- [ ] Destroy active instances.
- [ ] Release prepared Component state where appropriate.
- [ ] Normalize shutdown-related errors.
- [ ] Verify shutdown does not require generic Component `stop`.

## 25. Observability

- [ ] Emit Component validation observations.
- [ ] Emit preparation observations.
- [ ] Emit link-plan observations.
- [ ] Emit instantiation observations.
- [ ] Emit invocation observations.
- [ ] Emit trap observations.
- [ ] Emit interruption observations.
- [ ] Emit resource-limit observations.
- [ ] Redact engine-native handles.
- [ ] Ensure observability failures do not alter execution.

## 26. Test Fixture Build

- [ ] Add reproducible fixture source files.
- [ ] Add documented commands to build fixtures.
- [ ] Decide whether fixtures are checked in as `.wasm` artifacts, built during
      tests, or both.
- [ ] Ensure CI can build or validate fixtures.
- [ ] Avoid requiring network access during fixture tests.
- [ ] Avoid requiring Tachyon during fixture tests.

## 27. End-to-End Component Tests

- [ ] Test valid Component preparation.
- [ ] Test valid Component instantiation.
- [ ] Test valid Component invocation.
- [ ] Test authorized host import.
- [ ] Test missing import failure.
- [ ] Test unauthorized import failure.
- [ ] Test no ambient WASI.
- [ ] Test trapping Component.
- [ ] Test interruption where feasible.
- [ ] Test multiple isolated instances.
- [ ] Test destruction and post-destruction invocation failure.

## 28. Feature-Gated Tests

- [ ] Keep engine-neutral tests runnable without concrete engine feature if the
      feature is optional.
- [ ] Run Wasmtime end-to-end tests when the feature is enabled.
- [ ] Ensure CI enables the concrete engine feature for at least one job.
- [ ] Document local command for engine tests.

## 29. Public API Audit

- [x] Verify no public API exposes `wasmtime::Engine`.
- [x] Verify no public API exposes `wasmtime::Store`.
- [x] Verify no public API exposes `wasmtime::component::Component`.
- [x] Verify no public API exposes `wasmtime::component::Linker`.
- [x] Verify no public API exposes `wasmtime::component::Instance`.
- [x] Verify no public API exposes `wasmtime::Trap`.
- [x] Verify no public API exposes engine resource table handles.
- [x] Verify public APIs use Magnetar-owned abstractions.

## 30. Documentation

- [x] Document concrete engine usage.
- [x] Document feature flag policy.
- [ ] Document local fixture build commands.
- [x] Document no ambient WASI behavior.
- [x] Document limitations of the first implementation.
- [x] Document that Wasmtime is an implementation detail.
- [x] Update Component architecture documentation.
- [ ] Update README if Component execution status changes.

## 31. CI

- [ ] Ensure CI installs required toolchain for fixture compilation.
- [ ] Ensure CI runs engine-neutral Component Runtime tests.
- [ ] Ensure CI runs Wasmtime end-to-end tests.
- [ ] Ensure CI validates WIT fixtures.
- [ ] Ensure CI runs on Linux.
- [ ] Ensure Windows and macOS compile the integration where supported.
- [ ] Cache engine build artifacts where safe.
- [ ] Avoid relying on external network during tests.

## 32. Security

- [ ] Verify no ambient filesystem authority.
- [ ] Verify no ambient network authority.
- [ ] Verify no ambient environment variable access.
- [ ] Verify no ambient process execution.
- [ ] Verify no secret access without explicit link.
- [ ] Verify Component cannot access native Provider handles.
- [ ] Verify Component cannot access Device handles.
- [ ] Verify Component cannot access engine Store handles.

## 33. Final Validation

- [x] Run formatting.
- [x] Run compilation checks.
- [x] Run Clippy.
- [x] Run complete tests.
- [x] Run Wasmtime-feature tests.
- [ ] Run WIT validation.
- [x] Run OpenSpec validation.
- [ ] Run coverage validation.
- [ ] Verify Component fixtures execute successfully.
- [ ] Verify unauthorized imports fail closed.
- [ ] Verify traps normalize correctly.
- [x] Verify Wasmtime types remain private.
- [ ] Verify Provider resolution is not performed during Component linking.
