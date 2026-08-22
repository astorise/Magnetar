# Tasks

## 1. Dependency and Feature Setup

- [x] Add the selected WebAssembly Component Model engine dependency.
- [x] Prefer Wasmtime as the first implementation unless evidence requires a
      different engine.
- [x] Add required Component Model features.
- [x] Add async support if required by host calls.
- [x] Add test fixture build dependencies.
- [x] Define a Cargo feature for the concrete engine if appropriate.
- [x] Decide whether the Wasmtime engine feature is enabled by default.
- [x] Document the feature policy.

## 2. ComponentEngine Adapter Module

- [x] Create a concrete engine adapter module.
- [x] Implement the engine-neutral `ComponentEngine` abstraction.
- [x] Keep concrete Wasmtime types inside the adapter module.
- [x] Avoid exposing Wasmtime types through public Magnetar APIs.
- [x] Add compile-time checks where practical to prevent leakage.
- [x] Keep engine-specific configuration private or wrapped in Magnetar-owned
      config types.

## 3. Engine Configuration

- [x] Configure WebAssembly Component Model support.
- [x] Configure async support where needed.
- [x] Configure resource-limit support where feasible.
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
- [x] Prepare for future artifact model integration.

## 5. Component Preparation

- [x] Implement engine validation of Component bytes.
- [x] Implement Component preparation/compilation.
- [x] Store prepared Component state opaquely.
- [x] Associate prepared state with Component definition identity.
- [x] Allow multiple instances from one prepared Component.
- [x] Normalize preparation failures.
- [x] Add preparation tests.

## 6. WIT Import/Export Inspection

- [x] Inspect Component imports.
- [x] Inspect Component exports.
- [x] Map imported interfaces to Magnetar Component import requirements.
- [x] Map exported interfaces to Magnetar Component export descriptions.
- [x] Preserve package/interface/version identity.
- [x] Reject unsupported mandatory imports.
- [x] Do not auto-link exports globally.

## 7. Contract Validation

- [x] Validate required imports before instantiation.
- [x] Validate import compatibility.
- [x] Validate export compatibility where required.
- [x] Distinguish validation errors from engine preparation errors.
- [x] Add tests for malformed WIT.
- [x] Add tests for missing required import.
- [x] Add tests for unsupported import version.

## 8. Link Plan Translation

- [x] Use Runtime-owned Component Link Plan as the source of truth.
- [x] Translate approved Link Plan entries into engine-native linker entries.
- [x] Reject imports absent from the Link Plan.
- [x] Reject unauthorized imports.
- [x] Keep engine-native Linker private.
- [x] Ensure Link Plan is immutable for one instantiation.
- [x] Add Link Plan tests.

## 9. Capability Host Adapter Infrastructure

- [x] Implement host adapter infrastructure for linked Capabilities.
- [x] Map WIT imports to Runtime endpoints.
- [x] Ensure Capability linking does not select a Provider.
- [x] Ensure host adapter does not expose Provider handles.
- [x] Ensure host adapter does not expose Device handles.
- [x] Support async host operations where needed by failing closed when no typed
      async adapter exists.
- [x] Normalize host adapter errors.

## 10. Initial Test Capability

- [x] Define a minimal test Capability WIT fixture.
- [x] Implement a Runtime host adapter for the test Capability.
- [x] Build a test Component importing that Capability.
- [x] Invoke the Component export end-to-end.
- [x] Verify host call result crosses the WASM boundary correctly.
- [x] Verify the test path uses the same Link Plan mechanism as real
      Capabilities.

## 11. Compute Capability Fixture

- [x] Add a minimal fixture importing `magnetar:compute/run@2.0.0` where
      practical.
- [x] Link the fixture to a Runtime Compute endpoint.
- [x] Verify linking Compute does not select a Provider.
- [x] Verify Provider resolution happens only when Compute work is submitted.
- [x] Keep full Compute execution coverage limited to what current Runtime
      fixtures can support.
- [x] Defer complete model inference fixtures to later changes.

## 12. Instance Creation

- [x] Instantiate prepared Components.
- [x] Allocate Runtime-owned ComponentInstanceId.
- [x] Associate engine Store with ComponentInstanceId.
- [x] Preserve definition identity.
- [x] Support multiple instances per definition.
- [x] Reject invocation before instance readiness.
- [x] Add multiple-instance tests.

## 13. Store Isolation

- [x] Ensure each Component Instance receives isolated mutable Store state.
- [x] Prevent accidental shared state between instances.
- [x] Validate instance-local state with fixture Components.
- [x] Ensure Runtime invocation context is instance-specific.
- [x] Ensure cancellation state is instance-specific where required.
- [x] Ensure resource tables are not shared accidentally.

## 14. Invocation

- [x] Implement invocation through typed or contract-specific adapters.
- [x] Avoid making a generic string/dynamic invocation ABI the canonical API.
- [x] Support invocation of test Component exports.
- [x] Support returning primitive values in fixtures.
- [x] Support host-call round trips.
- [x] Normalize invocation errors.
- [x] Add invocation success tests.

## 15. Async Host Calls

- [x] Enable async host calls in the concrete engine where required.
- [x] Verify async host-call scope does not expose a concrete async runtime.
- [x] Verify long-running host operations without typed adapters fail closed
      rather than blocking an engine thread unnecessarily.
- [x] Avoid exposing a concrete async runtime in public API.
- [x] Defer async host-call test fixture until a typed async Runtime adapter is
      introduced.

## 16. WASI Fail-Closed Behavior

- [x] Do not install broad default WASI.
- [x] Reject or fail Components requiring unauthorized WASI imports.
- [x] Add fixture requesting filesystem without authorization.
- [x] Add fixture requesting environment without authorization where practical.
- [x] Verify such imports fail to link or instantiate.
- [x] Document how authorized WASI will be handled by a later scoping change.

## 17. Resource Limits

- [x] Map Magnetar resource-limit config to engine mechanisms where possible.
- [x] Support memory ceilings if feasible.
- [x] Support deadline or execution budget if feasible.
- [x] Support maximum concurrent invocation policy at Runtime level.
- [x] Fail closed when required limits cannot be enforced.
- [x] Add tests for unsupported required limit.
- [x] Add tests for enforced limit where practical.

## 18. Interruption

- [x] Implement Runtime-requested interruption using engine-supported
      mechanisms.
- [x] Support cancellation-triggered interruption.
- [x] Support shutdown-triggered interruption.
- [x] Support deadline-triggered interruption where feasible.
- [x] Normalize interruption result.
- [x] Keep fuel/epoch details private.
- [x] Add interruption tests.

## 19. Trap Normalization

- [x] Map engine traps to stable Component trap errors.
- [x] Distinguish trap from interruption.
- [x] Distinguish trap from link failure.
- [x] Distinguish trap from host Capability failure.
- [x] Redact diagnostic messages.
- [x] Add trapping Component fixture.
- [x] Verify public error does not expose engine-native trap type.

## 20. Engine Error Normalization

- [x] Normalize validation failures.
- [x] Normalize preparation failures.
- [x] Normalize linker failures.
- [x] Normalize instantiation failures.
- [x] Normalize invocation failures.
- [x] Normalize resource-limit failures.
- [x] Normalize internal engine failures.
- [x] Preserve source chaining internally where safe.

## 21. Resource Table Handling

- [x] Keep engine resource table entries private.
- [x] Validate WIT resource ownership on host calls by rejecting resource imports
      without explicit Runtime mappings.
- [x] Prevent cross-instance resource forgery by failing closed before resource
      table entries are exposed.
- [x] Release instance-owned resources on destruction.
- [x] Preserve Runtime-owned resources according to Runtime lifecycle by not
      transferring ownership through unsupported WIT resources.
- [x] Add resource ownership tests where fixtures allow.

## 22. Concurrency

- [x] Define per-instance invocation synchronization.
- [x] Prevent unsafe concurrent Store mutation.
- [x] Allow multiple independent instances to run concurrently when supported by
      the synchronous manager boundary.
- [x] Add tests for serialized same-instance calls where applicable.
- [x] Add tests for independent instance execution where practical.
- [x] Document current concurrency limits.

## 23. Component Destruction

- [x] Implement instance destruction.
- [x] Prevent invocation after destruction.
- [x] Release engine-owned Store state.
- [x] Release instance-local WIT resources.
- [x] Preserve independently owned Runtime resources.
- [x] Add destruction tests.
- [x] Add Runtime shutdown integration test.

## 24. Runtime Shutdown Integration

- [x] Prevent new Component invocations during shutdown.
- [x] Drain or interrupt active Component invocations according to policy.
- [x] Destroy active instances.
- [x] Release prepared Component state where appropriate.
- [x] Normalize shutdown-related errors.
- [x] Verify shutdown does not require generic Component `stop`.

## 25. Observability

- [x] Emit Component validation observations.
- [x] Emit preparation observations.
- [x] Emit link-plan observations.
- [x] Emit instantiation observations.
- [x] Emit invocation observations.
- [x] Emit trap observations.
- [x] Emit interruption observations.
- [x] Emit resource-limit observations.
- [x] Redact engine-native handles.
- [x] Ensure observability failures do not alter execution.

## 26. Test Fixture Build

- [x] Add reproducible fixture source files.
- [x] Add documented commands to build fixtures.
- [x] Decide whether fixtures are checked in as `.wasm` artifacts, built during
      tests, or both.
- [x] Ensure CI can build or validate fixtures.
- [x] Avoid requiring network access during fixture tests.
- [x] Avoid requiring Tachyon during fixture tests.

## 27. End-to-End Component Tests

- [x] Test valid Component preparation.
- [x] Test valid Component instantiation.
- [x] Test valid Component invocation.
- [x] Test authorized host import.
- [x] Test missing import failure.
- [x] Test unauthorized import failure.
- [x] Test no ambient WASI.
- [x] Test trapping Component.
- [x] Test interruption where feasible.
- [x] Test multiple isolated instances.
- [x] Test destruction and post-destruction invocation failure.

## 28. Feature-Gated Tests

- [x] Keep engine-neutral tests runnable without concrete engine feature if the
      feature is optional.
- [x] Run Wasmtime end-to-end tests when the feature is enabled.
- [x] Ensure CI enables the concrete engine feature for at least one job.
- [x] Document local command for engine tests.

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
- [x] Document local fixture build commands.
- [x] Document no ambient WASI behavior.
- [x] Document limitations of the first implementation.
- [x] Document that Wasmtime is an implementation detail.
- [x] Update Component architecture documentation.
- [x] Update README if Component execution status changes.

## 31. CI

- [x] Ensure CI installs required toolchain for fixture compilation.
- [x] Ensure CI runs engine-neutral Component Runtime tests.
- [x] Ensure CI runs Wasmtime end-to-end tests.
- [x] Ensure CI validates WIT fixtures.
- [x] Ensure CI runs on Linux.
- [x] Ensure Windows and macOS compile the integration where supported.
- [x] Cache engine build artifacts where safe.
- [x] Avoid relying on external network during tests.

## 32. Security

- [x] Verify no ambient filesystem authority.
- [x] Verify no ambient network authority.
- [x] Verify no ambient environment variable access.
- [x] Verify no ambient process execution.
- [x] Verify no secret access without explicit link.
- [x] Verify Component cannot access native Provider handles.
- [x] Verify Component cannot access Device handles.
- [x] Verify Component cannot access engine Store handles.

## 33. Final Validation

- [x] Run formatting.
- [x] Run compilation checks.
- [x] Run Clippy.
- [x] Run complete tests.
- [x] Run Wasmtime-feature tests.
- [x] Run WIT validation.
- [x] Run OpenSpec validation.
- [x] Run coverage validation.
- [x] Verify Component fixtures execute successfully.
- [x] Verify unauthorized imports fail closed.
- [x] Verify traps normalize correctly.
- [x] Verify Wasmtime types remain private.
- [x] Verify Provider resolution is not performed during Component linking.
