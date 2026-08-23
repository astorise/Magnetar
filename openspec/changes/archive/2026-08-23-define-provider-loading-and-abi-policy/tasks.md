# Tasks

## 1. Existing Loading Inventory

- [x] Inventory current dynamic Provider loading code.
- [x] Identify current factory symbol names.
- [x] Identify use of Rust trait objects across dynamic library boundaries.
- [x] Identify current memory ownership assumptions.
- [x] Identify current Provider metadata retrieval path.
- [x] Identify current Capability advertisement retrieval path.
- [x] Identify current Device metadata retrieval path.
- [x] Identify current health/status retrieval path.
- [x] Identify current execution API binding path.
- [x] Identify tests that rely on trait-object dynamic loading.

## 2. Loading Modes

- [x] Define built-in Provider loading mode.
- [x] Define dynamic-library Provider loading mode.
- [x] Define test-provider loading mode.
- [x] Define development-provider loading mode.
- [x] Ensure all modes register through Provider Registry.
- [x] Ensure built-in mode does not define dynamic ABI.
- [x] Document supported and unsupported modes.

## 3. Dynamic ABI Policy

- [x] Define that Rust trait objects are not the stable dynamic ABI.
- [x] Reject `Box<dyn Provider>` as the stable cross-library contract.
- [x] Reject `Arc<dyn Provider>` as the stable cross-library contract.
- [x] Reject raw `dyn Provider` pointers as the stable cross-library contract.
- [x] Define a stable ABI strategy.
- [x] Prefer C-compatible ABI unless implementation evidence chooses otherwise.
- [x] Document the reason for avoiding Rust trait objects at ABI boundary.

## 4. Factory Symbol

- [x] Define canonical factory symbol naming policy.
- [x] Include ABI major version in symbol name or returned descriptor.
- [x] Validate missing factory symbol.
- [x] Validate wrong factory symbol.
- [x] Validate duplicate/ambiguous symbols where applicable.
- [x] Add tests for missing symbol.
- [x] Add tests for unsupported symbol.

## 5. ABI Versioning

- [x] Define Provider ABI version type.
- [x] Define ABI major version semantics.
- [x] Define ABI minor version semantics.
- [x] Reject unsupported ABI major version.
- [x] Allow compatible minor version only when policy permits.
- [x] Keep ABI version separate from Provider version.
- [x] Keep ABI version separate from Capability version.
- [x] Add ABI version negotiation tests.

## 6. ABI Descriptor

- [x] Define ABI descriptor structure.
- [x] Include descriptor size or versioned layout guard.
- [x] Include ABI version.
- [x] Include required function pointers.
- [x] Include optional function pointers with feature flags.
- [x] Include Provider metadata function.
- [x] Include Capability advertisement function.
- [x] Include Device listing function.
- [x] Include status function.
- [x] Include execution function table or execution entrypoint.
- [x] Include destroy/release functions.
- [x] Validate descriptor before Provider registration.
- [x] Add malformed descriptor tests.

## 7. Handshake

- [x] Define loading handshake sequence.
- [x] Load library.
- [x] Resolve factory symbol.
- [x] Retrieve descriptor.
- [x] Validate descriptor.
- [x] Validate ABI version.
- [x] Retrieve metadata.
- [x] Validate metadata.
- [x] Retrieve Capability advertisements.
- [x] Validate advertisements.
- [x] Retrieve Device metadata.
- [x] Validate Devices.
- [x] Retrieve initial status.
- [x] Register Provider only after successful handshake.
- [x] Add handshake success and failure tests.

## 8. Metadata

- [x] Define Provider metadata ABI shape.
- [x] Include ProviderId.
- [x] Include Provider name.
- [x] Include Provider version.
- [x] Include vendor.
- [x] Include description.
- [x] Include Runtime compatibility.
- [x] Include feature flags.
- [x] Include loading mode.
- [x] Validate metadata strings.
- [x] Validate ProviderId uniqueness.
- [x] Add metadata tests.

## 9. Capability Advertisement ABI

- [x] Define Capability advertisement ABI shape.
- [x] Include Capability identifiers.
- [x] Include Capability versions.
- [x] Include Compute support where implemented.
- [x] Include operation-family support where implemented.
- [x] Include data movement support where implemented.
- [x] Validate advertisements before registration.
- [x] Reject malformed advertisements.
- [x] Add advertisement tests.

## 10. Device Metadata ABI

- [x] Define Device metadata ABI shape.
- [x] Include DeviceId.
- [x] Include Device type.
- [x] Include Provider ownership.
- [x] Include memory metadata where available.
- [x] Include feature metadata where available.
- [x] Avoid raw native handle exposure.
- [x] Validate Device identity uniqueness per Provider.
- [x] Add Device metadata tests.

## 11. Status Reporting ABI

- [x] Define status reporting ABI shape.
- [x] Include lifecycle.
- [x] Include health.
- [x] Include readiness.
- [x] Include pressure.
- [x] Include admission.
- [x] Include freshness or timestamp.
- [x] Include Device status.
- [x] Include Capability status.
- [x] Include diagnostic reason.
- [x] Avoid Provider-private layout dependence.
- [x] Add status ABI tests.

## 12. Execution ABI

- [x] Define execution entrypoint or function table shape.
- [x] Keep request and response ABI-compatible.
- [x] Avoid arbitrary Rust types in ABI payloads.
- [x] Preserve ProviderExecutionApi semantics.
- [x] Preserve Resource Affinity semantics.
- [x] Preserve Provider-owned resource semantics.
- [x] Preserve Device-bound resource semantics.
- [x] Preserve cancellation semantics.
- [x] Preserve structured error mapping.
- [x] Add execution ABI tests or fixtures.

## 13. Memory Ownership

- [x] Define string ownership rules.
- [x] Define list ownership rules.
- [x] Define descriptor ownership rules.
- [x] Define error message ownership rules.
- [x] Define opaque handle ownership rules.
- [x] Define release functions for Provider-allocated memory.
- [x] Define whether Runtime-owned buffers may be retained.
- [x] Define whether Provider-owned buffers may be retained.
- [x] Prevent cross-allocator freeing.
- [x] Add ownership tests.

## 14. Opaque Handles

- [x] Define Provider instance handle.
- [x] Define Provider-owned resource handle.
- [x] Define operation handle where required.
- [x] Define destroy/release for each handle.
- [x] Prevent WIT exposure of ABI handles.
- [x] Prevent Component exposure of ABI handles.
- [x] Prevent serialization of ABI handles as stable public identifiers.
- [x] Add opaque handle lifecycle tests.

## 15. Panic and Unwind Safety

- [x] Define no-unwind-across-ABI rule.
- [x] Require Provider adapter to catch panics or abort according to policy.
- [x] Treat unwind violation as Provider failure.
- [x] Add tests or documentation for panic behavior.
- [x] Ensure Runtime does not rely on unwinding through foreign code.

## 16. Error Model

- [x] Define invalid ABI descriptor error.
- [x] Define unsupported ABI version error.
- [x] Define invalid metadata error.
- [x] Define invalid advertisement error.
- [x] Define invalid Device metadata error.
- [x] Define initialization failure error.
- [x] Define Provider not ready error.
- [x] Define Provider draining error.
- [x] Define Provider saturated error.
- [x] Define execution rejected error.
- [x] Define execution failed error.
- [x] Define cancellation unsupported error.
- [x] Define resource invalid error.
- [x] Define panic/unwind violation error.
- [x] Normalize ABI errors into Runtime errors.

## 17. Thread Safety Declaration

- [x] Define Provider threading model declaration.
- [x] Support single-threaded Provider declaration.
- [x] Support Runtime-synchronized Provider declaration.
- [x] Support internally thread-safe Provider declaration.
- [x] Support reentrant Provider declaration where allowed.
- [x] Validate threading declaration.
- [x] Ensure Runtime respects declared threading model.
- [x] Add threading policy tests.

## 18. Blocking and Async Behavior

- [x] Define blocking execution declaration.
- [x] Define async-capable execution declaration where supported.
- [x] Define long-running operation behavior.
- [x] Ensure Runtime can isolate blocking Providers.
- [x] Ensure cancellation semantics are explicit.
- [x] Avoid universal async ABI unless intentionally designed.
- [x] Add blocking behavior tests.

## 19. Provider Lifecycle

- [x] Implement lifecycle states for dynamic loading.
- [x] Include discovered.
- [x] Include library-loaded.
- [x] Include descriptor-validated.
- [x] Include initialized.
- [x] Include registered.
- [x] Include ready.
- [x] Include draining.
- [x] Include stopped.
- [x] Include failed.
- [x] Include destroyed.
- [x] Add lifecycle transition tests.

## 20. Library Unloading

- [x] Define whether dynamic libraries are unloaded.
- [x] If unloaded, ensure no Provider resources remain.
- [x] If unloaded, ensure no in-flight operations remain.
- [x] If unloaded, ensure no callbacks remain.
- [x] If unloaded, ensure no background threads remain.
- [x] Permit conservative never-unload policy.
- [x] Add documentation and tests where feasible.

## 21. Loading Policy

- [x] Define allowed library path policy.
- [x] Define optional digest allowlist for native libraries.
- [x] Define optional signature metadata handling.
- [x] Define revoked Provider library handling.
- [x] Define development mode.
- [x] Ensure development mode is explicit.
- [x] Ensure ABI validation still runs in development mode.
- [x] Add loading policy tests.

## 22. Built-In Provider Adapter

- [x] Ensure built-in Providers can register through Provider Registry.
- [x] Allow built-in Providers to use Rust traits internally.
- [x] Do not expose built-in trait usage as dynamic ABI.
- [x] Align built-in Provider metadata with dynamic Provider metadata.
- [x] Align built-in Capability advertisements with dynamic Provider
      advertisements.
- [x] Align built-in status reporting with dynamic Provider status.

## 23. Test Provider Strategy

- [x] Keep simple in-process mock Providers for unit tests.
- [x] Add ABI-shaped test fixture for dynamic loading behavior.
- [x] Ensure tests using Rust trait objects do not claim ABI stability.
- [x] Add tests for invalid ABI descriptor.
- [x] Add tests for unsupported ABI version.
- [x] Add tests for invalid metadata.
- [x] Add tests for descriptor release behavior.

## 24. Public API Audit

- [x] Verify public Runtime APIs do not expose dynamic ABI internals.
- [x] Verify Component APIs do not expose Provider ABI handles.
- [x] Verify Compute WIT does not expose Provider ABI handles.
- [x] Verify Provider ABI descriptors remain native Runtime internals.
- [x] Verify Rust trait object APIs are documented as in-process only where they
      remain.

## 25. Observability

- [x] Emit library discovery observations.
- [x] Emit library load attempt observations.
- [x] Emit factory symbol resolution observations.
- [x] Emit ABI version rejection observations.
- [x] Emit descriptor validation observations.
- [x] Emit metadata validation observations.
- [x] Emit Provider initialization observations.
- [x] Emit Provider registration observations.
- [x] Emit Provider ready observations.
- [x] Emit Provider loading failure observations.
- [x] Emit Provider destruction observations.
- [x] Redact paths and sensitive metadata according to policy.

## 26. Documentation

- [x] Document Provider loading modes.
- [x] Document dynamic ABI policy.
- [x] Document why Rust trait objects are not stable ABI.
- [x] Document factory symbol policy.
- [x] Document ABI version negotiation.
- [x] Document memory ownership.
- [x] Document panic/unwind safety.
- [x] Document threading model.
- [x] Document built-in Provider behavior.
- [x] Document development mode.
- [x] Document security assumptions.

## 27. Security Review

- [x] Verify dynamic Provider loading is policy-gated.
- [x] Verify arbitrary library paths are not loaded by default.
- [x] Verify development mode is explicit.
- [x] Verify ABI descriptor validation happens before registration.
- [x] Verify Provider metadata cannot self-trust.
- [x] Verify unsupported ABI version rejects.
- [x] Verify panic/unwind cannot cross ABI boundary as normal behavior.
- [x] Verify library unloading is safe or disabled.
- [x] Verify native Provider loading is documented as trusted code execution.

## 28. Final Validation

- [x] Run formatting.
- [x] Run compilation checks.
- [x] Run Clippy.
- [x] Run complete tests.
- [x] Run dynamic loading tests.
- [x] Run Provider status tests.
- [x] Run Provider Registry tests.
- [x] Run OpenSpec validation.
- [x] Run coverage validation.
- [x] Verify no stable dynamic ABI uses `Box<dyn Provider>`.
- [x] Verify ABI version negotiation is defined.
- [x] Verify memory ownership rules are defined.
- [x] Verify Provider loading is policy-gated.
