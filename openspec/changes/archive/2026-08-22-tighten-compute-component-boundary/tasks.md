# Tasks

## 1. Current Compute Boundary Inventory

- [x] Inventory every `magnetar:compute/run@1.1.0` WIT type.
- [x] Inventory every WIT field containing Provider identity.
- [x] Inventory every WIT field containing Device identity.
- [x] Inventory every WIT field containing AffinityGroup identity.
- [x] Inventory every Rust conversion for `data-movement-descriptor`.
- [x] Inventory every test constructing `target-provider`.
- [x] Inventory every test constructing `target-device`.
- [x] Inventory every test constructing `target-affinity-group`.
- [x] Inventory every Provider advertisement referencing Compute version 1.1.0.
- [x] Inventory every architecture document referencing Compute version 1.1.0.
- [x] Distinguish Component input values from Runtime-produced diagnostics.

## 2. Compute Capability Version

- [x] Introduce `magnetar:compute@2.0.0`.
- [x] Define `magnetar:compute/run@2.0.0`.
- [x] Update the canonical Capability specification.
- [x] Update Provider advertisement fixtures to advertise `2.0.0`.
- [x] Update Runtime resolution fixtures to request `2.0.0`.
- [x] Do not treat `1.1.0` as automatically compatible with `2.0.0`.
- [x] Document the breaking migration.
- [x] Preserve archived OpenSpec references to earlier versions unchanged.

## 3. Remove Provider Target from WIT

- [x] Remove `target-provider` from `data-movement-descriptor`.
- [x] Remove conversions for `target-provider`.
- [x] Remove validation of Component-provided Provider names.
- [x] Remove tests that treat Provider name as portable placement input.
- [x] Verify no replacement portable Provider selector is introduced.

## 4. Remove Device Target from WIT

- [x] Remove `target-device` from `data-movement-descriptor`.
- [x] Remove conversions for `target-device`.
- [x] Remove validation of Component-provided Device names.
- [x] Remove tests that treat Device name as portable placement input.
- [x] Verify no replacement portable Device selector is introduced.

## 5. Remove Affinity Group Target from WIT

- [x] Remove `target-affinity-group` from `data-movement-descriptor`.
- [x] Remove WIT exposure of process-local `AffinityGroupId` for routing.
- [x] Remove Component construction of Runtime affinity groups.
- [x] Derive actual affinity from Runtime-owned source resources.
- [x] Verify Components cannot forge affinity by supplying numeric identifiers.

## 6. Portable Placement Intent

- [x] Define a portable `placement-intent`.
- [x] Add `preserve-source-affinity`.
- [x] Add `runtime-selected`.
- [x] Add `host-accessible`.
- [x] Document semantics of every placement intent.
- [x] Reject invalid intent/data-movement-kind combinations.
- [x] Keep placement intent independent from Provider names.
- [x] Keep placement intent independent from Device names.
- [x] Keep placement intent independent from vendor names.

## 7. Host Staging Policy

- [x] Replace `allow-host-staging: bool`.
- [x] Define `host-staging-policy`.
- [x] Add `forbid`.
- [x] Add `permit`.
- [x] Document that `permit` is semantic permission, not Runtime authorization.
- [x] Ensure Runtime policy can still prohibit host staging.
- [x] Ensure Provider advertisements can still report staging support.
- [x] Ensure Resource Affinity can still make staging invalid.
- [x] Do not add `force-host-staging` in this change.

## 8. Data Movement Descriptor

- [x] Define the v2 portable data-movement descriptor.
- [x] Keep `kind`.
- [x] Keep portable source information.
- [x] Keep portable output tensor description.
- [x] Add placement intent.
- [x] Add host staging policy.
- [x] Remove all concrete execution target identities.
- [x] Update Rust-side portable descriptor types.
- [x] Update WIT/Rust conversion tests.

## 9. Data Movement Kind Semantics

- [x] Define valid placement intent for `upload`.
- [x] Define valid placement intent for `download`.
- [x] Define valid placement intent for `copy`.
- [x] Define valid placement intent for `materialize`.
- [x] Define valid placement intent for `transfer`.
- [x] Define valid placement intent for `dtype-conversion`.
- [x] Define valid placement intent for `placement-conversion`.
- [x] Preserve explicit movement semantics.

## 10. Source Affinity

- [ ] Resolve source tensor identity to Runtime-owned resource state.
- [ ] Load Resource Affinity from the Runtime resource registry.
- [ ] Reject mismatched or stale resource identity.
- [ ] Preserve Provider-pinned affinity.
- [ ] Preserve Device-bound affinity.
- [ ] Preserve Capability bindings.
- [ ] Preserve Artifact bindings where applicable.
- [ ] Preserve affinity-group membership internally.
- [ ] Do not trust Component-supplied affinity metadata as authoritative.

## 11. Placement Resolution

- [ ] Introduce a Runtime-native placement resolution step.
- [ ] Resolve portable placement intent after validating source affinity.
- [ ] Apply mandatory Resource Affinity constraints first.
- [ ] Apply Capability compatibility.
- [ ] Apply Provider advertisements.
- [ ] Apply Resolution Policy.
- [ ] Apply Device availability.
- [ ] Apply memory-planning constraints.
- [ ] Produce a resolved native placement decision.

## 12. Resolved Data Movement Plan

- [ ] Define a Runtime-native resolved movement representation.
- [ ] Allow it to reference ProviderBinding.
- [ ] Allow it to reference DeviceBinding.
- [ ] Allow it to reference CapabilityBinding.
- [ ] Record whether transfer is required.
- [ ] Record whether materialization is required.
- [ ] Record whether host staging is selected.
- [ ] Record the resulting Resource Affinity.
- [ ] Keep this representation outside Component-facing WIT.

## 13. Preserve Affinity Intent

- [ ] Implement `preserve-source-affinity`.
- [ ] Reject plans that would violate authoritative Provider affinity.
- [ ] Reject plans that would violate authoritative Device affinity.
- [ ] Reject plans that would violate artifact affinity.
- [ ] Do not reinterpret preserve-affinity as permission to migrate.

## 14. Runtime-Selected Placement Intent

- [ ] Implement `runtime-selected`.
- [ ] Resolve destination only after mandatory constraints are known.
- [ ] Allow Resolution Policy to rank compatible candidates.
- [ ] Do not expose candidate Provider names back through the request API.
- [ ] Record the resulting binding internally.
- [ ] Return structured resolution failure when no compatible target exists.

## 15. Host-Accessible Intent

- [ ] Implement `host-accessible` as a semantic data accessibility property.
- [ ] Do not equate `host-accessible` with selecting a CPU Provider.
- [ ] Allow Provider-native mechanisms to satisfy host accessibility.
- [ ] Allow explicit download/staging where policy permits.
- [ ] Reject host staging when Component policy forbids it.
- [ ] Reject host staging when Runtime policy forbids it.

## 16. Explicit Transfer Semantics

- [ ] Preserve explicit cross-placement transfer.
- [ ] Preserve explicit copy.
- [ ] Preserve explicit materialization.
- [ ] Preserve explicit upload.
- [ ] Preserve explicit download.
- [ ] Preserve explicit placement conversion.
- [ ] Ensure removal of target IDs does not create implicit migration.
- [ ] Ensure execution planning inserts or requires explicit movement before an
      incompatible consumer executes.

## 17. No Automatic Migration

- [ ] Preserve Provider-pinned semantics.
- [ ] Preserve Device-bound semantics.
- [ ] Preserve affinity-group semantics internally.
- [ ] Do not migrate a live tensor merely because another Provider is preferred.
- [ ] Return structured affinity errors when movement is not valid.
- [ ] Keep replay/recovery semantics outside this change.

## 18. Compute Diagnostics

- [x] Preserve Runtime-produced Provider identity in diagnostics where useful.
- [x] Preserve Runtime-produced Device identity in diagnostics where useful.
- [x] Preserve rejected candidate diagnostics.
- [x] Document diagnostic identity as output-only metadata.
- [ ] Ensure diagnostic Provider/Device fields are never interpreted as future
      routing input.
- [x] Preserve redaction rules.

## 19. Observability

- [ ] Preserve Runtime observation correlation with ProviderId.
- [ ] Preserve Runtime observation correlation with DeviceId.
- [ ] Preserve ComputeExecutionPlanId correlation.
- [ ] Preserve Resource Affinity diagnostics.
- [ ] Distinguish observed resolution results from Component placement intent.
- [ ] Ensure observability cannot be used as an indirect routing control.

## 20. WIT Consumer Direction

- [x] Remove the canonical `world compute { export run; }` Component-facing
      interpretation.
- [x] Define a reference consumer world using `import run`, or document that
      Components import the interface from their own worlds.
- [x] Verify Model Components can import `magnetar:compute/run@2.0.0`.
- [x] Verify observability or tool Components can define independent worlds.
- [x] Do not define native Providers as WASM Component worlds.
- [x] Document Runtime as the implementation/linker side of the imported
      Capability.

## 21. WIT Contract Separation

- [x] Review Provider advertisement types currently declared in Compute WIT.
- [x] Ensure Provider advertisement data is not accidentally exposed as a
      Component routing API.
- [x] Keep Runtime/Provider-only binding types outside portable request fields.
- [x] Keep ProviderBinding outside portable WIT.
- [x] Keep DeviceBinding outside portable WIT.
- [x] Keep AffinityGroupId outside portable WIT.
- [x] Record larger advertisement-WIT separation as future work if required.

## 22. Provider-Specific Extension Isolation

- [x] Review provider-specific dtype values.
- [x] Review provider-opaque layout values.
- [x] Review provider-specific operation schemas.
- [x] Ensure these values cannot directly select a Provider.
- [ ] Require an explicitly non-portable extension context where they remain
      supported.
- [x] Preserve portable Compute behavior without requiring Provider-specific
      extension values.
- [x] Do not redesign the entire extension system in this change.

## 23. Compute Error Mapping

- [x] Preserve `no-compatible-provider`.
- [x] Preserve `policy-rejected-provider`.
- [x] Preserve `provider-unavailable`.
- [x] Preserve `device-unavailable`.
- [x] Preserve `incompatible-resource-affinity`.
- [x] Preserve `provider-pinned-resource`.
- [x] Preserve `device-bound-resource`.
- [x] Preserve `affinity-group-mismatch`.
- [x] Use these errors for Runtime-resolved placement failures.
- [x] Do not expose raw Provider-native errors as stable semantics.

## 24. Capability Resolution

- [x] Ensure Compute v2 requests enter Capability Resolution.
- [x] Ensure Provider selection remains Runtime-owned.
- [x] Ensure Device selection remains Runtime-owned.
- [x] Ensure Resource Affinity is applied before policy preferences.
- [x] Ensure Provider advertisements are checked.
- [x] Ensure incompatible Provider v1.1-only implementations are rejected for
      v2 requests.

## 25. Provider Advertisement Migration

- [x] Update Compute Providers to advertise `magnetar:compute/run@2.0.0`.
- [x] Update Compute advertisement fixtures.
- [x] Update capability-version validation tests.
- [x] Ensure v1.1-only Providers do not silently claim v2 support.
- [x] Preserve Provider-specific operation support metadata.

## 26. Scheduler and Execution Planning

- [ ] Update execution planning to consume placement intent rather than
      Component target bindings.
- [ ] Keep selected Provider/Device in ComputeExecutionPlan.
- [ ] Keep Scheduler independent from portable placement requests.
- [ ] Ensure Scheduler executes the validated resolved plan.
- [ ] Ensure Scheduler does not independently reinterpret Component intent.
- [ ] Preserve immutable historical execution plans after resolution.

## 27. Memory Planning

- [ ] Update memory planning to consume resolved placement.
- [ ] Preserve explicit materialization rules.
- [ ] Preserve explicit transfer rules.
- [ ] Preserve memory limit validation.
- [ ] Preserve Provider-specific memory constraints.
- [ ] Do not allow Component placement intent to bypass memory safety checks.

## 28. Public Rust API

- [x] Review public Rust data-movement types mirroring the old WIT record.
- [x] Remove public Component-facing concrete target Provider fields.
- [x] Remove public Component-facing concrete target Device fields.
- [x] Remove public Component-facing affinity-group target fields.
- [x] Introduce portable placement intent types.
- [x] Introduce explicit host-staging policy type.
- [x] Keep Runtime-native resolved bindings separate from portable request types.

## 29. Migration

- [x] Document `1.1.0 -> 2.0.0` migration.
- [x] Document removal of `target-provider`.
- [x] Document removal of `target-device`.
- [x] Document removal of `target-affinity-group`.
- [x] Document replacement of `allow-host-staging`.
- [x] Document placement-intent semantics.
- [x] Document WIT consumer-world direction.
- [x] Do not silently translate arbitrary old target Provider/Device strings.

## 30. Unit Tests

- [x] Test `preserve-source-affinity`.
- [x] Test `runtime-selected`.
- [x] Test `host-accessible`.
- [x] Test forbidden host staging.
- [x] Test permitted host staging.
- [x] Test Runtime policy overriding staging permission.
- [x] Test Provider-pinned source.
- [x] Test Device-bound source.
- [x] Test no compatible Provider.
- [ ] Test no compatible Device.
- [x] Test Provider advertisement mismatch.
- [x] Test v1.1/v2 version incompatibility.

## 31. WIT Contract Tests

- [x] Parse `magnetar:compute@2.0.0`.
- [x] Validate the `run` interface.
- [x] Validate the consumer world imports `run`.
- [ ] Compile a minimal fixture Component importing Compute v2.
- [x] Verify the fixture cannot provide a Provider target.
- [x] Verify the fixture cannot provide a Device target.
- [x] Verify the fixture cannot provide an AffinityGroupId.
- [x] Verify Provider/Device diagnostic outputs remain representable.

## 32. Architecture Regression Tests

- [x] Search Compute WIT request types for `target-provider`.
- [x] Search Compute WIT request types for `target-device`.
- [x] Search Compute WIT request types for `target-affinity-group`.
- [x] Fail architecture validation if these legacy routing fields return.
- [x] Allow Provider/Device identity in Runtime-generated diagnostics.
- [x] Allow Provider/Device identity in native Runtime planning structures.
- [x] Keep archived WIT/OpenSpec history outside the architecture check.

## 33. Documentation

- [x] Update Compute Capability documentation.
- [x] Update compute data-movement documentation.
- [x] Update Resource Affinity documentation.
- [x] Update execution-planning documentation.
- [x] Update WIT examples.
- [x] Add placement-intent examples.
- [x] Document output diagnostics versus input routing distinction.
- [x] Document the Compute v2 migration.

## 34. Final Validation

- [x] Run formatting.
- [x] Run compilation checks.
- [x] Run Clippy.
- [x] Run complete tests.
- [x] Run WIT validation.
- [x] Run OpenSpec validation.
- [ ] Run coverage validation.
- [x] Verify no portable Compute request contains Provider identity.
- [x] Verify no portable Compute request contains Device identity.
- [x] Verify no portable Compute request contains Runtime affinity-group
      identity.
- [x] Verify explicit data movement is preserved.
- [x] Verify Resource Affinity remains authoritative.
- [x] Verify Resolution Policy remains responsible for compatible target
      selection.
