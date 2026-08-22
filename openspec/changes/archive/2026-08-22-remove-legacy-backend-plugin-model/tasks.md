# Tasks

## 1. Legacy Backend Inventory

- [x] Search production Rust code for the `Backend` abstraction.
- [x] Search tests for Backend fixtures and mocks.
- [x] Search Runtime configuration for Backend-specific fields.
- [x] Search RuntimeBuilder for Backend registration.
- [x] Search Runtime registries for Backend storage.
- [x] Search execution contexts for Backend identity.
- [x] Search documentation for active Backend terminology.
- [x] Search canonical OpenSpec specifications for active Backend requirements.
- [x] Distinguish historical archived references from current architecture.

## 2. Remove Backend Trait

- [x] Remove the `Backend` trait.
- [x] Remove Backend-specific implementations.
- [x] Remove Backend-only metadata if any remains.
- [x] Remove Backend-specific test fixtures.
- [x] Ensure Device remains owned and exposed through Provider semantics.
- [x] Ensure no replacement `Backend` alias is introduced.

## 3. Remove Backend Registry State

- [x] Remove Backend storage from the Runtime registry.
- [x] Remove Backend maps and collections.
- [x] Remove Backend registration paths.
- [x] Remove Backend lookup paths.
- [x] Remove Backend enumeration paths.
- [x] Remove backend-name listing APIs.
- [x] Ensure ProviderRegistry stores only Provider-related native extension
      state.

## 4. Remove Backend Registration API

- [x] Remove `register_backend` or equivalent Runtime APIs.
- [x] Remove `backend` lookup APIs.
- [x] Remove `backend_names` APIs.
- [x] Remove Backend registration from RuntimeBuilder.
- [x] Remove Backend registration from initialization paths.
- [x] Update callers to Provider registration where semantically appropriate.

## 5. Runtime Configuration

- [x] Remove `preferred_backend` or equivalent Backend configuration.
- [x] Remove Backend configuration parsing if present.
- [x] Remove Backend defaults.
- [x] Do not replace `preferred_backend` with a direct `preferred_provider`
      selector.
- [x] Route Provider preference through Resolution Policy where required.
- [x] Preserve Runtime initialization without any Provider.

## 6. RuntimeBuilder

- [x] Remove the Backend collection from RuntimeBuilder.
- [x] Remove Backend builder methods.
- [x] Preserve Provider builder support.
- [x] Ensure RuntimeBuilder can construct an empty Runtime.
- [x] Ensure RuntimeBuilder can register one or more Providers.
- [x] Ensure builder behavior does not bypass Capability resolution.

## 7. Execution Context

- [x] Remove `backend_name` or equivalent legacy Backend identity from execution
      contexts.
- [x] Review execution-context constructors for Backend parameters.
- [x] Review execution-context accessors for Backend identity.
- [x] Use ProviderBinding where Provider identity is actually required.
- [x] Use DeviceBinding where Device identity is actually required.
- [x] Use Resource Affinity for stateful ownership constraints.
- [x] Avoid adding Provider identity where the execution context does not need
      it.

## 8. Device Ownership

- [x] Ensure every Device exposed for execution belongs to a Provider.
- [x] Preserve Provider identity in Device metadata where required.
- [x] Ensure Devices are enumerated through Provider semantics.
- [x] Remove any Device discovery route owned exclusively by Backend.
- [x] Ensure Device resolution remains a Runtime responsibility.

## 9. Provider Registry

- [x] Make ProviderRegistry the sole native extension registry.
- [x] Preserve Provider capability advertisements.
- [x] Preserve Provider compute advertisements.
- [x] Preserve Device enumeration.
- [x] Preserve Provider health integration.
- [x] Preserve Capability Registry integration.
- [x] Preserve Resource Affinity validation.
- [x] Preserve Resolution Policy candidate evaluation.

## 10. Provider Resolution

- [x] Verify Capability requests resolve only through Providers.
- [x] Verify Provider selection continues to use Resolution Policy.
- [x] Verify Provider advertisements remain part of compatibility evaluation.
- [x] Verify Device selection remains Runtime-owned.
- [x] Verify Resource Affinity still constrains Provider selection.
- [x] Verify Resource Affinity still constrains Device selection.
- [x] Verify no removed Backend path can bypass resolution.

## 11. Provider Loading Terminology

- [x] Review dynamic native-library loading code.
- [x] Rename Backend-oriented loader names to Provider terminology where present.
- [x] Rename Backend-oriented loader errors to Provider terminology where
      present.
- [x] Preserve `magnetar_provider_create` or the current Provider factory
      contract where already Provider-oriented.
- [x] Document that native Provider loading is trusted.
- [x] Do not stabilize the Provider binary ABI in this change.

## 12. Plugin Specification Removal

- [x] Remove `Plugin Discovery` from the canonical Plugin specification.
- [x] Remove `Plugin Initialization` from the canonical Plugin specification.
- [x] Remove `Plugin Version Compatibility` from the canonical Plugin
      specification.
- [x] Remove `Plugin Metadata` from the canonical Plugin specification.
- [x] Remove `General Plugin Interface` from the canonical Plugin specification.
- [x] Remove `Extensible Plugin Registry` from the canonical Plugin
      specification.
- [x] Remove `Plugin Lifecycle` from the canonical Plugin specification.
- [x] Remove the empty canonical `openspec/specs/plugin/` domain after the
      change is archived if OpenSpec tooling permits it.
- [x] Preserve archived Plugin-related changes unchanged.

## 13. Plugin Implementation Cleanup

- [x] Search Rust production code for `Plugin`.
- [x] Remove generic Plugin interfaces if still present.
- [x] Remove generic Plugin Registry implementations if still present.
- [x] Remove Plugin lifecycle code not already represented by Provider or
      Component semantics.
- [x] Remove Plugin compatibility/version code not used by Providers or
      Components.
- [x] Do not remove legitimate historical comments from archives.

## 14. Responsibility Migration

- [x] Classify native hardware extensions as Providers.
- [x] Classify native kernel execution as Provider responsibility.
- [x] Classify WASM observability extensions as Components.
- [x] Classify future portable model extensions as Components where applicable.
- [x] Classify future portable tool extensions as Components where applicable.
- [x] Avoid creating generic extension categories without a dedicated
      architectural change.

## 15. Component Model Preservation

- [x] Preserve Component types introduced by the Component Model.
- [x] Preserve Component metadata.
- [x] Preserve Component lifecycle prototypes.
- [x] Preserve WIT import/export metadata.
- [x] Do not merge Component and Provider registries.
- [x] Do not make Components trusted native extensions.
- [x] Leave real WASM engine implementation to the dedicated Component Runtime
      changes.

## 16. Provider Model Preservation

- [x] Preserve ProviderMetadata.
- [x] Preserve ProviderDescriptor.
- [x] Preserve Provider Capability advertisement.
- [x] Preserve Device ownership through Provider metadata.
- [x] Preserve ProviderExecutionApi.
- [x] Preserve Provider health reporting.
- [x] Preserve Provider compute advertisement behavior.

## 17. Error Model

- [x] Remove Backend-specific Runtime errors.
- [x] Remove Plugin-specific Runtime errors that no longer apply.
- [x] Map relevant failures to Provider errors.
- [x] Map portable Component failures to Component errors.
- [x] Preserve structured Compute errors.
- [x] Avoid exposing native Provider implementation errors directly to
      Components.

## 18. Observability

- [x] Rename active Runtime observation labels that describe the legacy Backend
      abstraction where appropriate.
- [x] Preserve provider-native diagnostic text where `backend` is merely
      descriptive implementation terminology.
- [x] Do not treat a diagnostic field name alone as an architectural Backend.
- [x] Ensure Provider identity remains observable through Provider bindings.
- [x] Ensure removal of Backend does not break observability correlation.

## 19. Documentation

- [x] Update README after Backend implementation removal.
- [x] Update `docs/architecture/overview.md` if required.
- [x] Update current architecture documents containing active Backend diagrams.
- [x] Update current architecture documents containing active Plugin diagrams.
- [x] Preserve archived OpenSpec documents unchanged.
- [x] Distinguish descriptive low-level "backend" wording from the removed
      architectural `Backend` type where necessary.

## 20. Public Rust API Migration

- [x] Identify every public Rust item removed by this change.
- [x] Document its canonical replacement where one exists.
- [x] Document that Backend registration becomes Provider registration.
- [x] Document that Backend preference becomes Resolution Policy.
- [x] Document that Plugin native extensions become Providers.
- [x] Document that portable Plugin extensions become Components.
- [x] Do not preserve deprecated aliases solely for source compatibility.

## 21. Unit Tests

- [x] Remove obsolete Backend tests.
- [x] Remove obsolete Plugin tests.
- [x] Add test that Runtime initializes with zero Providers.
- [x] Add test that a Provider registers successfully.
- [x] Add test that Provider Devices are discoverable.
- [x] Add test that Capability resolution uses Providers.
- [x] Add test that multiple Providers are evaluated by Resolution Policy.
- [x] Add test that Resource Affinity rejects incompatible Provider switching.
- [x] Add test that Resource Affinity rejects incompatible Device switching.

## 22. Regression Tests

- [x] Verify Runtime initialization behavior remains correct.
- [x] Verify Provider isolation remains correct.
- [x] Verify Provider fallback before state creation remains correct.
- [x] Verify Provider-pinned state does not migrate implicitly.
- [x] Verify Scheduler still consumes resolved execution plans.
- [x] Verify ProviderExecutionApi still receives Provider-bound execution.
- [x] Verify observability still reports Provider and Device identities.

## 23. Static Architecture Validation

- [x] Add a repository check that production code no longer defines
      `trait Backend`.
- [x] Add a repository check that production code no longer exposes
      `register_backend`.
- [x] Add a repository check that production code no longer contains
      `preferred_backend`.
- [x] Add a repository check that production code no longer exposes a generic
      `Plugin` interface.
- [x] Exclude OpenSpec archives from these historical-term checks.
- [x] Allow descriptive native implementation wording only where it is not an
      architectural type or API.

## 24. OpenSpec Validation

- [x] Validate the runtime delta specification.
- [x] Validate the provider delta specification.
- [x] Validate removal of every current Plugin requirement by exact name.
- [x] Verify canonical Plugin requirements are absent after archive.
- [x] Verify historical Plugin changes remain in `openspec/changes/archive/`.

## 25. Final Validation

- [x] Run formatting.
- [x] Run compilation checks.
- [x] Run Clippy.
- [x] Run the complete test suite.
- [x] Run WIT validation.
- [x] Run OpenSpec validation.
- [x] Run coverage validation.
- [x] Verify no production execution path depends on Backend.
- [x] Verify no production extension path depends on Plugin.
- [x] Verify Provider is the sole native extension mechanism.
- [x] Verify Component remains the sole portable WASM extension mechanism.