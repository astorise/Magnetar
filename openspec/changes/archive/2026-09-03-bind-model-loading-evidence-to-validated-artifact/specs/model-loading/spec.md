## ADDED Requirements

### Requirement: Loaded Model Context Is Runtime-Issued

A `LoadedModelContext` SHALL be producible only as the result of a successful `ModelLoadingCoordinator::load()` call.

The `ModelLoadingResidencyPlan` it carries is subject to the same
constraint. An external caller SHALL NOT be able to construct a
`LoadedModelContext` or `ModelLoadingResidencyPlan` directly, whether
claiming a real or fabricated `ModelArtifactId`, `ModelLoadingState`, or
residency plan.

#### Scenario: A caller cannot fabricate a loaded context

Given a caller has not invoked `ModelLoadingCoordinator::load()`

When the caller attempts to obtain a `LoadedModelContext` claiming a given `ModelArtifactId` and a ready `ModelLoadingState`

Then no public constructor or field-literal path exists to produce one, and `Runtime::create_model_instance()` can only ever be called with a context Model Loading itself produced.

#### Scenario: A successfully loaded context is usable as before

Given `ModelLoadingCoordinator::load()` completes successfully for a validated, trusted Model Artifact

When the returned `LoadedModelContext` is passed to `Runtime::create_model_instance()`

Then instance creation proceeds exactly as it did before this requirement, with no change to the public signature of `create_model_instance` or `ModelInstanceDefinition::from_loaded_context`.
