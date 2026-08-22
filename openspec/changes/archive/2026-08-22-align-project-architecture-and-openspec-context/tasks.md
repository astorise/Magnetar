# Tasks

## 1. Canonical Architecture

- [x] Define the canonical high-level Magnetar architecture.
- [x] Document Runtime as the global local-node orchestration authority.
- [x] Document Component as portable WASM application/runtime extension code.
- [x] Document Capability as a portable WIT contract.
- [x] Document Provider as a native implementation of one or more Capabilities.
- [x] Document Device as a Provider-owned physical or logical execution target.
- [x] Document Resource Affinity.
- [x] Document Resolution Policy.
- [x] Document the canonical resolution flow.

## 2. Canonical Execution Flow

- [x] Document `Component -> Capability -> Runtime -> Provider -> Device`.
- [x] Document that Components request Capabilities rather than Providers.
- [x] Document that Components SHALL NOT directly select CUDA, Metal, CPU, ROCm,
      OpenVINO, QNN, or another hardware implementation.
- [x] Document that Components SHALL NOT directly select a Provider.
- [x] Document that Components SHALL NOT directly select a Device.
- [x] Document that Runtime Resolution Policy selects compatible execution
      targets.
- [x] Document that Resource Affinity can restrict the set of valid targets.

## 3. Terminology

- [x] Define canonical terminology for Runtime.
- [x] Define canonical terminology for Component.
- [x] Define canonical terminology for Capability.
- [x] Define canonical terminology for Provider.
- [x] Define canonical terminology for Device.
- [x] Define canonical terminology for Resource Affinity.
- [x] Define canonical terminology for Resolution Policy.
- [x] Define canonical terminology for Artifact.
- [x] Define canonical terminology for Model.
- [x] Define canonical terminology for Agent.
- [x] Define canonical terminology for Tool.

## 4. Deprecated Terminology

- [x] Mark `Backend` as deprecated as a primary Magnetar architectural concept.
- [x] Mark `Plugin` as deprecated in favor of Provider or Component depending on
      the actual role.
- [x] Mark `Host` as non-canonical as a primary architectural entity.
- [x] Document migration terminology for historical specifications.
- [x] Do not rewrite archived OpenSpec history merely to remove historical
      terminology.
- [x] Prevent new specifications from introducing these terms as primary
      concepts without explicit architectural justification.

## 5. Component Boundary

- [x] Document Components as WASM Components.
- [x] Document that Components consume portable WIT contracts.
- [x] Document that Components may expose higher-level behavior.
- [x] Document that Components do not receive native Runtime handles.
- [x] Document that Components do not receive Provider-native handles.
- [x] Document that Components do not receive Device-native handles.
- [x] Document that Components do not receive raw pointers, queues, streams,
      backend storage, or kernel objects.
- [x] Document coarse-grained Component-to-Runtime calls.

## 6. Provider Boundary

- [x] Document Providers as native trusted Runtime extensions.
- [x] Document that Providers implement Capabilities.
- [x] Document that Providers expose Devices.
- [x] Document that Providers own native execution details.
- [x] Document that Providers may own kernels, allocators, queues, streams,
      native contexts, and device APIs.
- [x] Document that Provider internals remain invisible to Components.
- [x] Document that Providers do not perform global Provider resolution.

## 7. Provider Versus Component

- [x] Add an explicit Provider versus Component comparison.
- [x] Document that WASM integrations are Components, not Providers.
- [x] Document that native CUDA/CPU/Metal/OpenVINO/QNN implementations are
      Providers.
- [x] Document that OpenTelemetry/Prometheus/Jaeger integrations can be
      Components.
- [x] Document that future Model, Agent, and Tool implementations may use
      Components where the boundary is portable.
- [x] Prevent portable Components from being treated as trusted native
      extensions.

## 8. AI Runtime Scope

- [x] Document Magnetar as a standalone AI Runtime.
- [x] Document that AI execution belongs to Magnetar.
- [x] Document future model loading responsibility.
- [x] Document future model residency responsibility.
- [x] Document future tokenization responsibility.
- [x] Document future prompt-template responsibility.
- [x] Document future generation responsibility.
- [x] Document future streaming responsibility.
- [x] Document future continuous batching responsibility.
- [x] Document future KV cache responsibility.
- [x] Document future prefix-cache responsibility.
- [x] Document future adapter/LoRA responsibility.
- [x] Document future quantization responsibility.
- [x] Document future multi-device execution responsibility.
- [x] Document future agent runtime responsibility.
- [x] Document future tool execution responsibility.
- [x] Distinguish future scope from already implemented functionality.

## 9. Magnetar and Tachyon Boundary

- [x] Document the canonical Magnetar/Tachyon responsibility split.
- [x] Define Magnetar as owner of local AI execution.
- [x] Define Tachyon as owner of distributed service orchestration.
- [x] Define Tachyon as owner of inter-node discovery and routing.
- [x] Define Tachyon as owner of cluster-level deployment and GitOps.
- [x] Define Tachyon as a possible distributor of Magnetar Components.
- [x] Define Tachyon as a possible distributor of Model artifacts.
- [x] Prevent Magnetar from requiring Tachyon for standalone execution.
- [x] Prevent Tachyon from owning model-specific inference implementation after
      migration.
- [x] Document local versus cluster-level scheduling boundaries.

## 10. Scheduling Boundary

- [x] Document generic local Runtime scheduling as a Magnetar responsibility.
- [x] Document future inference scheduling as a Magnetar responsibility.
- [x] Document future continuous batching as a Magnetar responsibility.
- [x] Document Tachyon cluster routing as distinct from Magnetar scheduling.
- [x] Prevent duplicate intra-node inference scheduling between Magnetar and
      Tachyon.

## 11. Component Distribution Boundary

- [x] Document that Components may originate from external sources.
- [x] Document that Tachyon may be one such source.
- [x] Keep the distribution contract vendor-neutral.
- [x] Preserve the dependency direction `Tachyon -> Magnetar`.
- [x] Prevent architectural dependency `Magnetar -> Tachyon`.
- [x] Document that Magnetar validates Components before execution.
- [x] Document that Magnetar controls Capability linking and authority.

## 12. Artifact Terminology

- [x] Define Component Artifact as executable WASM Component code.
- [x] Define Model Artifact as weights and associated model data.
- [x] Prevent Model Artifact and Component Artifact from being conflated.
- [x] Document that a future model instance may combine Component code, Model
      Artifact, Provider, Device, and Runtime resources.
- [x] Reserve room for future artifact digest and trust metadata.

## 13. Model Architecture Boundary

- [x] Document that model architecture is not a Provider.
- [x] Explicitly reject `LlamaProvider`, `QwenProvider`, or equivalent naming
      when the object represents model architecture rather than hardware
      execution.
- [x] Document that future model architecture logic may be implemented as
      Components or Runtime modules depending on portability and performance
      requirements.
- [x] Preserve coarse execution boundaries for model Components.

## 14. magnetar-cli

- [x] Document `magnetar-cli` as a future first-party client of Magnetar.
- [x] Document that `magnetar-cli` SHALL use the same Runtime services as other
      Magnetar clients.
- [x] Prevent inference logic from being duplicated inside the CLI.
- [x] Document local model inference as a future CLI use case.
- [x] Document interactive chat as a future CLI use case.
- [x] Document coding-agent functionality as a future CLI use case.
- [x] Document Provider, Device, Model, Component, and observability inspection
      as future CLI capabilities.

## 15. Service API

- [x] Document a future Magnetar service/API mode.
- [x] Document that embedded, CLI, and service usage SHALL share the same Runtime
      semantics.
- [x] Document that Tachyon may consume Magnetar through an appropriate
      integration boundary.
- [x] Avoid defining the concrete transport in this change.

## 16. README

- [x] Rewrite the README high-level architecture.
- [x] Replace Backend-centric diagrams with Provider/Device terminology.
- [x] Remove Backend selection as a canonical Runtime behavior.
- [x] Document Component and Capability roles.
- [x] Document Resource Affinity and Resolution Policy.
- [x] Document Magnetar's standalone AI Runtime direction.
- [x] Document the Magnetar/Tachyon boundary.
- [x] Clearly distinguish implemented features from roadmap features.
- [x] Fix malformed diagram/tree character encoding where present.
- [x] Avoid advertising crates that do not yet exist as implemented workspace
      members.
- [x] Present planned crates explicitly as roadmap items if retained.

## 17. Canonical Architecture Document

- [x] Add `docs/architecture/overview.md`.
- [x] Make the architecture overview the canonical conceptual entry point.
- [x] Link the overview from README.
- [x] Link existing detailed architecture documents from the overview.
- [x] Include the canonical architecture diagram.
- [x] Include the responsibility matrix.
- [x] Include the terminology table.
- [x] Include the Magnetar/Tachyon boundary.
- [x] Include Component/Provider comparison.
- [x] Include implemented versus planned architecture markers.

## 18. OpenSpec Project Context

- [x] Replace the template-only context in `openspec/config.yaml`.
- [x] Describe the Rust and WebAssembly Component Model technology direction.
- [x] Define canonical project vocabulary.
- [x] Define architectural invariants.
- [x] Define the Magnetar/Tachyon boundary.
- [x] Define Component/Capability/Provider/Device relationships.
- [x] Define Resource Affinity rules.
- [x] Define native-handle isolation rules.
- [x] Define coarse WIT boundary rules.
- [x] Define standalone Magnetar requirements.
- [x] Define future AI Runtime scope.

## 19. OpenSpec Proposal Rules

- [x] Require proposals to state why a responsibility belongs in Magnetar.
- [x] Require proposals involving execution to identify the owning architectural
      layer.
- [x] Require proposals introducing Components to identify imported and exported
      Capabilities.
- [x] Require proposals introducing Providers to identify exposed Devices and
      implemented Capabilities.
- [x] Require proposals involving resources to state Resource Affinity behavior.
- [x] Require proposals involving failure to state recovery semantics.
- [x] Require proposals involving external access to state authority/scoping
      requirements.
- [x] Require proposals to identify compatibility impact on existing WIT
      contracts.

## 20. OpenSpec Specification Rules

- [x] Require normative SHALL/SHOULD/MAY language where applicable.
- [x] Require scenarios for architectural invariants.
- [x] Require explicit non-goals for ambiguous cross-layer responsibilities.
- [x] Prevent specs from treating archived historical terminology as canonical.
- [x] Require portable Component contracts to avoid native implementation types.

## 21. OpenSpec Task Rules

- [x] Require implementation tasks.
- [x] Require test tasks.
- [x] Require documentation tasks when public architecture changes.
- [x] Require WIT validation tasks when WIT changes.
- [x] Require cross-platform consideration for native Runtime changes.
- [x] Require migration tasks when an existing public contract changes.

## 22. Architecture Stability Classification

- [x] Mark Runtime/Component/Capability/Provider/Device as canonical concepts.
- [x] Mark Resource Affinity and Resolution Policy as canonical concepts.
- [x] Mark exact future model/generation/agent WIT contracts as not yet stable.
- [x] Mark concrete Component distribution protocol as future work.
- [x] Mark concrete Provider ABI stabilization as future work.
- [x] Mark concrete Magnetar service transport as future work.
- [x] Prevent roadmap ideas from being documented as already stable contracts.

## 23. Documentation Consistency

- [x] Review active OpenSpec specifications for conflicting current terminology.
- [x] Review architecture documentation for conflicting current terminology.
- [x] Do not rewrite archived OpenSpec history.
- [x] Add forward-looking notes where historical documents remain useful but use
      obsolete naming.
- [x] Ensure new canonical documentation takes precedence over historical
      artifacts.

## 24. Validation

- [x] Verify README uses canonical terminology.
- [x] Verify architecture overview uses canonical terminology.
- [x] Verify `openspec/config.yaml` contains Magnetar-specific context.
- [x] Verify new OpenSpec proposals receive sufficient architecture context.
- [x] Verify no new canonical diagram presents Backend as a primary entity.
- [x] Verify no new canonical diagram presents Plugin as a primary entity.
- [x] Verify Tachyon is an optional external integration rather than a Magnetar
      dependency.
- [x] Verify future AI functionality is clearly distinguished from currently
      implemented functionality.
