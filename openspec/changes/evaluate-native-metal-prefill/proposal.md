## Why

The 29 October 2026 inference optimization roadmap identifies execution gaps beyond Magnetar v1.0.0. Existing contracts provide the foundation, but do not establish production performance for this scope. This change covers G03 and belongs at the following boundary: Experimental Metal Provider on Apple Silicon. It extends local AI execution or its explicitly separated tooling; cluster orchestration remains outside Magnetar Core.

## What Changes

- `G03` (P2, effort XL): Conditional native Metal specialization.

Require actual macOS hardware and a qualified WGPU comparison before adding native Metal kernels. Investigate supported simdgroup/MPS paths rather than assuming AMX access; preserve platform-safe dynamic availability.

Status: **Conditional experiment / interface proposal; no production support commitment.** All tasks are future work; authoring or validating this proposal does not implement an optimization. Indicative waves, dependencies, priorities, acceptance profiles and unresolved decisions are recorded in the [roadmap coverage index](../../../docs/roadmaps/inference-optimizations-2026-2028.md).

## Capabilities

### New Capabilities

- `native-metal-experiment`: an implementation/qualification profile for this work, extending the existing capabilities listed below rather than replacing their baseline contracts.

### Modified Capabilities

- `metal-provider`: replace "Metal Provider Is Unconditionally Unavailable" with "Metal Provider Availability Requires Hardware Qualification" for the explicitly qualified profile; retain the baseline outside that scope.

- `metal-provider`: replace "Metal Provider Documents Its Real Future Scope" with "Metal Provider Documents Its Measured Scope" for the explicitly qualified profile; retain the baseline outside that scope.

## Impact

Owning layer: Experimental Metal Provider on Apple Silicon. Implementation may span Magnetar and the appropriate pinned Component, Provider, Format or Loader repositories; submodule revisions are updated only after their own validation. No runtime code, dependencies, WIT or canonical specifications are changed by this proposal.

Existing contracts to reuse:
- [`metal-provider`](../../specs/metal-provider/spec.md)
- [`wgpu-provider`](../../specs/wgpu-provider/spec.md)
- [`provider-roadmap`](../../specs/provider-roadmap/spec.md)

## Dependencies

- `G01`: [`integrate-wgpu-production-compute`](../integrate-wgpu-production-compute/proposal.md).

Dependencies are exit-gate prerequisites, not a ban on independent design or hardware experiments.

## Compatibility and authority

Keep the existing F32/single-Device first-native profile valid. Components import only authorized portable graph/model/compute/state Capabilities and export model graph or scoped result Capabilities where this scope needs them; tooling-only changes add no Component imports/exports. They never choose Providers or Devices or receive native handles. Provider implementations expose real Devices and only qualified compute/movement profiles; native kernels, allocation, streams, events and compiler objects remain Provider-owned. Runtime derives ResourceAffinity and resolves eligible candidates before ranking.

No WIT change is assumed by this draft. Prefer existing contracts. If new portable operators, state or result schemas are necessary, first design a versioned additive WIT extension, validate old/new linking and reject unsupported profiles explicitly; do not alter existing signatures in place. Stateful work uses versioned leases, committed token boundaries and completion-safe retirement, and revalidates trust, affinity and capacity after failure. No optimization creates implicit movement or grants filesystem, network, process, source-control, secret or service authority. Any external adapter declares explicit allowed sources, destinations and identities outside Runtime Core.

## Non-goals

- No claim of delivered performance or production support from contracts, fixtures or Device discovery alone.
- No vendor code in Runtime Core, model-family Provider or native types in WIT.
- No automatic checkpoint downloads, cluster management or mandatory Tachyon dependency.
- No work outside G03 except the explicitly named qualification and compatibility prerequisites.
