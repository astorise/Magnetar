## Why

The 29 October 2026 inference optimization roadmap identifies execution gaps beyond Magnetar v1.0.0. Existing contracts provide the foundation, but do not establish production performance for this scope. This change covers G08, G09, G10 and belongs at the following boundary: Optional Tachyon-side integration and transport adapter; Magnetar local execution. It extends local AI execution or its explicitly separated tooling; cluster orchestration remains outside Magnetar Core.

## What Changes

- `G08` (P2, effort XL): Tachyon prefill decode integration proposal.
- `G09` (P2, effort XL): Distributed session recovery and drain.
- `G10` (P3, effort XL): Evidence gated remote pooling.

Audit Tachyon interfaces before committing cross-repository APIs. Build cost-aware P/D transfer over the explicit state envelope, with global quotas, producer fencing, drain and checkpoint recovery. RDMA, remote KV/pooling and topology-specific NVLink claims require separate infrastructure experiments and go/no-go evidence.

Status: **Conditional experiment / interface proposal; no production support commitment.** All tasks are future work; authoring or validating this proposal does not implement an optimization. Indicative waves, dependencies, priorities, acceptance profiles and unresolved decisions are recorded in the [roadmap coverage index](../../../docs/roadmaps/inference-optimizations-2026-2028.md).

## Capabilities

### New Capabilities

- `distributed-inference-integration-proposal`: an implementation/qualification profile for this work, extending the existing capabilities listed below rather than replacing their baseline contracts.

### Modified Capabilities

None by default; profile-specific replacements, where required, are enumerated in the delta specs and compatibility section.

## Impact

Owning layer: Optional Tachyon-side integration and transport adapter; Magnetar local execution. Implementation may span Magnetar and the appropriate pinned Component, Provider, Format or Loader repositories; submodule revisions are updated only after their own validation. No runtime code, dependencies, WIT or canonical specifications are changed by this proposal.

Existing contracts to reuse:
- [`project-architecture`](../../specs/project-architecture/spec.md)
- [`session`](../../specs/session/spec.md)
- [`inference-api`](../../specs/inference-api/spec.md)
- [`multi-device-placement`](../../specs/multi-device-placement/spec.md)

## Dependencies

- `E01`: [`serve-continuous-batches-with-qos`](../serve-continuous-batches-with-qos/proposal.md).
- `G05`: [`enforce-local-multigpu-and-expert-placement`](../enforce-local-multigpu-and-expert-placement/proposal.md).
- `G07`: [`export-and-import-versioned-inference-state`](../export-and-import-versioned-inference-state/proposal.md).

Dependencies are exit-gate prerequisites, not a ban on independent design or hardware experiments.

## Compatibility and authority

Keep the existing F32/single-Device first-native profile valid. Components import only authorized portable graph/model/compute/state Capabilities and export model graph or scoped result Capabilities where this scope needs them; tooling-only changes add no Component imports/exports. They never choose Providers or Devices or receive native handles. Provider implementations expose real Devices and only qualified compute/movement profiles; native kernels, allocation, streams, events and compiler objects remain Provider-owned. Runtime derives ResourceAffinity and resolves eligible candidates before ranking.

No WIT change is assumed by this draft. Prefer existing contracts. If new portable operators, state or result schemas are necessary, first design a versioned additive WIT extension, validate old/new linking and reject unsupported profiles explicitly; do not alter existing signatures in place. Stateful work uses versioned leases, committed token boundaries and completion-safe retirement, and revalidates trust, affinity and capacity after failure. No optimization creates implicit movement or grants filesystem, network, process, source-control, secret or service authority. Any external adapter declares explicit allowed sources, destinations and identities outside Runtime Core.

## Non-goals

- No claim of delivered performance or production support from contracts, fixtures or Device discovery alone.
- No vendor code in Runtime Core, model-family Provider or native types in WIT.
- No automatic checkpoint downloads, cluster management or mandatory Tachyon dependency.
- No work outside G08, G09, G10 except the explicitly named qualification and compatibility prerequisites.
