## Why

Lot 2 of the [accelerator packages RFC](../../../docs/rfcs/accelerator-packages-2026-11-23.md) delivers two things.

**1. Bundles served by WebAssembly Components.** A Component can carry kernel bundles embedded in it (offline installs), or list bundles whose bytes the Runtime resolves from an authorized OCI source. Today no WIT contract lets a Component serve bytes: the Runtime's packages are `model-component-graph@1.12.0`, `compute@2.0.0`, `decision@1.0.0` and `observability@1.0.0`.

**2. Hot swap without restart.**
- Generation N+1 of a kernel is prepared while requests run on generation N, promoted atomically at a safe boundary, and N is drained, retired and released once nothing references it. A rollback returns to a generation known to be correct.
- The registry already models the generation calls: `promote_generation`, `rollback_generation`, `retire_prepared_kernel`, `destroy_prepared_kernel_with_rollback_window` and `RevocationInFlightPolicy`.
- What is missing is everything that holds a reference to a generation: prepared plans, CPU and CUDA step captures (including pooled decode captures reused across requests), and in-flight requests. Without tracking those references, a Provider could unload a module a CUDA graph still launches.

## What Changes

- **WIT `magnetar:kernel-source@0.1.0`**, world `kernel-package`. It exports `source` with `list-bundles` (descriptors: label, manifest digest, declared size, availability) and `read-chunk(manifest-digest, offset, max-bytes)`. It imports nothing, and it is additive: no existing package changes.
- **Runtime host for kernel-package Components:**
  - the Component is verified as a Component (existing trust store) and instantiated with no imports, the engine's `StoreLimits`, fuel and epoch deadline;
  - it is driven as a lot 0 Kernel Artifact Source;
  - a descriptor marked as resolvable from an authorized source is fetched by the Runtime, by manifest digest, through `ArtifactSourceAuthority`; the Component never reaches the network.
- **Generation references.** A Ready Plan, a step capture and an in-flight request each hold a counted reference to the prepared generations they dispatch. Draining waits for the count to reach zero, and a Provider releases a module only then.
- **Atomic promotion at plan boundaries:**
  - promoting N+1 publishes a new registry revision;
  - plans resolved before it keep N until the next plan boundary;
  - captures that recorded N are invalidated and recaptured on the next warm step: the CPU and CUDA capture keys gain the generation set;
  - requests in flight finish on N.
- **Rollback and revocation:**
  - rollback re-promotes a retained generation inside the rollback window;
  - revocation stops new work at once, and in-flight work follows `RevocationInFlightPolicy` (allow to complete, cancel if safe, or fail closed at the next step boundary);
  - retired modules are released only after drain.
- **Promotion policy (RFC Q9, decided)**, per hardware target, from an interleaved A/B comparison on a qualified workload:
  - a significant gain of at least 10% with no significant regression promotes automatically;
  - between 5% and 10%, an operator may promote manually;
  - below 5%, nothing is promoted;
  - any promotion is watched, and rolled back automatically on a significant regression or a correctness failure.
- **Adversarial tests:** corruption, wrong digest, wrong architecture, trust refused, concurrent load, cancellation mid-read, revocation during a request, handle leaks.

## Capabilities

### New Capabilities

- `kernel-source-component`: the `kernel-package` world, its host, and the Component-served and OCI-resolved sources.
- `kernel-generation-swap`: generation references, promotion at plan boundaries, capture invalidation, drain, rollback and revocation.
- `kernel-promotion-policy`: evidence per hardware target, gain and significance, guarded metrics, promotion bands, automatic rollback.

### Modified Capabilities

- `wit`: the additive `magnetar:kernel-source@0.1.0` package (added requirement).

## Impact

- `magnetar-runtime/wit/kernel-source.wit` (new), WIT validation and old/new linking tests.
- `magnetar-engine`: instantiating `kernel-package` Components with no imports.
- `magnetar-kernel` and `magnetar-runtime`: generation references, plan and capture keys, drain.
- `providers/cuda` and `providers/cpu`: capture keys include prepared generations; release after drain.
- A fixture Component (`components/kernel-package-fixture`, test-only) embedding the lot 1 bundles.

## Dependencies

- Lot 0 ([`define-kernel-artifact-sources`](../define-kernel-artifact-sources/proposal.md)) and lot 1 ([`prepare-prebuilt-cuda-kernels`](../prepare-prebuilt-cuda-kernels/proposal.md)).

## Compatibility and authority

- Purely additive WIT. A model Component neither imports nor exports `kernel-source`, and a kernel-package Component cannot import graph, compute or any other Capability.
- Instantiation grants no filesystem, network, process, clock or random authority. Reads are bounded by the lot 0 gateway, and the Component's own execution by the engine's limits.
- Promotion never mutates a running plan or a captured graph: it publishes a new revision and invalidates.

## Non-goals

- Streaming or asynchronous WIT reads (chunked reads are the MVP, RFC question 2).
- A Component choosing which generation runs, or promoting anything.
- Compiled graphs (lot 4) and publication (lot 5).
