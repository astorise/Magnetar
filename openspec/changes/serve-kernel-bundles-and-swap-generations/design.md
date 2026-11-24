## Context

- The engine hosts Components with Wasmtime under `StoreLimits`, fuel and epoch interruption (`magnetar-engine/src/component_wasmtime.rs`). Component Artifacts are verified against `ComponentTrustStore` (digest pins, Ed25519 publisher keys).
- The registry models prepared generations, atomic promotion, rollback with a retention window, and revocation with an in-flight policy (`kernel_registry.rs`). Ready Plans avoid repeated registry resolution (`Ready Plan Avoids Repeated Full Registry Resolution`, `Registry Change Does Not Mutate Existing Plan`).
- Step captures exist on both Providers:
  - the CUDA Provider: CUDA graphs, keyed by the decode graph fingerprint and pooled KV store set;
  - the optimized CPU Provider: recorded invocations.

  A capture key does not yet include which prepared kernel generations the capture launched.

## Goals / Non-Goals

**Goals:**

- Serve bytes from a Component with no authority beyond returning bytes.
- Promote, roll back and revoke without a restart, without mutating running work, and without releasing a module still referenced.

**Non-Goals:**

- Streaming WIT, Component-driven selection, cross-process isolation of kernels.

## Decisions

### D1. The WIT contract

```wit
package magnetar:kernel-source@0.1.0;

interface source {
  enum availability {
    /// The Component serves the bytes through `read-chunk`.
    embedded,
    /// The Runtime may resolve the manifest digest from an operator-authorized source.
    resolvable,
  }

  record bundle-descriptor {
    /// A label for diagnostics; never an identity.
    label: string,
    /// "sha256:<hex>" of the canonical kernel.manifest.json.
    manifest-digest: string,
    /// Declared size of the bundle archive in bytes: a claim the Runtime bounds.
    size: u64,
    availability: availability,
  }

  variant read-error {
    unknown-bundle,
    out-of-range,
    unavailable(string),
  }

  list-bundles: func() -> list<bundle-descriptor>;

  read-chunk: func(manifest-digest: string, offset: u64, max-bytes: u32)
    -> result<list<u8>, read-error>;
}

world kernel-package {
  export source;
}
```

- The bytes are the archive of a Kernel Exchange Bundle (the `application/vnd.magnetar.kernel-bundle.v1+tar+gzip` format). The existing extractor, path safety and validation apply unchanged.
- `manifest-digest` names the manifest only. The bundle identity is established after extraction (lot 0, D1).
- Chunked, synchronous reads are the MVP. An asynchronous or stream resource can come in `0.2.0` without changing `0.1.0`.

### D2. The host

- Verify the Component Artifact as any Component. Instantiate it with an empty linker, so any import fails linking.
- Apply `StoreLimits` (linear memory cap), fuel per call and an epoch deadline per call. A trap or exhaustion is a source failure for that transaction only.
- Drive `list-bundles` and `read-chunk` through the lot 0 gateway. Each Component call is one bounded chunk.
- A `resolvable` descriptor is resolved by the Runtime from operator-configured sources through `ArtifactSourceAuthority`, looking up the manifest digest. The Component supplies nothing else: no URL, no credentials.
- The observed source is `ComponentServed { component_digest }` for embedded bytes, and the authorized OCI locator for resolved ones.

### D3. Generation references

```text
GenerationRef = (KernelId, PreparedKernelGeneration)
```

- Each holder of dispatch-ready state records the set of generations it can launch and holds a counted lease on each:
  - a Ready Plan;
  - a step capture (CPU or CUDA);
  - an in-flight request's plan.
- `retire_prepared_kernel` moves a generation to draining. It is destroyed, and its Provider module released, only when its lease count is zero. The rollback window still applies (`destroy_prepared_kernel_with_rollback_window`).
- Leases are released on plan drop, capture release and request completion or cancellation. A leak shows up as a draining generation that never reaches zero, which is observable and tested.

### D4. Promotion boundary

- Promotion publishes a new registry revision. The safe boundary is the plan: a plan resolved before the revision keeps its generations, and the next plan resolution after it picks N+1.
- A request that is mid-decode keeps its plan, hence N, to completion. This matches `Plan Switch Occurs At Safe Generation Boundary` and `Dynamic Tuning Publication Preserves In-Flight Execution`.
- Capture keys gain the generation set:
  - a capture keyed on N is not reused by a step resolved to N+1, which records a new capture;
  - the old capture is released when no pooled set references it, releasing its leases.

### D5. Rollback and revocation

- **Rollback** re-promotes the retained generation N while it is inside its window; otherwise rollback fails with a structured error. Draining N+1 then follows D3.
- **Revocation** removes the generation from selection immediately. In-flight requests then follow `RevocationInFlightPolicy`:
  - `AllowToComplete`: they finish on it;
  - `CancelIfSafe`: they are cancelled at the next step boundary;
  - `FailClosed`: they fail there with a structured error.

  Draining applies in every case.
- Revocation survives re-import (`Revocation Survives Re-Import`).

### D6. Answers to RFC questions

- **Q2, the WIT interface:** chunked reads for the MVP, a stream resource later, and manifest identity only in WIT (D1).
- **Q5, the activation boundary:** the plan, with capture keys carrying generations (D4).

## Risks / Trade-offs

- **Recapture after promotion costs one warm step per pooled store set.** That cost is paid once per promotion, not per token.
- **Lease accounting across the CUDA executor adds bookkeeping** to capture creation and release. These are not per-token paths, because a replay reuses the lease taken at capture.
- **A kernel-package Component could serve a bundle larger than useful.** The gateway quota bounds it whatever the Component does.

## Open Questions

- Whether kernel-package Components are discovered from a configured directory or named explicitly in deployment configuration. Proposed: explicit configuration, like Model Artifacts.
