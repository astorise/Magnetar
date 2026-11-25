Status: **Proposed (RFC lot 2).** No task is implemented yet. Requires lots 0 and 1.

## 1. WIT and host

- [ ] 1.1 `magnetar-runtime/wit/kernel-source.wit` (design D1). WIT validation; old/new linking tests showing every existing Component still links.
- [ ] 1.2 `magnetar-engine`: instantiate a kernel-package Component with an empty linker, `StoreLimits`, per-call fuel and epoch deadline, after Component trust verification.
- [ ] 1.3 A `ComponentServedSource` implementing the lot 0 `KernelArtifactSource` over `list-bundles` and `read-chunk`.
- [ ] 1.4 Resolution of `resolvable` descriptors by manifest digest from operator-configured sources through `ArtifactSourceAuthority`. With no configured source, no request is made.
- [ ] 1.5 `components/kernel-package-fixture` (test-only): embeds the lot 1 bundles and lists one resolvable bundle.

## 2. Generation leases and swap

- [ ] 2.1 `GenerationRef` leases held by Ready Plans, step captures and in-flight request plans; released on drop, release, completion or cancellation.
- [ ] 2.2 Drain: a retired generation is destroyed, and its module released, only at zero leases and per its rollback window.
- [ ] 2.3 Capture keys (CUDA pooled decode and prefill captures, CPU captures) include the generation set. A step resolved to another generation records a new capture.
- [ ] 2.4 Promotion publishes a revision without mutating plans. Rollback re-promotes within the window. Revocation follows `RevocationInFlightPolicy`.
- [ ] 2.5 Observability: lease counts per generation, draining generations, promotions, rollbacks and revocations, with bundle identities and no payload data.

## 3. Promotion policy (design D6)

- [ ] 3.1 Hardware target identity (Provider, Device architecture or CPU model with features, driver or runtime major version); evidence and promotions keyed on it.
- [ ] 3.2 Interleaved A/B comparison on a qualified workload, off the token path:
  - a policy minimum of 10 runs per generation;
  - a bootstrap 95% interval of the relative gain on the primary latency metric;
  - guarded metrics (TTFT, TPOT, p95, peak memory, error rate) with a 2% default tolerance;
  - correctness checks.
- [ ] 3.3 Bands: automatic promotion at 10% or more; manual promotion offered from 5% to below 10%; none below 5%; each decision recorded with its evidence.
- [ ] 3.4 Automatic rollback within the observation and rollback windows, recorded with its reason, and a bar on automatic re-promotion until new evidence.
- [ ] 3.5 Tests with scripted measurements:
  - 11% significant (automatic); 12% not significant (refused); 7% (manual only); 3% (refused);
  - 12% with a significant 8% memory regression (refused);
  - evidence from another target (refused);
  - a regression after promotion (rolled back and barred).

## 4. Adversarial and concurrency tests

- [ ] 4.1 Sources: corrupt archive, wrong digest, a size lie, an oversized chunk, a Component that traps, loops or imports; a resolvable bundle without a configured source; the same bundle embedded and from OCI.
- [ ] 4.2 Trust: a trusted Component serving an unsigned bundle, a revoked key, a wrong architecture (`sm_90` on `sm_86`).
- [ ] 4.3 Swap, on the GPU runner:
  - prepare N+1 while 4 concurrent requests decode on N;
  - promote; check in-flight tokens are unchanged and the next requests use N+1;
  - roll back; revoke under each in-flight policy;
  - cancel requests mid-drain;
  - check modules are released at zero leases and nothing leaks after 100 promote and rollback cycles.
- [ ] 4.4 The same swap tests on the optimized CPU Provider's captures (lot 3 kernels, or a test-only prepared kernel).
- [ ] 4.5 `cargo fmt`, `clippy -D warnings`, tests of both workspaces, and the Providers' tests.

## 5. Documentation

- [ ] 5.1 `docs/architecture/kernel-artifact-sources.md`: the kernel-package world, embedded and resolvable bundles, leases, promotion boundary, drain, rollback.
- [ ] 5.2 `docs/architecture/component-runtime.md`: the kernel-package role and its empty import set.
- [ ] 5.3 RFC acceptance criteria 1, 2, 3, 6, 7 and 8 (section 13), checked against the tests.
- [ ] 5.4 `docs/architecture/kernel-artifact-sources.md`: the promotion policy (bands, significance, guarded metrics, per-target scope, automatic rollback) and its parameters.
