# RFCs

## Accelerator packages (2026-11-23)

[`accelerator-packages-2026-11-23.md`](accelerator-packages-2026-11-23.md) (in French): specialized kernels and accelerator packages, distributed independently of models and Providers, admitted and activated by the Runtime.

Each lot is an OpenSpec change. All are proposed and none is implemented yet.

| Lot | Change | Scope |
| --- | --- | --- |
| 0 | [`define-kernel-artifact-sources`](../../openspec/changes/define-kernel-artifact-sources/proposal.md) | Kernel Artifact Sources with bounded reads, distinct identities, a kernel trust store with Ed25519 signatures, execution permissions separate from trust, the delivery lifecycle |
| 1 | [`prepare-prebuilt-cuda-kernels`](../../openspec/changes/prepare-prebuilt-cuda-kernels/proposal.md) | The Provider preparation capability, the `magnetar:cuda-launch@1` launch descriptor, CUDA preparation of precompiled CUBINs, the first hardware-verified external kernel |
| 2 | [`serve-kernel-bundles-and-swap-generations`](../../openspec/changes/serve-kernel-bundles-and-swap-generations/proposal.md) | WIT `magnetar:kernel-source@0.1.0` (`kernel-package` world), embedded and OCI-resolved bundles, generation leases, promotion at plan boundaries, drain, rollback, revocation |
| 3 | [`load-specialized-cpu-kernels`](../../openspec/changes/load-specialized-cpu-kernels/proposal.md) | The `magnetar:cpu-kernel@1` ABI, ELF checks before loading, feature-matched variants with tested fallback, a measured sandboxed tier |
| 4 | [`prepare-compiled-graph-artifacts`](../../openspec/changes/prepare-compiled-graph-artifacts/proposal.md) | The `magnetar:accelerator-package@1` envelope, compiled graphs with exact segment or instance substitution, other Providers gated on their compute paths, the support matrix |
| 5 | [`publish-accelerator-packages`](../../openspec/changes/publish-accelerator-packages/proposal.md) | A package library built and qualified from source, evidence as OCI referrers, third-party ports with provenance and licenses, discovery without installation, a revocation feed |

The RFC's open questions (section 14) and where each is answered:

| Question | Answer |
| --- | --- |
| 1. Publication granularity | lot 0 design, D5 |
| 2. The WIT interface | lot 2 design, D1 and D6 |
| 3. Association with model Components | lot 0 design, D5 |
| 4. Trust policy | lot 0 design, D3; the signing key holder is open |
| 5. The hot swap boundary | lot 2 design, D4 |
| 6. The CPU ABI | lot 3 design; decided after the measurement of D5 |
| 7. CompiledGraph | lot 4 design, D1 and D6 |
| 8. Intel | deferred to `evaluate-hardware-gated-npu-compute` (G04) |
| 9. Benchmarks and thresholds | lot 1 design (proposal); lot 5, open |
