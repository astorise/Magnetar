## 1. Scope and compatibility

- [ ] 1.1 Reconcile G05, G06 with the existing contracts linked in proposal.md and record the exact initial supported profile.
- [ ] 1.2 Confirm upstream exit-gate dependencies, hardware access, checkpoint authorization, numeric/quality thresholds and memory/SLO budgets.
- [ ] 1.3 Prefer current WIT; if new portable operators/state/results are needed, design a versioned additive extension, migration and unsupported behavior before implementation.

- [ ] 1.4 Concrete contract migration: reconcile `multi-device-placement` / "Multi-Device Placement Plan Does Not Yet Gate Execution" with "Production Multi Device Placement Gates Execution" and test both old baseline and new qualified profile.

## 2. G05 Enforced local multi GPU placement

- [ ] 2.1 Implement G05: Production multi-GPU profiles SHALL enforce versioned layer/tensor placement, per-Device reservations and explicit P2P, host-copy or collective edges before submission.
- [ ] 2.2 Add the positive conformance case: when a real model is partitioned across distinct local GPUs, verify logits and multi-step tokens match the unsplit reference and movement/capacity costs are recorded.
- [ ] 2.3 Add the failure/isolation case: when a Device is lost or a required peer link/collective is unavailable, verify dependent plans invalidate and only an explicitly authorized replan may resume with compatible state.

- [ ] 2.4 Confirm in CI whether the GeForce RTX 3060 pair has peer access (log `cuDeviceCanAccessPeer`, make the two-GPU joined-capture test report when it skips); if peer access can be enabled, compare direct-read reductions with the host-staged ones. Peer access on GeForce needs a patched NVIDIA open kernel module (BAR1 P2P, e.g. the aikitoria fork) on the runner host, with Resizable BAR covering each GPU's VRAM and IOMMU passthrough; the CI prints the runner's readiness. https://github.com/Zagitta/gpu-unlock does not provide peer access (it unlocks full-rate FP16/BF16/TF32 tensor cores, relevant to prefill); see design.md, "Hardware limit: no peer access between the CI's two RTX 3060".

## 3. G06 Conditional expert parallelism

- [ ] 3.1 Implement G06: Multi-GPU MoE SHALL qualify expert placement, duplication and load balancing under explicit interconnect capacity, routing ownership and per-Device budgets.
- [ ] 3.2 Add the positive conformance case: when a real multi-GPU MoE workload changes expert demand, verify expert balance, communication, duplication memory and accepted throughput are measured.
- [ ] 3.3 Add the failure/isolation case: when rebalancing would violate affinity or exceed link/memory budgets, verify the new placement is rejected and the current valid generation remains or fails explicitly.

## 4. Qualification, rollout and documentation

- [ ] 4.1 Run affected cargo fmt/check/clippy/tests and conformance in both relevant workspaces and affected extension repositories; keep baseline F32 tests passing.
- [ ] 4.2 Validate any changed WIT and old/new linking, OpenSpec deltas and native/wasm boundaries; test Linux, Windows/macOS and ARM where applicable, with explicit hardware skips.
- [ ] 4.3 Exercise cancellation, failure, memory pressure, invalid artifact/state reuse and tenant/session isolation; prove lease retirement waits for completion.
- [ ] 4.4 Execute versioned before/after real-checkpoint campaigns on appropriate real hardware using shared V0 evidence, recording tail latency, total memory, I/O, numeric and downstream quality. Do not mark skipped hardware passed.
- [ ] 4.5 Check third-party licenses, SBOM, signatures/provenance and artifact reproducibility where native libraries or generated artifacts change.
- [ ] 4.6 Verify targeted disable, safe fallback/Unsupported, pinned mode and generation rollback; document support limits, public architecture/API migration and measurements.
- [ ] 4.7 Update the roadmap coverage/support matrix with evidence and actual maturity only after implementation gates pass; archive only completed scope.

## Inherited items

- [ ] An early refusal for a prefill that cannot fit (inherited from `bound-one-pass-prefill-memory` 3.2, closed there on 2026-11-25 by the owner's decision). DeepSeek-V2-Lite-Chat Q8_0 across two 12 GB GPUs with an `f32` latent history at 16,384 tokens failed with `KernelOutOfDeviceMemory` (a structured memory-limit finish) after 74 s, at layer 25, with 176 MiB free on the second GPU and 1,806 MiB on the first (run 44382825307). The history's size is known at the first step; what does not fit is the prefill's working memory on one Device, which only the placement knows. Wanted: a per-Device free-memory query and a per-segment peak estimate at the first prefill, refusing only when the estimate's lower bound already exceeds the Device's free memory (no false refusal), cached with the plan so no decode step pays for it. That case now fits (run 45576847703: TTFT 82.9 s), so the acceptance case is a longer prompt or a larger model that does not fit. Acceptance: the same request is refused before the first layer runs, with the shortfall named; TTFT and TPOT of the requests that fit are unchanged in the same GPU run.
