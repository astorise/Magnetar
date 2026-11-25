Status: **Proposed (RFC lot 4).** No task is implemented yet. Requires lots 0 to 2. Each Provider row is gated on its own change.

## 1. Envelope

- [ ] 1.1 `magnetar-kernel`: `magnetar:accelerator-package@1` parsing with manifest v1's limits; canonical digest; variant kinds; bundle and blob references.
- [ ] 1.2 Resolution of operator and fused variants through lots 0 to 2; an envelope grants nothing (tests: a signed envelope with an unsigned bundle, a duplicate key, an unknown kind).

## 2. Compiled graphs

- [ ] 2.1 `CompiledGraphBinding` and `SegmentSignature` types and validation (design D2).
- [ ] 2.2 Exact segment matching on prepared execution graphs; instance matching by Model Artifact; toolchain and accelerator checks against Provider reports.
- [ ] 2.3 Memory Manager admission of declared memory, and reconciliation after preparation.
- [ ] 2.4 Plans with a matched segment dispatched to the prepared graph; swap through new instances and plan revisions with lot 2 leases.
- [ ] 2.5 Tests with a fixture Provider that "prepares" a compiled graph as a known composition of Reference CPU kernels:
  - a match; a weight-digest mismatch; a driver mismatch;
  - an under-declared memory report;
  - a swap with requests in flight;
  - a compiled graph offered as an operator kernel.

## 3. Providers (each in its gating change)

- [ ] 3.1 WGPU operator kernels from `wgsl`, after `integrate-wgpu-production-compute`. Pipeline compilation is recorded as cold-path compilation.
- [ ] 3.2 ROCm operator kernels from HSACO, after `qualify-rocm-compute-profiles`.
- [ ] 3.3 Metal operator kernels from `metallib`, after `evaluate-native-metal-prefill`.
- [ ] 3.4 Intel NPU and QNN compiled graphs, after G04 (`evaluate-hardware-gated-npu-compute`) and on real hardware.
- [ ] 3.5 Optionally, IREE as a producer of compiled graphs (C07, `evaluate-iree-provider-compilation`).

## 4. Validation and documentation

- [ ] 4.1 `cargo fmt`, `clippy -D warnings`, tests of both workspaces; Linux, Windows and macOS for the envelope and matching code.
- [ ] 4.2 `docs/architecture/accelerator-support-matrix.md`, with every cell at its evidenced status. Updated only with recorded runs.
- [ ] 4.3 `docs/architecture/kernel-artifact-sources.md`: envelopes, kinds, compiled graphs, substitution.
- [ ] 4.4 RFC questions 7 and 8: answer 7 recorded (design D6), 8 referred to G04.
