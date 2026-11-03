## 1. Scope and compatibility

- [x] 1.1 Reconcile A01, A02, A03, A04, A07, A08 with the existing contracts linked in proposal.md and record the exact initial supported profile. Initial profile recorded in design.md ("Initial implemented profile").
- [ ] 1.2 Confirm upstream exit-gate dependencies, hardware access, checkpoint authorization, numeric/quality thresholds and memory/SLO budgets.
- [x] 1.3 Prefer current WIT; if new portable operators/state/results are needed, design a versioned additive extension, migration and unsupported behavior before implementation. No WIT change: the harness only observes the existing streaming boundary.

## 2. A01 Real checkpoint performance harness

- [x] 2.1 Implement A01: The qualification harness SHALL execute pinned Qwen2.5-0.5B-Instruct multi-step generation and record TTFT, TPOT, throughput, p50/p95 latency, RSS/VRAM, utilization, I/O bytes, context and batch sizes. Implemented by `tools/inference-qualification` and `integration-tests/production-loading/src/tests_real_checkpoint_qualification.rs`; device memory is recorded as not-measured until a Provider exposes device-memory telemetry.
- [x] 2.2 Add the positive conformance case: when a real checkpoint completes prefill and multiple decode steps, verify the report contains measured phase timings, token counts and resource peaks tied to the run.
- [x] 2.3 Add the failure/isolation case: when only synthetic fixture execution is available, verify the run is labeled fixture-only and cannot qualify production performance.

## 3. A02 Numerical and lifecycle reference corpus

- [x] 3.1 Implement A02: The harness SHALL maintain checkpoint-bound token/logit references, per-dtype tolerances, and streaming, resume and cancellation checks against Reference CPU and each qualified Provider. Implemented by `tools/inference-qualification::numerics`, `ProductionLoadedModel::generate_streaming_with_logits` and `integration-tests/production-loading/src/tests_real_checkpoint_numerics.rs`, against the checked-in Reference CPU corpus.
- [x] 3.2 Add the positive conformance case: when CPU and CUDA execute the same pinned checkpoint and token inputs, verify logits and tokens are compared under the declared numerical profile and lifecycle events are checked. GPU run 41008921749 (RTX 3060): CUDA tokens identical to the independent corpus, max |logit difference| 6.6e-5; streaming, cancellation and resume checks pass.
- [x] 3.3 Add the failure/isolation case: when logits exceed the declared tolerance despite identical decoded text, verify qualification fails and preserves the differential evidence.

## 4. A03 Hardware capability evidence

- [ ] 4.1 Implement A03: The qualification matrix SHALL distinguish AVX2, AVX-512, NEON, CUDA SM, graph capture, FP16/BF16, bandwidth, HIP and WGPU support using discovered hardware and executed tests.
- [ ] 4.2 Add the positive conformance case: when a target machine completes capability probes and real compute tests, verify the report names the tested Device and separates discovery, compute and production validation.
- [x] 4.3 Add the failure/isolation case: when required hardware is absent, verify the target is explicitly skipped with a reason and no production capability is claimed.

## 5. A04 Repeatable cold and warm campaigns

- [x] 5.1 Implement A04: Campaigns SHALL separate cold and warm runs, control ordering and interference, perform at least five post-warmup repetitions when reasonably affordable, and report median, dispersion and uncertainty keyed by code, model, Provider and driver identities.
- [x] 5.2 Add the positive conformance case: when repeated runs of an identical profile finish, verify raw samples and aggregate statistics are retained with the warmup and ordering protocol.
- [x] 5.3 Add the failure/isolation case: when five repetitions are unaffordable or results overlap measurement noise, verify the exception and uncertainty are recorded and no unsupported speedup is promoted.

## 6. A07 Fixed budget memory campaigns

- [ ] 6.1 Implement A07: Qualification SHALL compare memory-ladder and fixed-budget split profiles with enforced budgets, OOM injection and CPU/GPU saturation, accounting for weights, KV, workspace and capture/prefetch overhead.
- [ ] 6.2 Add the positive conformance case: when two residency strategies execute under the same total budget, verify reports compare throughput and tail latency alongside actual memory and I/O usage.
- [ ] 6.3 Add the failure/isolation case: when a strategy exceeds the admitted budget or exhausts capacity, verify the run reports structured OOM/admission failure rather than an invalid speedup.

## 7. A08 Downstream quality qualification

- [ ] 7.1 Implement A08: Numerical qualification SHALL include pinned perplexity or targeted task/conformance datasets and explicit acceptance thresholds in addition to token identity.
- [ ] 7.2 Add the positive conformance case: when a lower-precision or alternative model path is evaluated, verify logit errors and downstream quality deltas are compared with the declared baseline and dataset revisions.
- [ ] 7.3 Add the failure/isolation case: when text looks plausible but the quality threshold is breached, verify the profile fails qualification and retains its higher-precision fallback.

## 8. Qualification, rollout and documentation

- [ ] 8.1 Run affected cargo fmt/check/clippy/tests and conformance in both relevant workspaces and affected extension repositories; keep baseline F32 tests passing.
- [ ] 8.2 Validate any changed WIT and old/new linking, OpenSpec deltas and native/wasm boundaries; test Linux, Windows/macOS and ARM where applicable, with explicit hardware skips.
- [ ] 8.3 Exercise cancellation, failure, memory pressure, invalid artifact/state reuse and tenant/session isolation; prove lease retirement waits for completion.
- [ ] 8.4 Execute versioned before/after real-checkpoint campaigns on appropriate real hardware using shared V0 evidence, recording tail latency, total memory, I/O, numeric and downstream quality. Do not mark skipped hardware passed.
- [ ] 8.5 Check third-party licenses, SBOM, signatures/provenance and artifact reproducibility where native libraries or generated artifacts change.
- [ ] 8.6 Verify targeted disable, safe fallback/Unsupported, pinned mode and generation rollback; document support limits, public architecture/API migration and measurements.
- [ ] 8.7 Update the roadmap coverage/support matrix with evidence and actual maturity only after implementation gates pass; archive only completed scope.
