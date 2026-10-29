## 1. Scope and compatibility

- [ ] 1.1 Reconcile D07, E07 with the existing contracts linked in proposal.md and record the exact initial supported profile.
- [ ] 1.2 Confirm upstream exit-gate dependencies, hardware access, checkpoint authorization, numeric/quality thresholds and memory/SLO budgets.
- [ ] 1.3 Prefer current WIT; if new portable operators/state/results are needed, design a versioned additive extension, migration and unsupported behavior before implementation.

## 2. D07 Operational compatible prefix cache

- [ ] 2.1 Implement D07: Prefix reuse SHALL use versioned tokenized fingerprints including model, tokenizer, template, adapter, position, attention configuration, dtype/layout and sharing scope; Radix/Trie choices SHALL be benchmarked.
- [ ] 2.2 Add the positive conformance case: when compatible authorized sessions reuse a sealed prefix, verify prefill work is skipped only for the validated prefix and retained KV pages remain owned.
- [ ] 2.3 Add the failure/isolation case: when tokens match but adapter, tenant scope or model identity differs, verify the lookup is a safe miss and no cross-session state is exposed.

## 3. E07 Measured prefix savings in batches

- [ ] 3.1 Implement E07: Batch scheduling SHALL integrate compatible prefix reuse and report actual avoided prefill tokens/time separately from cache lookups and prefetch hits.
- [ ] 3.2 Add the positive conformance case: when a batch contains requests with authorized common prefixes, verify results match uncached generation and useful saved-prefill work is reported.
- [ ] 3.3 Add the failure/isolation case: when prefix sharing increases memory pressure or crosses a privacy scope, verify admission uses a safe miss or eviction without counting lookup hits as performance savings.

## 4. Qualification, rollout and documentation

- [ ] 4.1 Run affected cargo fmt/check/clippy/tests and conformance in both relevant workspaces and affected extension repositories; keep baseline F32 tests passing.
- [ ] 4.2 Validate any changed WIT and old/new linking, OpenSpec deltas and native/wasm boundaries; test Linux, Windows/macOS and ARM where applicable, with explicit hardware skips.
- [ ] 4.3 Exercise cancellation, failure, memory pressure, invalid artifact/state reuse and tenant/session isolation; prove lease retirement waits for completion.
- [ ] 4.4 Execute versioned before/after real-checkpoint campaigns on appropriate real hardware using shared V0 evidence, recording tail latency, total memory, I/O, numeric and downstream quality. Do not mark skipped hardware passed.
- [ ] 4.5 Check third-party licenses, SBOM, signatures/provenance and artifact reproducibility where native libraries or generated artifacts change.
- [ ] 4.6 Verify targeted disable, safe fallback/Unsupported, pinned mode and generation rollback; document support limits, public architecture/API migration and measurements.
- [ ] 4.7 Update the roadmap coverage/support matrix with evidence and actual maturity only after implementation gates pass; archive only completed scope.
