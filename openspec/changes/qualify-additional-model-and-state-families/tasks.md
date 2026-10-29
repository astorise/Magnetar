## 1. Scope and compatibility

- [ ] 1.1 Reconcile F01, F04 with the existing contracts linked in proposal.md and record the exact initial supported profile.
- [ ] 1.2 Confirm upstream exit-gate dependencies, hardware access, checkpoint authorization, numeric/quality thresholds and memory/SLO budgets.
- [ ] 1.3 Prefer current WIT; if new portable operators/state/results are needed, design a versioned additive extension, migration and unsupported behavior before implementation.

## 2. F01 Real dense family qualification

- [ ] 2.1 Implement F01: Model Components and loaders SHALL qualify a second dense family such as Llama or Gemma and additional Qwen versions using real checkpoints, tokenizer, RoPE, graph and logit conformance.
- [ ] 2.2 Add the positive conformance case: when a declared family profile executes a real checkpoint, verify family-specific semantics are supplied through portable graph/model Capabilities and compared with a trusted reference.
- [ ] 2.3 Add the failure/isolation case: when configuration identifies an unsupported architecture variant, verify ingestion or Component binding rejects it without guessing compatibility from a family name.

## 3. F04 Specialized attention and recurrent state

- [ ] 3.1 Implement F04: Specialized Components SHALL express MLA, GQA, SWA, sparse attention and KDA/recurrent semantics through validated portable operators and state schemas with explicit supported profiles.
- [ ] 3.2 Add the positive conformance case: when a declared specialized checkpoint generates across state boundaries, verify position/window/compression/recurrent updates and quality match its reference.
- [ ] 3.3 Add the failure/isolation case: when a Provider lacks the operator or the state schema is incompatible, verify planning rejects the profile without embedding model-family or vendor code in Runtime Core.

## 4. Qualification, rollout and documentation

- [ ] 4.1 Run affected cargo fmt/check/clippy/tests and conformance in both relevant workspaces and affected extension repositories; keep baseline F32 tests passing.
- [ ] 4.2 Validate any changed WIT and old/new linking, OpenSpec deltas and native/wasm boundaries; test Linux, Windows/macOS and ARM where applicable, with explicit hardware skips.
- [ ] 4.3 Exercise cancellation, failure, memory pressure, invalid artifact/state reuse and tenant/session isolation; prove lease retirement waits for completion.
- [ ] 4.4 Execute versioned before/after real-checkpoint campaigns on appropriate real hardware using shared V0 evidence, recording tail latency, total memory, I/O, numeric and downstream quality. Do not mark skipped hardware passed.
- [ ] 4.5 Check third-party licenses, SBOM, signatures/provenance and artifact reproducibility where native libraries or generated artifacts change.
- [ ] 4.6 Verify targeted disable, safe fallback/Unsupported, pinned mode and generation rollback; document support limits, public architecture/API migration and measurements.
- [ ] 4.7 Update the roadmap coverage/support matrix with evidence and actual maturity only after implementation gates pass; archive only completed scope.
