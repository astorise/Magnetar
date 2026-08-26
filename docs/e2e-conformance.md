# End-to-End Local Inference Conformance Suite

The End-to-End Local Inference Conformance suite is the correctness gate for
the first executable Magnetar baseline. It proves that a small, deterministic
local model fixture can move through the complete Runtime inference pipeline
-- resolution, loading, Model Instance creation, session creation,
tokenization, generation, streaming, and cleanup -- without bypassing any
Runtime contract.

The suite runs entirely on CPU. It requires no GPU hardware, no CUDA, Metal,
OpenVINO, or QNN, no network access, no Git access, no workspace scanning, no
shell/process execution, and no Tachyon distributed orchestration.

## Fixture

The suite defines a minimal Qwen-like decoder-only fixture: one layer, hidden
size 4, two attention heads, head dimension 2, intermediate size 8, and a
byte-level tokenizer fixture. Fixture weights are deterministic (derived from
a pure hash function, no RNG dependency), and the fixture Model Artifact
manifest passes normal Model Artifact and Qwen baseline validation like any
other Model Component.

Unlike a purely structural fixture, the success path drives an actual forward
pass through the real Reference CPU numeric kernels -- embedding lookup,
RMSNorm, matmul, RoPE, attention, softmax, SiLU, elementwise add/mul, and
residual-add -- so generated output is genuinely deterministic rather than a
canned stub.

## What The Suite Validates

- The full required path: resolve model, load model, create Model Instance,
  create session, tokenize (plain text, already-tokenized, and chat-message
  prompts), run generation (prefill, decode, Sampling, streaming), return a
  result with usage accounting, close the session, and clean up resources.
- The one-shot inference path exercises the same normal Model Instance,
  Tokenizer, Generation, Sampling, and Kernel Registry paths as the
  multi-call path.
- No-shortcut detection: direct Provider/Kernel invocation, Model Loading
  bypass, Model Component bypass, and Memory Manager bypass are all
  detected and rejected; dtype/layout conversion is always explicit, never
  silent.
- Reference CPU is selected through Kernel Registry, not used as a hidden
  fallback.
- Operator coverage for every required-now operator in the first decoder
  baseline.
- Execution graph production, validation, planning, and execution for both
  prefill and decode graphs, including a deliberately invalid graph fixture.
- Generation behavior: max new tokens, max total tokens, EOS, cancellation,
  finish reason, usage accounting, and streaming event ordering.
- Sampling Contract usage for both greedy (success path) and seeded
  stochastic selection.
- Session lifecycle, including rejection of generation against a closed
  session.
- KV Cache and Prefix Cache lifecycle (allocation, prefill append, decode
  append/lookup, cleanup) with raw cache contents and prompts never exposed
  in observations.
- Tensor Resource lifecycle (descriptor, allocation, readiness, release) and
  Memory Manager accounting for operator-output and workspace allocations,
  with no allocation left untracked after release.
- CLI boundary: Runtime never reads workspace files, executes Git, executes
  tools, or executes shell/process commands, and never receives ambient CLI
  authority.
- Diagnostics and observability redaction: no raw prompts, weights, tensor
  values, cache contents, secrets, or handles are exposed by default.
- Fourteen structured failure cases (invalid model reference, untrusted
  artifact, incompatible tokenizer, unsupported operator, missing kernel,
  invalid tensor shape, memory admission failure, closed session,
  cancellation, timeout, policy denial, raw handle access denial, Runtime
  file access denial, Runtime tool execution denial), each reporting a
  stable `e2e-*` error code.
- Determinism: the success path produces identical generated tokens across
  repeated runs.

## Report Format

The suite emits a structured report through `E2eConformanceReport`, including
suite/fixture/Runtime version, Provider/Device/Model Component summaries,
operator and kernel coverage, per-test-case pass/fail/skipped status with
redacted diagnostics, a redaction flag, and duration metadata. Use
`e2e_conformance_report_json` to produce machine-readable JSON output.

## Local Commands

Run the suite:

```powershell
cargo test -p magnetar-runtime e2e_conformance -- --nocapture
```

Run the full Runtime suite:

```powershell
cargo test --workspace --all-targets
```

Validate the OpenSpec change:

```powershell
openspec validate define-end-to-end-local-inference-conformance --strict
```

## Compatibility Versioning

The current suite version is `0.1.0`, exposed as `E2E_SUITE_VERSION`. The
fixture version is tracked separately as `E2E_FIXTURE_VERSION`. Passing one
suite version does not imply passing future versions as the fixture model
grows toward a real Qwen baseline.
