# Define End-to-End Local Inference Conformance

## Why

Magnetar now has enough contracts to describe a complete local inference path:

- `magnetar-cli` boundary
- Runtime Inference API
- Model Artifact
- Model Loading
- Model Instance lifecycle
- Qwen-like Model Component baseline
- Tokenizer Contract
- Inference Session
- Generation Contract
- Sampling Contract
- Tensor Resource and Layout Contract
- Execution Graph
- Operator Contract
- Kernel Registry and Dispatch
- Reference CPU Provider
- Observability

The next step is to prove the full path works.

Unit tests and individual conformance suites are not enough.

Magnetar needs an end-to-end local conformance suite that validates the complete
inference pipeline without GPU hardware and without hidden shortcuts.

This suite becomes the correctness gate for the first executable Magnetar
baseline.

## What Changes

This change introduces End-to-End Local Inference Conformance.

The suite SHALL validate that a small local model fixture can move through the
entire Runtime pipeline and produce deterministic or tolerance-bound output.

The suite SHALL verify:

- API entrypoint
- model resolution
- model artifact validation
- model loading
- Model Component resolution
- tensor inventory validation
- Model Instance readiness
- session creation
- tokenizer encode
- generation prefill
- generation decode
- sampling
- streaming events
- tensor resource allocation
- graph production
- graph validation
- operator scope validation
- kernel registry selection
- Reference CPU dispatch
- usage reporting
- diagnostics
- cancellation
- redaction
- cleanup

## E2E Scope

The first E2E suite SHALL target local inference only.

It SHALL run without:

```text
GPU hardware
CUDA
Metal
OpenVINO
QNN
WebGPU
network access
Git access
workspace scanning
shell/process execution
tool execution
external model download
Tachyon distribution
```

The first suite MAY use test fixtures instead of large real model files.

## Fixture Model

The suite SHALL define a minimal model fixture.

The fixture SHOULD be small enough to run quickly on CPU.

It MAY be a Qwen-like decoder-only toy model with:

- minimal vocabulary
- minimal tokenizer fixture
- one or two decoder layers
- small hidden size
- small attention head count
- small intermediate size
- deterministic weights
- deterministic generation settings
- known prompt input
- expected token output or expected structured behavior

The fixture need not represent a production Qwen model.

It SHALL exercise Qwen-like architecture paths.

## Fixture Artifact

The fixture Model Artifact SHALL still pass Model Artifact validation.

It SHALL include or reference:

- manifest
- config metadata
- tensor inventory
- tokenizer fixture metadata
- generation defaults
- trust state appropriate for test
- deterministic weights
- architecture family metadata
- integrity metadata where applicable

The fixture SHALL not bypass Model Loading.

## Fixture Tokenizer

The suite SHALL include a tokenizer fixture.

Tokenizer fixture SHOULD support:

- known vocabulary
- deterministic encode
- deterministic decode
- EOS token metadata
- optional BOS token metadata
- special token metadata
- streaming decode fixture behavior

The tokenizer fixture SHALL still use Tokenizer Contract.

## Fixture Prompt

The suite SHALL define prompt inputs.

Prompt inputs MAY include:

```text
plain text
already-tokenized input
chat-message fixture
```

The first conformance path SHOULD include at least one already-tokenized path
and one text tokenization path.

Raw prompt logging SHALL remain disabled by default.

## Required Path

The E2E suite SHALL validate the normal path:

```text
resolve model
load model
create Model Instance
create session
tokenize prompt
run generation
prefill
decode
sample
stream events
return result
close session
cleanup resources
```

A test MAY combine steps through one-shot inference only if the one-shot path is
proven to use normal contracts internally.

## No Shortcut Rule

The E2E suite SHALL fail if inference bypasses required Runtime contracts.

It SHALL detect and reject shortcuts such as:

- direct Provider invocation
- direct Kernel invocation
- direct tensor pointer access
- skipping Model Loading
- skipping Model Component graph production
- skipping Kernel Registry
- skipping Memory Manager
- skipping Tokenizer for text input
- executing tools from generated text
- reading workspace files in Runtime
- silently using CPU fallback without policy
- silently converting dtype/layout

## Reference CPU Requirement

The first E2E suite SHALL run on Reference CPU Provider.

Reference CPU Provider SHALL be selected through Kernel Registry and Dispatch.

The suite SHALL verify that Reference CPU is not used as a hidden fallback.

CPU use SHALL be explicit through normal policy.

## Operator Coverage

The suite SHALL verify required-now operator coverage.

At minimum, the fixture path SHOULD exercise:

```text
embedding
rmsnorm
matmul
rope
attention
softmax
silu
add
mul
residual-add
```

DType conversion and layout conversion MAY be tested in separate E2E cases.

## Graph Validation

The suite SHALL verify that Model Component-produced graphs are validated before
execution.

It SHOULD include:

- valid prefill graph
- valid decode graph
- invalid graph fixture
- unsupported operator fixture
- invalid tensor shape fixture
- missing kernel fixture

## Generation Validation

The suite SHALL verify Generation behavior.

It SHOULD validate:

- max new tokens
- max total tokens
- stop condition
- EOS behavior
- deterministic sampling or greedy mode
- finish reason
- usage accounting
- cancellation behavior
- streaming event sequence

## Sampling Validation

The suite SHALL validate Sampling through the Sampling Contract.

Initial E2E path SHOULD use deterministic greedy sampling.

If stochastic sampling is tested, seed behavior SHALL be explicit.

The suite SHALL not require Provider-assisted sampling.

## Streaming Validation

The suite SHALL validate ordered streaming events.

Expected event order SHOULD include:

```text
generation-started
prefill-started
prefill-completed
decode-started
decode-token
decoded-text
usage-updated
stop-reached or generation-completed
stream-closed
```

The exact event names are implementation-defined but SHALL be semantically
stable.

## Session Validation

The suite SHALL validate session lifecycle.

It SHOULD include:

- session created
- session used
- session closed
- session not usable after close
- session cleanup releases inference-scoped resources
- session does not store CLI workspace/Git/tool/secret state

## Cache Validation

The first E2E suite MAY include KV cache validation.

If included, it SHOULD validate:

- cache allocation
- cache append during prefill
- cache consumption during decode
- cache cleanup
- no raw cache exposure

Prefix Cache may be optional in first E2E suite.

If Prefix Cache is enabled, hit/miss metadata SHALL be redacted.

## Tensor Validation

The suite SHALL validate Tensor Resource behavior.

It SHOULD verify:

- Tensor Descriptors created
- Tensor Resources allocated
- host contiguous layout used for Reference CPU
- dtype is explicit
- layout is explicit
- no raw pointer exposure
- output readiness updated
- tensor cleanup
- explicit conversion behavior where tested

## Memory Validation

The suite SHALL validate Memory Manager participation.

It SHOULD verify:

- model tensors are accounted
- operator outputs are accounted
- workspace is accounted where needed
- memory pressure failure path is structured
- cleanup releases resources
- no untracked Runtime-visible allocation

## CLI Boundary Validation

The suite SHALL include at least one CLI-boundary conformance case.

It SHOULD verify:

- CLI sends explicit prompt/context
- Runtime does not read workspace files
- Runtime does not execute Git
- Runtime does not execute tools
- Runtime does not execute shell/process
- Runtime does not receive ambient CLI authority
- Runtime structured errors are preserved by CLI

This case may use a test CLI harness rather than the final CLI UX.

## Diagnostics And Redaction

The suite SHALL validate diagnostics and redaction.

It SHALL verify that diagnostics and observability do not expose by default:

- raw prompts
- raw model weights
- raw tensor values
- raw KV cache contents
- secrets
- filesystem authority
- Provider handles
- Device handles
- Kernel handles
- memory pointers

## Failure Path Coverage

The E2E suite SHALL include failure cases.

Failure cases SHOULD include:

- invalid model reference
- untrusted artifact
- incompatible tokenizer
- unsupported operator
- missing required kernel
- invalid tensor shape
- memory admission failure
- session closed
- generation cancelled
- generation timeout
- policy denied
- raw handle access denied
- Runtime file access denied
- Runtime tool execution denied

## Determinism

The first E2E success path SHALL be deterministic where feasible.

Determinism MAY be achieved through:

- deterministic fixture weights
- greedy sampling
- fixed input tokens
- fixed generation limit
- Reference CPU execution
- explicit dtype/layout policy
- no stochastic sampling

Expected output SHALL be specified as tokens, decoded text, or structured result
metadata.

## Report Format

The E2E suite SHALL produce a machine-readable report.

The report SHOULD include:

- suite version
- fixture version
- Runtime version
- Provider summary
- Device summary
- Model Component summary
- Operator coverage summary
- Kernel coverage summary
- test cases
- pass/fail/skipped status
- structured failure reasons
- redaction status
- duration metadata

The report SHALL not include raw sensitive values by default.

## CI Integration

The E2E suite SHOULD run in CI without GPU hardware.

CI SHALL be able to run:

```text
unit tests
contract tests
provider conformance
first operator scope conformance
E2E local inference conformance
OpenSpec validation
coverage checks
```

Long-running or large-model tests MAY be gated separately.

The first E2E suite SHALL remain lightweight.

## Browser Target

The E2E local suite is primarily native.

Browser E2E conformance is not required by this change.

Browser unsupported paths SHALL be structured.

## Tachyon Boundary

The E2E local suite SHALL not require Tachyon.

A future Tachyon conformance suite may validate distributed orchestration.

This suite validates local Magnetar inference only.

## Non-Goals

This change does not:

- require production model accuracy
- require large Qwen model execution
- require GPU hardware
- require CUDA
- require Metal
- require OpenVINO
- require QNN
- require WebGPU
- require network downloads
- define Tachyon distributed conformance
- define HTTP server conformance
- define full CLI UX
- define agent/tool behavior
- define benchmark performance targets
- validate training or fine-tuning
- validate quantized production inference
- validate flash attention
- validate paged attention as required

## Impact

Magnetar gains a phase-closing correctness gate.

After this change, the architecture phase has a concrete proof target:

```text
local fixture
  -> Runtime Inference API
  -> Qwen-like Model Component
  -> Reference CPU Provider
  -> deterministic generated result
```

This provides the bridge from OpenSpec architecture to implementation work.