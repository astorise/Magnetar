# Implementation Plan Runtime Baseline

## Why

Magnetar now has a complete OpenSpec architecture for the first local inference
baseline.

The architecture defines:

- Runtime inference boundary
- CLI boundary
- Component Runtime
- Model Artifact
- Model Loading
- Model Instance lifecycle
- Model Component
- Qwen baseline Component
- Tokenizer
- Session
- Generation
- Sampling
- KV Cache
- Prefix Cache
- Tensor Resource and Layout
- Memory Manager
- Execution Graph
- Operator Contract
- Kernel Contract
- Kernel Registry and Dispatch
- Reference CPU Provider
- End-to-End Local Inference Conformance

The next risk is implementation disorder.

If implementation starts from the CLI or Qwen path directly, too many internal
contracts may be skipped or mocked incorrectly.

This change defines the implementation order, PR boundaries, completion gates,
and anti-bypass rules for the first executable Runtime baseline.

## What Changes

This change introduces an implementation plan for the Runtime baseline.

It defines:

- implementation phases
- recommended PR sequence
- module creation order
- required test gates
- conformance gates
- no-shortcut validation
- expected baseline behavior
- deferred work
- acceptance criteria

The exact branch names and PR numbers are implementation-defined.

## Implementation Principle

Implementation SHALL proceed from lower-level contracts to higher-level
inference API.

The intended order is:

```text
1. Runtime module skeleton and public façade
2. Tensor Resource and Layout
3. Memory Manager integration
4. Operator catalog and first operator scope
5. Reference CPU Provider kernels
6. Kernel Registry and Dispatch
7. Model Artifact, Loading, and Model Instance
8. Tokenizer fixture and contract path
9. Qwen Model Component baseline
10. Generation and Sampling integration
11. Runtime Inference API
12. CLI boundary harness
13. E2E local inference conformance
```

Implementation MAY adjust order only if it does not bypass contract validation.

## PR 1: Runtime Module Skeleton

The first PR SHOULD establish module layout and compile-safe scaffolding.

It SHOULD include:

- module tree
- crate façade re-exports
- error type skeletons
- ID types
- policy placeholder types
- observability event skeletons
- feature flags
- test fixture conventions
- no raw handle public API checks

This PR SHOULD avoid fake execution behavior.

## PR 2: Tensor Resource And Memory Baseline

The second PR SHOULD implement Tensor Resource and Memory Manager baseline.

It SHOULD include:

- TensorDescriptor
- TensorResourceId
- TensorResource metadata
- TensorLayout
- TensorDType
- TensorShape
- Tensor readiness
- Tensor lifecycle
- host memory class
- contiguous layout
- basic allocation tracking
- size accounting
- no raw pointer exposure in public APIs

This PR SHOULD not implement complex Provider-owned memory yet.

## PR 3: Operator Catalog And First Scope

The third PR SHOULD implement the first Operator Catalog scope.

It SHOULD include:

- OperatorId
- Operator metadata
- required-now classification
- placeholder classification
- unsupported classification
- shape validation helpers
- dtype validation helpers
- layout validation helpers
- operator conformance fixtures

This PR SHOULD not depend on CUDA, Metal, OpenVINO, QNN, or WebGPU.

## PR 4: Reference CPU Provider Baseline

The fourth PR SHOULD implement the Reference CPU Provider.

It SHOULD include:

- Reference CPU Provider identity
- CPU Device metadata
- Provider status snapshot
- kernel advertisements
- host contiguous f32 execution path
- first required-now CPU kernels
- structured CPU errors
- CPU conformance fixtures

The first implementation MAY be slow.

Correctness is required before performance.

## PR 5: Kernel Registry And Dispatch

The fifth PR SHOULD implement Kernel Registry and Dispatch.

It SHOULD include:

- KernelAdvertisement validation
- KernelCandidate
- KernelSelectionRequest
- KernelDispatchPlan
- KernelDispatchResult
- candidate filtering
- Resource Affinity validation
- Memory Manager feasibility checks
- Provider/Device readiness checks
- dispatch revalidation
- explicit fallback policy placeholder

This PR SHALL prevent direct Provider execution bypass.

## PR 6: Model Artifact, Loading, And Instance

The sixth PR SHOULD implement Model Artifact, Model Loading, and Model Instance
baseline.

It SHOULD include:

- fixture Model Artifact metadata
- artifact validation
- trust state validation
- tensor inventory validation hooks
- loading request
- loading lifecycle
- ModelInstanceId
- Model Instance readiness
- unload cleanup path

This PR SHOULD use test fixtures before production model formats.

## PR 7: Tokenizer Fixture And Contract Path

The seventh PR SHOULD implement Tokenizer baseline.

It SHOULD include:

- tokenizer fixture
- encode path
- decode path
- streaming decode path
- special token metadata
- tokenizer/model compatibility validation
- raw prompt redaction tests

This PR SHOULD not depend on external tokenizer downloads.

## PR 8: Qwen Model Component Baseline

The eighth PR SHOULD implement Qwen-like Model Component baseline.

It SHOULD include:

- Qwen config validation
- tensor inventory validation
- target modules
- KV cache metadata
- tokenizer compatibility metadata
- prefill graph production
- decode graph production
- required operator scope validation
- no `QwenProvider`
- no direct Kernel/Provider access

The initial baseline MAY use a toy Qwen-like fixture.

## PR 9: Generation And Sampling Integration

The ninth PR SHOULD implement Generation and Sampling integration.

It SHOULD include:

- GenerationRequest
- prefill/decode orchestration
- greedy sampling
- stop condition
- max new tokens
- usage accounting
- cancellation points
- streaming event skeleton
- no Provider-assisted sampling requirement

This PR SHOULD run against Reference CPU and Qwen-like fixture.

## PR 10: Runtime Inference API

The tenth PR SHOULD expose Runtime Inference API.

It SHOULD include:

- model resolution API
- model loading API
- session API
- tokenization API
- generation API
- streaming API
- cancellation API
- diagnostics API
- usage reporting
- handle redaction
- inference-only scope checks

The API SHALL not include workspace, Git, shell, process, secrets, tools, or
agent orchestration.

## PR 11: CLI Boundary Harness

The eleventh PR SHOULD add a minimal CLI boundary harness or tests.

It SHOULD validate:

- CLI sends explicit prompt/context
- Runtime does not read workspace files
- Runtime does not execute tools
- Runtime does not execute shell/process
- Runtime does not execute Git
- Runtime receives no ambient CLI authority
- Runtime errors are preserved

This PR need not finalize CLI UX.

## PR 12: E2E Local Inference Conformance

The twelfth PR SHOULD implement the first E2E local inference conformance suite.

It SHOULD include:

- fixture model
- fixture tokenizer
- fixture artifact
- Reference CPU path
- Qwen baseline graph
- Runtime Inference API entrypoint
- session lifecycle
- generation
- streaming
- diagnostics
- redaction
- failure cases
- machine-readable report
- CI integration

This PR closes the baseline.

## No Shortcut Rule

Implementation SHALL NOT introduce shortcuts that bypass core contracts.

Forbidden shortcuts include:

- direct Provider invocation from tests as the E2E success path
- direct Kernel invocation from Model Component
- Model Artifact validation bypass
- Model Loading bypass
- Model Instance bypass
- Tokenizer bypass for text prompt
- Kernel Registry bypass
- Memory Manager bypass
- Runtime Inference API bypass for E2E
- raw tensor pointer exposure
- raw Provider/Device/Kernel handle exposure
- silent dtype conversion
- silent layout conversion
- silent CPU fallback
- Runtime filesystem access
- Runtime tool execution
- Runtime shell/process execution
- Runtime Git execution

Unit tests may directly test units, but E2E success path SHALL use normal
contracts.

## Acceptance Criteria

The Runtime baseline is accepted when:

- all modules compile
- all public IDs are opaque
- Tensor Resource has no raw pointer public API
- Memory Manager tracks host contiguous tensors
- required-now Operators are classified
- Reference CPU Provider advertises required kernels
- Kernel Registry selects Reference CPU kernels
- Model Loading validates fixture artifact
- Qwen baseline produces validated graphs
- Generation produces deterministic output
- Runtime Inference API exposes one-shot or session inference
- CLI boundary tests prove no ambient authority
- E2E local conformance passes CPU-only
- diagnostics and observability are redacted by default
- OpenSpec validation passes
- coverage gate passes

## Deferred Work

The baseline implementation SHALL defer:

- production model download UX
- large Qwen model execution
- optimized CPU kernels
- SIMD/BLAS acceleration
- CUDA Provider
- Metal Provider
- OpenVINO Provider
- QNN Provider
- WebGPU Provider
- GGUF support
- full quantized inference
- paged attention
- flash attention
- speculative decoding
- beam search
- agent/tool runtime
- production CLI UX
- HTTP server API
- Tachyon distributed conformance

## CI Gates

CI SHOULD run:

```text
cargo fmt
cargo check
cargo clippy
cargo test
wasm32 check where feasible
OpenSpec validation
unit tests
contract tests
Reference CPU conformance
first operator scope conformance
Qwen baseline conformance
Runtime Inference API tests
CLI boundary tests
E2E local inference conformance
coverage validation
```

GPU-dependent checks SHALL not be required for the baseline.

## Observability

Implementation milestones SHOULD emit or test observability for:

- module initialization
- model loading
- Model Instance readiness
- session lifecycle
- tokenization
- generation
- operator planning
- Kernel Registry selection
- Reference CPU dispatch
- memory allocation
- streaming
- cancellation
- errors
- E2E report generation

Observability SHALL remain redacted by default.

## Non-Goals

This change does not:

- implement the code directly
- finalize PR branch names
- define production CLI UX
- define server API
- define Provider ABI implementation
- define GPU implementation
- define optimized kernels
- define full Qwen production support
- define model download behavior
- define agent/tool behavior

## Impact

This change converts the architecture phase into an executable implementation
roadmap.

After this change, implementation can start from the bottom of the Runtime stack
instead of jumping directly to CLI or model-family behavior.

The intended final baseline is:

```text
fixture prompt
  -> Runtime Inference API
  -> fixture Qwen-like model
  -> Qwen baseline graph
  -> required-now Operators
  -> Reference CPU Provider
  -> deterministic output
```