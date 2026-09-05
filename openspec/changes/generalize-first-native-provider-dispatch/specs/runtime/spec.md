## MODIFIED Requirements

### Requirement: Runtime Treats Reference CPU As Normal Provider

Runtime SHALL route Reference CPU execution through Provider, Kernel Registry,
Kernel Dispatch, Memory Manager, and observability contracts.

Runtime SHALL apply this identically to weight materialization and KV cache
commit: both SHALL resolve the Provider binding actually associated with the
Model Instance or prepared plan rather than assuming Reference CPU
unconditionally, falling back to Reference CPU only when no other Provider
binding was resolved.

#### Scenario: CPU dispatch

Given Reference CPU Kernel is selected

When execution runs

Then Runtime creates normal Kernel Dispatch Plan and Invocation.

#### Scenario: Weight materialization uses the resolved Model Instance Provider

Given a Model Instance was created with Resource Affinity binding a non-Reference-CPU Provider

When Runtime materializes that Model Instance's weights

Then weight materialization resolves and writes through the bound Provider, not Reference CPU.

#### Scenario: KV commit uses the Provider resolved for the same generation

Given a generation step's graph execution resolved a non-Reference-CPU Provider binding

When Runtime commits or discards that step's pending KV resources

Then the commit or discard uses the same resolved Provider binding, not Reference CPU.

#### Scenario: No other Provider resolved

Given no Resource Affinity or prepared plan binding names a Provider

When weight materialization or KV commit runs

Then Runtime falls back to Reference CPU, preserving existing behavior.

---

### Requirement: Runtime Prevents Silent CPU Fallback

Runtime SHALL not use Reference CPU fallback silently.

#### Scenario: CUDA unavailable

Given CUDA Kernel is unavailable

When CPU fallback is not permitted

Then Runtime reports failure instead of using Reference CPU.
