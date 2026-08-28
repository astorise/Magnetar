# provider-kernel-compilation Specification

## Purpose
This specification defines the optional Provider Kernel Compilation Capability: capability discovery, source and output format negotiation, compilation target and job lifecycle, cancellation, deadlines, resource limits, isolation and trust separation, AOT-only providers, and compiler diagnostics redaction.
## Requirements
### Requirement: Optional Provider Kernel Compilation Capability

Provider MAY expose an independently versioned Kernel Compilation Capability, and absence of this capability SHALL NOT invalidate the Provider.

#### Scenario: Reference CPU has no compiler

Given Reference CPU Provider supports built-in Kernels only

When capability discovery runs

Then absence of compilation capability does not invalidate the Provider.

---

### Requirement: Capability Advertises Source Formats

Compilation Provider SHALL explicitly advertise accepted Kernel Source Formats.

#### Scenario: WGSL unsupported

Given Provider does not advertise `webgpu:wgsl`

When WGSL compilation is requested

Then request is rejected before compilation.

---

### Requirement: Capability Advertises Output Formats

Compilation Provider SHALL identify the formats it can produce.

#### Scenario: PTX produced

Given compilation converts source to PTX

When result is returned

Then resulting artifact declares PTX format explicitly.

---

### Requirement: Compilation Target Is Portable

Kernel compilation target SHALL identify Provider/Device requirements without
native handles.

#### Scenario: CUDA target

Given Runtime selects CUDA Device

When compilation request is submitted

Then target contains portable Device binding and architecture metadata, not
native CUDA device pointer.

---

### Requirement: Runtime Owns Device Selection

Compilation Provider SHALL compile for the Runtime-selected target rather than
selecting an unrelated Device.

#### Scenario: Two GPUs available

Given Runtime selects GPU 1

When Provider compiles kernel

Then Provider does not silently target GPU 0.

---

### Requirement: Compilation Input Is Explicit Artifact Data

Compilation SHALL consume explicit Kernel Source Artifact data and metadata.

#### Scenario: Source file

Given source originated from filesystem

When compilation reaches Provider

Then Provider receives source artifact bytes rather than arbitrary Runtime-owned
file authority.

---

### Requirement: Compilation Jobs Have Lifecycle

Compilation SHALL have explicit job lifecycle.

#### Scenario: Asynchronous compiler

Given compilation takes multiple seconds

When Runtime submits it

Then it may observe queued/compiling/succeeded states without blocking token
decode.

---

### Requirement: Compilation Deadlines Are Explicit

Compilation deadline support SHALL be declared.

#### Scenario: Policy requires timeout

Given Provider cannot enforce deadline

When policy requires enforceable deadline

Then compilation fails closed.

---

### Requirement: Compilation Cancellation Is Explicit

Cancellation behavior SHALL be advertised and structured.

#### Scenario: Non-interruptible compiler

Given compilation is already running and Provider cannot interrupt it

When cancellation is requested

Then Provider reports cancellation unsupported rather than falsely reporting
success.

---

### Requirement: Compilation Limits Are Enforced

Provider SHALL enforce declared compilation limits.

#### Scenario: Oversized source

Given source exceeds maximum supported size

When submitted

Then compilation is rejected before compiler invocation.

---

### Requirement: Compilation Isolation Is Declared

Provider SHALL declare its compiler isolation model.

#### Scenario: Untrusted source and in-process compiler

Given Runtime policy forbids in-process compilation of untrusted source

When Provider declares `InProcessTrustedCompiler`

Then compilation is denied by policy.

---

### Requirement: Successful Compilation Does Not Grant Trust

Successful compilation SHALL NOT automatically mark output trusted or
qualified.

#### Scenario: Malicious but compilable kernel

Given untrusted kernel compiles successfully

When output artifact is created

Then trust status remains governed by artifact policy.

---

### Requirement: Failed Compilation Is Atomic

Failed compilation SHALL NOT publish ready Compiled Kernel Artifact.

#### Scenario: Compiler crashes

Given compiler emits partial binary then crashes

When job ends

Then partial binary is not eligible for preparation.

---

### Requirement: Compilation Is Cold Path

Normal execution path SHALL not invoke source compiler.

#### Scenario: Token decode

Given required Prepared Kernel is unavailable

When decode executes

Then decode does not synchronously launch compiler.

---

### Requirement: AOT-Only Provider Is Supported

Provider MAY support Compiled Kernel Artifact preparation without source compilation, and doing so SHALL NOT require implementing source compilation.

#### Scenario: Mobile deployment

Given AOT artifact was compiled during build

When deployment Provider loads it

Then Provider can prepare and execute it without implementing source compilation.

---

### Requirement: Compiler Diagnostics Are Redacted

Compiler diagnostics SHALL be structured and redacted by default.

#### Scenario: Compiler output contains local path

Given compiler error contains temporary filesystem path

When Runtime exposes diagnostic

Then path is redacted according to policy.
