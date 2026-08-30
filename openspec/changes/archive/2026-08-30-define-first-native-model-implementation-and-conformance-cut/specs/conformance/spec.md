## ADDED Requirements
### Requirement: First Native Baseline Has One Authoritative E2E

Conformance SHALL define at least one authoritative deterministic native Qwen
vertical-slice test.

#### Scenario: Fixed prompt

Given fixed fixture and prompt

When test runs

Then expected generated token sequence and structural evidence are produced.

### Requirement: E2E Traversal Evidence Is Mandatory

The authoritative E2E SHALL prove all mandatory layers participated.

#### Scenario: Expected output via shortcut

Given output is correct

But Qwen Component was bypassed

When evidence validates

Then conformance fails.

### Requirement: Model Artifact Traversal Evidence

Conformance SHALL prove Model Artifact was loaded.

#### Scenario: Test constructs weights directly

Given Model Artifact loader was not traversed

When authoritative E2E runs

Then test is non-conformant.

### Requirement: Qwen WASM Traversal Evidence

Conformance SHALL prove Qwen WASM Component was instantiated.

#### Scenario: Native Qwen Rust fallback

Given a Rust model-specific implementation produces output

When Component load evidence is absent

Then authoritative E2E fails.

### Requirement: Registry Traversal Evidence

Conformance SHALL prove mandatory Operators were resolved through Kernel
Registry.

#### Scenario: Direct MatMul call

Given MatMul result is correct

When no Registry resolution exists

Then native E2E fails.

### Requirement: Provider Traversal Evidence

Conformance SHALL prove Reference CPU Provider executed mandatory Kernels.

#### Scenario: E2E invokes Kernel implementation directly

Given Provider execution evidence is absent

When conformance validates

Then test fails.

### Requirement: Candle Exclusion Evidence

Conformance SHALL prove Candle did not execute model forward in native profile.

#### Scenario: Candle available in workspace

Given E2E is executed

When Provider evidence is inspected

Then Candle model execution is absent.

### Requirement: Real KV Evidence

Conformance SHALL prove prefill and decode use the same logical Session KV
state.

#### Scenario: Second decode step

Given prior KV length exists

When next step executes

Then prior state is consumed and new state appended.

### Requirement: Incremental Decode Evidence

Conformance SHALL prove normal decode work is incremental.

#### Scenario: Sequence history length ten

Given one new token is decoded

When instrumentation is inspected

Then model does not process all ten historical tokens as fresh mandatory input.

### Requirement: RoPE Position Evidence

At least one decode test SHALL prove a non-zero sequence position reaches RoPE.

#### Scenario: Position seven

Given decode step at position seven

When RoPE executes

Then position seven semantics are used.

### Requirement: Logits Provenance Evidence

Conformance SHALL prove Sampling receives logits from model execution.

#### Scenario: Sampling called

Given logits exist

When provenance is inspected

Then they originate from Runtime Qwen Plan execution.

### Requirement: Deterministic Token Golden

The baseline SHALL include versioned expected greedy output.

#### Scenario: Repeated clean run

Given same fixture and implementation semantics

When E2E repeats

Then token sequence matches golden.

### Requirement: Structural Evidence Is Redacted

Conformance evidence SHALL not require sensitive Tensor or model contents.

#### Scenario: Evidence artifact

Given E2E completes

When evidence is stored

Then it may contain IDs/digests/counts but not raw weights or KV contents.

### Requirement: Failure Paths Are Part Of Cut

Baseline conformance SHALL include structured failure scenarios, not only happy
path.

#### Scenario: Missing Kernel

Given required Kernel is deliberately unavailable

When model prepares

Then structured failure is returned without bypass.

### Requirement: Minimal Features Are Conformant

First-native profile SHALL pass without deferred advanced subsystems.

#### Scenario: Minimal build

Given generated Kernels, autotuning, multi-Device, and accelerators disabled

When conformance runs

Then baseline remains functional.

### Requirement: Definition Of Done Is Atomic

The baseline SHALL not be declared complete while any mandatory Definition of
Done criterion remains unmet.

#### Scenario: Everything works except incremental KV

Given generation still recomputes history

When milestone is evaluated

Then first native implementation baseline is not established.