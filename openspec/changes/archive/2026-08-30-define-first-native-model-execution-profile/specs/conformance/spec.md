## ADDED Requirements
### Requirement: Full Native E2E Conformance

Conformance SHALL prove a text prompt executes through the complete native
Magnetar first-profile path.

#### Scenario: Deterministic Qwen fixture

Given Qwen fixture Model Artifact and WASM Component

When conformance client generates from a fixed prompt

Then expected deterministic output is produced.

### Requirement: Component Participation Conformance

Conformance SHALL prove Qwen architecture WASM Component participated in the
System Under Test.

#### Scenario: Component unavailable

Given Qwen Component Artifact is removed

When E2E runs

Then test fails rather than falling back to direct CPU model implementation.

### Requirement: Registry Dispatch Conformance

Conformance SHALL prove required Operators resolved through Kernel Registry.

#### Scenario: Registry instrumentation

Given E2E generation runs

When evidence is inspected

Then required Kernel resolutions are present.

### Requirement: Reference CPU Provider Conformance

Conformance SHALL prove Reference CPU Provider executed selected Kernels.

#### Scenario: Provider disabled

Given Reference CPU Provider is unavailable

When first-profile E2E runs

Then inference fails instead of directly executing Kernel bypass.

### Requirement: No Candle Execution Conformance

Mandatory first-profile E2E SHALL not perform model execution through Candle.

#### Scenario: Candle feature enabled in repository

Given native profile test runs

When execution evidence is inspected

Then Candle Provider is absent from required model-forward path.

### Requirement: No Caller Logits Conformance

Conformance SHALL prove model logits are not supplied by client/test callback.

#### Scenario: E2E invocation

Given prompt text is supplied

When logits are produced

Then provenance points to model Plan execution.

### Requirement: No Direct Kernel Test Bypass Conformance

The System Under Test path SHALL not call Reference CPU Kernels directly from
E2E harness.

#### Scenario: Attention execution

Given E2E reaches Attention

When call path is observed

Then Registry/Provider dispatch is traversed.

### Requirement: Real KV Conformance

Conformance SHALL prove decode consumes previously generated KV state.

#### Scenario: Two decode steps

Given first step appends K/V

When second step executes

Then evidence shows prior KV length is consumed.

### Requirement: No Full Sequence Decode Recompute Conformance

Mandatory decode path SHALL not recompute all historical tokens for every new
token.

#### Scenario: Prompt plus three generated tokens

Given next decode occurs

When model input/decode evidence is inspected

Then incremental new-token computation is used according to profile.

### Requirement: RoPE Position Conformance

Conformance SHALL prove non-zero decode position affects RoPE execution.

#### Scenario: Second decode position

Given token position is greater than zero

When RoPE executes

Then correct position is supplied.

### Requirement: Deterministic Golden Conformance

Fixture SHALL produce stable greedy generation evidence.

#### Scenario: Same fixture and prompt

Given two clean runs

When greedy generation executes

Then generated token sequence matches versioned golden result.

### Requirement: Deferred Feature Independence Conformance

First-profile test SHALL succeed with optional advanced subsystems disabled.

#### Scenario: Minimal feature build

Given multi-Device, generated Kernels, autotuning and accelerators are disabled

When first-profile test runs

Then native Qwen CPU execution remains functional.

### Requirement: Security Boundary Conformance

Qwen WASM Component SHALL execute without ambient filesystem/network/native
Provider authority.

#### Scenario: Component requests unauthorized ambient capability

Given inference is running

When unauthorized access is attempted

Then it is denied.

## MODIFIED Requirements
### Requirement: Observability Redaction Conformance

Conformance SHALL prove execution traces and structural evidence contain no
native stream/event handles, Tensor addresses, model weights, KV contents,
Tensor payloads, prompts, secrets, or credentials.

#### Scenario: Failed synchronization trace

Given detailed internal Provider context exists

When trace is exported

Then only safe logical identities remain.

#### Scenario: E2E report produced

Given conformance succeeds

When evidence is inspected

Then safe identifiers/events prove the path without sensitive contents.
