## ADDED Requirements

### Requirement: Qwen May Be First Model Component Baseline

Model Component contract SHALL allow a Qwen-like decoder-only baseline as the
first concrete architecture implementation.

#### Scenario: First component

Given Runtime supports first baseline Components

When Qwen-compatible metadata is loaded

Then Qwen Model Component may satisfy architecture implementation.

---

### Requirement: Model Component Baseline Uses Portable Operators

First baseline Model Components SHALL use portable Operator identities.

#### Scenario: Provider-specific operator rejected

Given Qwen Component graph references `cuda.qwen_attention`

When Runtime validates graph

Then validation fails.

---

### Requirement: Model Component Baseline May Be Runtime-Native First

The first Model Component baseline SHALL be permitted to be Runtime-native
before a WASM Component implementation exists.

#### Scenario: Native Qwen implementation

Given WASM Component path is not implemented

When Runtime resolves Qwen support

Then Runtime may use native architecture implementation if policy allows.