## ADDED Requirements

### Requirement: RoPE Multi-Head Scope

RoPE SHALL support rotating multiple equal-width head blocks of a row in a
single Kernel invocation via an explicit `head_count` attribute, in
addition to the single-block baseline mode `RoPE Scope` already defines.
Runtime SHALL NOT require one Kernel invocation per head when a Provider's
`rope` Kernel advertises multi-head support.

#### Scenario: Multi-head RoPE within scope

Given a graph node rotates Q or K across multiple attention heads in one
row

When first scope validates it

Then a single `rope` Kernel invocation with `head_count` set to the head
count is within scope, and per-head decomposition into separate Kernel
invocations is not required.
