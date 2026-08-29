## ADDED Requirements
### Requirement: Generation May Use Phase-Specific Prepared Plans

Generation SHALL be able to resolve distinct compatible Plans for prefill and
decode.

#### Scenario: Prefill finishes

Given prefill Plan A and decode Plan B exist

When generation transitions to decode

Then Runtime may switch at safe phase boundary.

---

### Requirement: Decode Uses Bounded Plan Dispatch Path

Normal decode SHALL avoid full Kernel re-selection while compatible Plan
remains ready.

#### Scenario: Hundred decode steps

Given workload remains within Plan guards

When tokens are generated

Then same Plan generation may be reused.

---

### Requirement: Plan Switch Occurs At Safe Generation Boundary

Generation SHALL not replace Plan underneath active Operator invocation.

#### Scenario: Better decode Plan becomes ready

Given current token computation is in flight

When replacement is published

Then new Plan is used only from subsequent safe boundary.

---

### Requirement: Generation Resources Remain Session-Owned

Prepared Plan SHALL not capture one generation Session's KV or output resource
as global Plan state.

#### Scenario: Concurrent conversations

Given both execute same Model Instance

When same decode Plan is reused

Then each binds separate Session resources.
